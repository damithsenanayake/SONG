import numpy as np

from umap import UMAP
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import issparse, csr_matrix
import scipy as sp
from sklearn.base import BaseEstimator

from song.util import find_spread_tightness, \
    train_for_batch_online, bulk_grow_sans_drifters, bulk_grow_with_drifters, embed_batch_epochs, \
    train_for_batch_batch, sq_eucl_opt, get_closest_for_inputs

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


class SONG(BaseEstimator):

    def __init__(self, n_components=2, n_neighbors=2,
                 lr=1., gamma=None, so_steps = None,
                 spread_factor=0.6,
                 spread=1., min_dist=0.1, ns_rate=5,
                 agility=1., verbose=0,
                 max_age=3,
                 random_seed=1, epsilon=.9, a=None, b=None, final_vector_count=None, dispersion_method = None,
                 online_portion=.0, fvc_growth=0.5, chunk_size = 1000, non_so_rate = 0, low_memory = False, sampled_batches = None, um_min_dist = 0.001, um_lr = 0.01, um_epochs = 11):

        ''' Initialize a SONG to reduce data.

        :param n_components : Dimensionality of the required output. Default set at 2 dimensions

        :param n_neighbors : Neighborhood size of the Neural (Coding Vector) Graph. Keep it low (1 or 2) for clear
         cluster separation. Higher values result in highly connected graphs, providing poor cluster separation. High
         values are more tolerant to noise.

        :param lr : Learning Rate starting value. Set at 1.0. For large data sets, smaller values may be required
        to prevent cluster drifts

        :param gamma : This variable governs the sampling schedule of the algorithm. Higher gamma will force SONG to
         start with smaller sample numbers and progressively reach the full dataset in each self-organizing iteration.
         gamma = 0. will sample the full dataset at each self-organizing iteration.

        :param so_steps : How many self organization steps are required. Higher numbers of so_steps will cause better
        visualization at the cost of running time.

        :param spread_factor : Spread Factor as defined in the work by Alahakoon et al (2000). Values in (0, 1). Higher
        values result in higher resolutions, and finer grain clustering, but inhibits time consumption performance.

        :param verbose : Print out learning process on command line.

        :param agility : A more agile map will adopt the map to new data (growth and all) faster than an inert map.
        Setting agility == 1. will treat all new data as equally weighed to old data. agility = (0, 1].

        :param spread : Hyperparameter to govern how spread-out the visualization needs to be.

        :param min_dist : Hyperparameter to govern how close together neighbors should be.

        :param epsilon : The edge-strength decay constant. Set at a small value for sparse graph generation. For dense
        graphs, use a higher value (close to 1.0 but less than 1.0).

        :param max_age : parameter to define the lowest permitted edge-strength in the graph. Higher the max_age, the
        smaller the minimum edge-strength.

        :param a : constant term for the rational quadratic kernel

        :param b : constant term for the rational quadratic kernel

        :param final_vector_count : the number of expected final coding vectors.

        :param online_portion : the portion of online distance calculation operations. if 1.0 all distances will be
        calculated online for all iterations. If 0.5, the latter half will be calculated batch-wise. Can be (0, 1.].

        :param fvc_growth: In incremental scenarios, what is the expected growth ratio of coding vectors in subsequent
        visualizations.

        '''

        self.max_epochs_per_sample = np.float32(1)
        self.ns_rate = ns_rate
        self.dim = n_components
        self.lrst = lr
        self.map_ratio = 1.5
        self.sf = spread_factor
        self.spread = spread
        self.min_dist = min_dist
        self.n_neighbors = n_neighbors
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(seed=random_seed)
        self.rng_state = self.random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
        self.agility = agility
        self.gamma = gamma
        self.verbose = verbose
        self.epsilon = epsilon
        self.alpha = a
        self.beta = b
        self.fvc = final_vector_count
        self.max_age = max_age
        self.reduced_lr = 1.
        self.prototypes = None
        self.fast_portion = online_portion
        self.min_strength = epsilon ** ((self.dim + self.max_age))
        self.ss = so_steps
        self.non_so_rate = non_so_rate + 1
        self.prot_inc_portion = fvc_growth
        self.low_memory = low_memory
        self.trained = False
        self.dispersion_method = dispersion_method
        self.um_lr = um_lr
        self.um_epochs = um_epochs
        self.um_min_dist = um_min_dist
        self.chunk_size = chunk_size

        if not sampled_batches is None:
            self.sampled_batches = sampled_batches
        else:
            self.sampled_batches = low_memory


    def fit(self, X, reduction = 'PCA'):
        '''
        :param X: The input dataset normalized over the dataset to range [0, 1].
        :param L: If needed, provide labels for the intermediate visualizations. Must be same length as input array and
        same order
        :return: Y : The Mapped Coordinates in the desired output space (default = 2 ).
        '''

        if reduction == 'PCA':
            self.pca = PCA(n_components=np.min([50, X.shape[0], X.shape[1]]) - 1, random_state= self.random_seed) if not issparse(X) else TruncatedSVD(
            n_components=np.min([50, X.shape[0], X.shape[1]])-1, random_state= self.random_seed)
            self.pca.fit(X)
        verbose = self.verbose
        sparse = issparse(X)
        min_dist = self.min_dist
        spread = self.spread

        if self.ss is None:
            self.ss = (X.shape[0]//1000 * (3 if X.shape[0] < 100000 else 2)) if self.low_memory else (30 if X.shape[0] > 100000 else 50)
        self.max_its = self.ss * self.non_so_rate

        min_batch = 1000
        if self.alpha is None and self.beta is None:
            alpha, beta = find_spread_tightness(spread, min_dist)
        else:
            alpha = self.alpha
            beta = self.beta

        if not sparse:
            scale = np.median(np.linalg.norm(X-X.mean(axis=0), axis=1))
        else:
            scale = np.median(sp.sparse.linalg.norm(csr_matrix(X[self.random_state.permutation(X.shape[0])[:1000]] - X.mean(axis=0)), axis=1))
            #if not sparse else sp.sparse.linalg.norm(csr_matrix(X), axis=1)) ** 2.

        if verbose:
            print(f'scaling factor for growth threshold: {scale}')

        if self.sf is None:
            self.sf = np.log(4) / (2 * self.ss)
        thresh_g = -np.log(X.shape[1]) * np.log(self.sf) * (scale)

        so_lr_st = self.lrst
        if self.prototypes is None:
            if not self.fvc is None:
                self.prototypes = self.fvc
            else:
                self.prototypes = int(np.exp(np.log(X.shape[0]) / 1.5))
        ''' Initialize coding vector weights and output coordinate system  '''
        ''' index of the last nearest neighbor : depends on output dimensionality. Higher the output dimensionality, 
                       higher the connectedness '''

        im_neix = self.dim + self.n_neighbors

        X = X.astype(np.float32)
        if self.trained:
            ''' When reusing a trained model, this block will be executed.'''
            W = self.W
            G = self.G
            Y = self.Y
            E_q = self.E_q
            self.prototypes += self.prototypes * self.prot_inc_portion
            self.prototypes = np.int(self.prototypes)
            if verbose:
                print('Using Trained Map')
        else:
            ''' If the model is not already initialized ...'''
            init_size = im_neix
            W = X[self.random_state.choice(X.shape[0], init_size)] + self.random_state.random_sample(
                (init_size, X.shape[1])).astype(np.float32) * 0.0001
            Y = self.random_state.random_sample((init_size, self.dim)).astype(np.float32)
            if verbose:
                print('random map initialization')
                print('Stopping at {} prototypes'.format(self.prototypes))

            ''' Initialize Graph Adjacency and Weight Matrices '''
            G = np.identity(W.shape[0]).astype(np.float32)
            E_q = np.zeros(W.shape[0]).astype(np.float32)

        lrst = self.lrst

        presented_len = X.shape[0]
        soed = 0
        lrdec = 1.
        soeds = np.arange(self.ss)
        sratios = ((soeds) * 1. / (self.ss - 1))

        batch_sizes = (X.shape[0] - min_batch) * ((sratios * 0) if self.low_memory else sratios ** (10) ) + min_batch
        epsilon = self.epsilon
        lr_sigma = np.float32(5)
        drifters = np.array([])
        order = self.random_state.permutation(X.shape[0])
        for i in range(self.max_its):

            order = self.random_state.permutation(X.shape[0]) if not self.sampled_batches else order
            if not i % self.non_so_rate:
                presented_len = int(batch_sizes[soed]) if not self.sampled_batches else 1000
                soed += 1
            X_presented = X[order[:presented_len]] if not self.sampled_batches else X[(presented_len * i)%X.shape[0] : min((presented_len * i)%X.shape[0]+presented_len, X.shape[0])].astype(np.float32)
            if not i % self.non_so_rate:
                non_growing_iter = 0
                shp = np.arange(G.shape[0]).astype(np.int32)

                if self.prototypes >= G.shape[0] and i > 0:

                    ''' Growing of new coding vectors and low-dimensional vectors '''

                    if not len(drifters):
                        W, G, Y, E_q = bulk_grow_sans_drifters(shp, E_q, thresh_g, G, W, Y)
                    else:
                        W, G, Y, E_q = bulk_grow_with_drifters(shp, E_q, thresh_g, drifters, G, W, Y)

                '''shp is an index set used for optimizing search operations'''
                shp = np.arange(G.shape[0], dtype=np.int32)

                if verbose:
                    print(
                        f'\r  |G| = {G.shape[0]}, |X| = {X_presented.shape[0]}, epoch = {i+1}/{self.max_its}, Self-organizing', end='')

                if ((i * 1. / self.max_its < self.fast_portion) or X.shape[0] < 10000) and not sparse:
                    '''ONLINE TRAINING for small datasets '''
                    W, Y, G, E_q = train_for_batch_online(X_presented, i, self.max_its, lrst, lrdec, im_neix, W,
                                                          self.max_epochs_per_sample, G, epsilon, self.min_strength,
                                                          shp, Y,
                                                          self.ns_rate, alpha, beta, self.rng_state, E_q, lr_sigma,
                                                          self.reduced_lr)
                else:
                    '''BATCH TRAINING for large and sparse datasets'''

                    chunk_size =  self.chunk_size
                    n_chunks = X_presented.shape[0]//chunk_size + 1
                    if reduction == 'PCA':
                        W_ = self.pca.transform(W).astype(np.float32)
                    else:
                        W_ = W

                    for chunk in range(n_chunks):
                        chunk_st = chunk * chunk_size
                        chunk_en = chunk_st + chunk_size
                        X_chunk = X_presented[chunk_st:chunk_en].toarray() if sparse else X_presented[chunk_st:chunk_en]
                        if not X_chunk.shape[0]:
                            continue
                        if reduction == 'PCA':
                            X_chunk_ = self.pca.transform(X_presented[chunk_st:chunk_en]).astype(np.float32)
                        else:
                            X_chunk_ = X_chunk

                        pdists = sq_eucl_opt(X_chunk_, W_).astype(np.float32)
                        if verbose:
                            print(f'\r  |G| = {G.shape[0]}, |X| = {X_presented.shape[0]}, epoch = {i+1}/{self.max_its}, Training chunk {chunk + 1} of {n_chunks}', end='')
                        W, Y, G, E_q = train_for_batch_batch(X_chunk, pdists, i, self.max_its, lrst, lrdec, im_neix,
                                                             W,
                                                             self.max_epochs_per_sample, G, epsilon,
                                                             self.min_strength,
                                                             shp, Y,
                                                             self.ns_rate, alpha, beta, self.rng_state, E_q,
                                                             lr_sigma, self.reduced_lr)

                drifters = np.where(G.sum(axis=1) == 0)[0]

            else:
                if verbose:
                    print('\r Training sans SO epoch {} / {} , |X| = {} , |G| = {}  '.format(i + 1, self.max_its, X_presented.shape[0], G.shape[0]), end='')

                repeats = 1 if not soed == self.ss else 1
                for repeat in range(repeats):
                    Y = embed_batch_epochs(Y, G, self.max_its, i, alpha, beta, self.rng_state, self.reduced_lr)
                non_growing_iter += 1



        self.reduced_lr = self.reduced_lr * self.agility
        self.W = W
        self.Y = Y
        self.G = G
        self.E_q = E_q * 0
        self.so_lr_st = so_lr_st
        self.trained = True
        if verbose:
            print('\n Done ...')
        return self

    def batch_train(self, X):
        pass

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X, reduction = 'PCA'):

        '''Adding a PCA-reduction to speed up the transform process'''


        if reduction == 'PCA':

            W_pc = self.pca.transform(self.W).astype(np.float32)
            X_pc = self.pca.transform(X).astype(np.float32)

            if len(X_pc.shape) == 1:
                X_pc = np.array([X_pc])

            min_dist_args = []
            chunk = 1000

            for i in range(X.shape[0] // chunk + 1):
                X_b = X_pc[i * chunk: (i + 1) * chunk]
                min_dist_args.extend(list(get_closest_for_inputs(np.float32(X_b), W_pc)))
        else:
            if len(X.shape) == 1:
                X = np.array([X])

            min_dist_args = []

            chunk = 1000

            for i in range(X.shape[0]//chunk + 1):
                X_b = X[i * chunk : (i+1) * chunk].toarray()
                min_dist_args.extend(list(get_closest_for_inputs(np.float32(X_b), self.W)))

        output = self.Y[min_dist_args]
        Y = output.copy()
        if self.dispersion_method == 'UMAP':
            ''' This step is needed to synchronize with UMAP dispersion'''
            self.Y_loc = Y.min(axis=0)
            self.Y_scale = Y.max(axis=0) - Y.min(axis=0)
            Y_init = 10 * (Y - self.Y_loc) / self.Y_scale
            if Y.shape[0] == 1:
                if self.disp_model == None:
                    raise ValueError('Please ensure that your input is larger than 1 vector')
                else:
                    x_pc = self.pca.transform(X)
                    output = self.disp_model.fit_transform(x_pc)
                    output = ((output) * (self.Y_scale) / 10.) + self.Y_loc

            else:
                self.disp_model = UMAP(learning_rate=self.um_lr, n_components=self.dim, n_epochs=self.um_epochs, init=Y_init, min_dist=self.um_min_dist)
                X_pc = self.pca.fit_transform(X)
                if self.verbose:
                    print('\nDispersing output using UMAP...')
                output = self.disp_model.fit_transform(X_pc)
                output = ((output) * (self.Y_scale) / 10.) + self.Y_loc
            if self.verbose:
                print('\nUMAP dispersion finished!')

        return output



    def get_representatives(self, X, reduction = 'PCA'):

        '''Adding a PCA-reduction to speed up the transform process'''

        if reduction == 'PCA':
            self.pca = PCA(n_components=np.min([50, X.shape[0], X.shape[1]]) - 1) if not issparse(X) else TruncatedSVD(
                n_components=np.min([50, X.shape[0], X.shape[1]]))
            self.pca.fit(X)
            W_pc = self.pca.transform(self.W).astype(np.float32)
            X_pc = self.pca.transform(X).astype(np.float32)

            if len(X_pc.shape) == 1:
                X_pc = np.array([X_pc])

            min_dist_args = []
            chunk = 1000

            for i in range(X.shape[0] // chunk + 1):
                X_b = X_pc[i * chunk: (i + 1) * chunk]
                min_dist_args.extend(list(get_closest_for_inputs(np.float32(X_b), W_pc)))
        else:
            if len(X.shape) == 1:
                X = np.array([X])

            min_dist_args = []

            chunk = 1000

            for i in range(X.shape[0] // chunk + 1):
                X_b = X[i * chunk: (i + 1) * chunk].toarray()
                min_dist_args.extend(list(get_closest_for_inputs(np.float32(X_b), self.W)))

        return min_dist_args

import numpy as np
from umap import UMAP
from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from umap.umap_ import optimize_layout_euclidean
from scipy.sparse import coo_matrix

from song.util import find_spread_tightness, \
    bulk_grow_with_drifters_duplex, embed_batch_epochs, \
    train_for_batch_batch, sq_eucl_opt, get_closest_for_inputs, get_fast_knn

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


class SONG(BaseEstimator):

    def __init__(self, n_components=2, n_neighbors=1,
                 lr=1., gamma=None, so_steps = None, mutable_graph = True,
                 spread_factor=0.9,
                 spread=1., min_dist=0.1, ns_rate=5,
                 agility=1., verbose=0,
                 max_age=3,
                 random_seed=1, epsilon=.9, a=None, b=None, final_vector_count=None, dispersion_method = 'UMAP',
                 fvc_growth=0.5, chunk_size = 1000, pc_components = 50, non_so_rate = 0, low_memory = False, sampled_batches = False, um_min_dist = 0.001, um_lr = 0.01, um_epochs = 11):

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
        self.pca = [None, None]
        self.pc_components = pc_components
        self.mutable_graph = mutable_graph

        if not sampled_batches is None:
            self.sampled_batches = sampled_batches
        else:
            self.sampled_batches = low_memory

    def __repr__(self):
        return f'SONG model: n(coding vectors) = {self.W.shape[0]}, n(dimensions) = {self.W.shape[1]}, n(components) = {self.Y.shape[1]}' if self.trained else f'untrained SONG model'

    def _train_pca(self, X, ix):

        self.pca[ix] = PCA(n_components=np.min([self.pc_components, X.shape[0], X.shape[1]]), random_state = self.random_seed)
        self.pca[ix].fit(X.toarray() if issparse(X) else X)

    def _get_XPCA(self, X, ix):
        nc = np.min([self.pc_components, X.shape[0], X.shape[1]])
        X_pca = np.zeros((X.shape[0], nc))
        chunk_size = 5000
        if issparse(X):
            n_chunks = X.shape[0] // chunk_size + 1

            for i in range(n_chunks):
                st = i * chunk_size
                et = (i + 1) * chunk_size

                X_c = X[st:et].toarray()
                if X_c.shape[0]:
                    X_pca[st:et] = self.pca[ix].transform(X_c)

        else:

            X_pca = self.pca[ix].transform(X)

        return X_pca

    def fit(self, X, reduction = 'PCA', corrected_PC = np.array([])):
        '''
        :param X: The input dataset normalized over the dataset to range [0, 1].
        :param L: If needed, provide labels for the intermediate visualizations. Must be same length as input array and
        same order
        :return: Y : The Mapped Coordinates in the desired output space (default = 2 ).
        '''
        X[0] = X[0].astype(np.float32)
        X[1] = X[1].astype(np.float32)
        X_PCA = [None, None]

        for set_ix in range(2):
            if reduction == 'PCA' and X[set_ix].shape[1] > self.pc_components:
                if not corrected_PC.shape[0]:
                    if self.verbose:
                        print('Fitting Reduction for Neighborhood Function Calculation')
                    self._train_pca(X[set_ix][self.random_state.permutation(X[set_ix].shape[0])[:10000]], set_ix)

                    if self.verbose:
                        print ('reduction model fitted!')
                if not corrected_PC.shape[0]:

                    X_PCA[set_ix] = self._get_XPCA(X[set_ix], set_ix)
                else:
                    X_PCA[set_ix] = corrected_PC
            else:
                X_PCA[set_ix] = X[set_ix]


        verbose = self.verbose
        min_dist = self.min_dist
        spread = self.spread

        min_batch = 1000 if not self.sampled_batches else 10000




        if self.alpha is None and self.beta is None:
            alpha, beta = find_spread_tightness(spread, min_dist)
            self.alpha = alpha
            self.beta = beta
        else:
            alpha = self.alpha
            beta = self.beta


        if self.sf is None:
            self.sf = np.log(4) / (2 * self.ss)


        so_lr_st = self.lrst
        if self.prototypes is None:
            if not self.fvc is None:
                self.prototypes = self.fvc
            else:
                self.prototypes = int(np.exp(np.log(min(X[0].shape[0], X[1].shape[0])) / 1.5))
        ''' Initialize coding vector weights and output coordinate system  '''
        ''' index of the last nearest neighbor : depends on output dimensionality. Higher the output dimensionality, 
                       higher the connectedness '''

        im_neix = self.dim + self.n_neighbors



        if self.ss is None:
            if X[0].shape[0] < 100000 or not self.sampled_batches:
                self.ss = 100  # (20 if X.shape[0] > 100000 else 20)
            else:
                self.ss = X[0].shape[0] // min_batch + 1


        self.max_its = self.ss * self.non_so_rate

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
            W = [None, None]
            E_q = [None, None]

            Y = self.random_state.random_sample((init_size, self.dim)).astype(np.float32)
            if verbose:
                print('random map initialization')
                print('Stopping at {} prototypes'.format(self.prototypes))

            ''' Initialize Graph Adjacency and Weight Matrices '''
            for set_ix in range(2):
                W[set_ix] = X[set_ix][self.random_state.choice(X[set_ix].shape[0],
                                                               init_size)] + self.random_state.random_sample(
                    (init_size, X[set_ix].shape[1])).astype(np.float32) * 0.0001
                E_q[set_ix] = np.zeros(W[set_ix].shape[0]).astype(np.float32)
            G = np.identity(W[0].shape[0]).astype(np.float32)

        order = [None, None]
        presented_len = [None, None]
        sratios = [None, None]
        batch_sizes = [None, None]
        soeds = np.arange(self.ss)
        soed = 0
        lrdec = 1.
        lrst = self.lrst

        for set_ix in range(2):
            error_scale = np.median(np.linalg.norm(
                X - X.mean(axis=0) if not (reduction == 'PCA') else X_PCA[set_ix] - X_PCA[set_ix].mean(axis=0),
                axis=1)) ** 2
            thresh_g = -(X[set_ix].shape[1]) if not (reduction == 'PCA') else -(X_PCA[set_ix].shape[1]) * np.log(
                self.sf) * error_scale

            order[set_ix] = self.random_state.permutation(X[set_ix].shape[0])
            presented_len[set_ix] = X[set_ix].shape[0]
            sratios[set_ix] = ((soeds) * 1. / (self.ss - 1))

            batch_sizes[set_ix] = (X[set_ix].shape[0] - min_batch) * (
                    (sratios[set_ix] * 0) if self.low_memory else sratios[set_ix] ** (100)) + min_batch
        sratios[1][1:] = sratios[0][:-1]

        epsilon = self.epsilon
        lr_sigma = np.float32(5)
        drifters = np.array([])

        split_ratio = (np.log(X[0].shape[0])//np.log(X[1].shape[0])) + 1
        for i in range(self.max_its):
            set_ix = i%2#int(i%split_ratio > 0)
            sparse = issparse(X[set_ix])

            order = self.random_state.permutation(X[set_ix].shape[0]) if not self.sampled_batches else order
            if not i % self.non_so_rate:
                presented_len = int(batch_sizes[set_ix][soed]) if not self.sampled_batches else min_batch
                soed += 1
            X_presented = X[set_ix][order[:presented_len]] if not self.sampled_batches else X[set_ix][(presented_len * i)%X[set_ix].shape[0] : min((presented_len * i)%X[set_ix].shape[0]+presented_len, X[set_ix].shape[0])].astype(np.float32)
            X_presented_pc = X_PCA[set_ix][order[:presented_len]] if not self.sampled_batches else X_PCA[set_ix][(presented_len * i)%X[set_ix].shape[0] : min((presented_len * i)%X[set_ix].shape[0]+presented_len, X[set_ix].shape[0])].astype(np.float32)

            if not i % self.non_so_rate:
                non_growing_iter = 0
                shp = np.arange(G.shape[0]).astype(np.int32)

                if self.mutable_graph and self.prototypes >= G.shape[0] and i > 0:
                    ''' Growing of new coding vectors and low-dimensional vectors '''
                    W_ret_0, W_ret_1, G, Y, E_q_ret = bulk_grow_with_drifters_duplex(shp, np.array(E_q), set_ix, thresh_g, drifters, G, W[0], W[1], Y)
                    W[0] = W_ret_0
                    W[1] = W_ret_1
                    E_q[0] = E_q_ret[0]
                    E_q[1] = E_q_ret[1]
                '''shp is an index set used for optimizing search operations'''
                shp = np.arange(G.shape[0], dtype=np.int32)

                chunk_size =  self.chunk_size if i else 100
                n_chunks = X_presented.shape[0]//chunk_size + 1
                if reduction == 'PCA' and X[set_ix].shape[1] > 100:
                    W_ = self.pca[set_ix].transform(W[set_ix]).astype(np.float32)
                else:
                    W_ = W[set_ix]

                for chunk in range(n_chunks):
                    chunk_st = chunk * chunk_size
                    chunk_en = chunk_st + chunk_size
                    X_chunk = X_presented[chunk_st:chunk_en].toarray() if sparse else X_presented[chunk_st:chunk_en]
                    if not X_chunk.shape[0]:
                        continue
                    if reduction == 'PCA':
                        X_chunk_ = (X_presented_pc[chunk_st:chunk_en]).astype(np.float32)
                    else:
                        X_chunk_ = X_chunk

                    pdists = sq_eucl_opt(X_chunk_, W_).astype(np.float32)
                    if verbose:
                        print(f'\r split ratio: {split_ratio} set:{set_ix},  |G| = {G.shape[0]}, |X| = {X_presented.shape[0]}, epoch = {i+1}/{self.max_its}, Training chunk {chunk + 1} of {n_chunks}', end='')
                    W_ret, Y, G, E_q_ret = train_for_batch_batch(X_chunk, pdists, i, self.max_its, lrst, lrdec, im_neix,
                                                         W[set_ix],
                                                         self.max_epochs_per_sample, G, epsilon,
                                                         self.min_strength,
                                                         shp, Y,
                                                         self.ns_rate, alpha, beta, self.rng_state, E_q[set_ix],
                                                         lr_sigma, self.reduced_lr)
                    W[set_ix] = W_ret
                    E_q[set_ix] = E_q_ret
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
        self.E_q = E_q
        self.so_lr_st = so_lr_st
        self.trained = True
        if verbose:
            print('\n Done ...')
        return self

    def batch_train(self, X):
        pass

    def fit_transform(self, X, reduction = 'PCA', corrected_PC = np.array([])):
        self.fit(X, reduction, corrected_PC)
        return self.transform(X, reduction)

    def transform(self, X, reduction = 'PCA', corrected_PC = np.array([])):
        output = [None, None]
        '''Adding a PCA-reduction to speed up the transform process'''
        min_dist_args, _, X_pc, W_pc = self.get_representatives(X, reduction=reduction, corrected_PC=corrected_PC)

        for set_ix in range(2):

            output[set_ix] = self.Y[min_dist_args[set_ix]]

            Y = output[set_ix]
            if self.dispersion_method == 'UMAP':
                Y += self.random_state.random_sample(Y.shape) * 0.001
                output[set_ix] = self.get_umap_dispersion(Y, X_pc[set_ix].astype(np.float32), W_pc[set_ix].astype(np.float32))

        return output

    def get_umap_dispersion(self, Y_init, X_pc, W_pc):

        if self.verbose:
            print('constructing graph')
        # values, rows, cols = get_fast_knn(X_pc,  W_pc, 15, self.G)
        #
        # epochs_per_sample = np.exp(-values) * 10
        # if self.verbose:
        #     print('optimizing layout')
        # Y = optimize_layout_euclidean(Y_init, Y_init.copy(), rows, cols, 20, X_pc.shape[0], epochs_per_sample, self.alpha, self.beta, self.rng_state, initial_alpha=0.01)
        self.Y_loc = Y_init.min(axis=0)
        self.Y_scale = Y_init.max(axis=0) - self.Y_loc

        Y = UMAP(init=Y_init, min_dist=self.min_dist, n_components= self.dim, spread= self.spread, learning_rate=0.01, n_epochs=11).fit_transform(X_pc)
        if self.verbose:
            print('transformation done...')
        return (Y * self.Y_scale / 10.) + self.Y_loc


    def get_representatives(self, X, reduction = 'PCA', corrected_PC = np.array([])):
        min_dist_args = [None, None]
        min_dist_vals = [None, None]
        X_pc = [None, None]
        W_pc = [None, None]
        for set_ix in range(2):
            min_dist_args[set_ix] = []
            min_dist_vals[set_ix] = []
            if reduction == 'PCA' and X[set_ix].shape[1] > self.pc_components:

                W_pc[set_ix] = self._get_XPCA(self.W[set_ix], set_ix).astype(np.float32)
                if not corrected_PC.shape[0]:
                    X_pc[set_ix] = self._get_XPCA(X[set_ix], set_ix )
                else:
                    X_pc[set_ix] = corrected_PC
                if len(X_pc[set_ix].shape) == 1:
                    X_pc[set_ix] = np.array([X_pc[set_ix]])


                chunk = 1000
                n_chunks = X[set_ix].shape[0] // chunk + 1
                for i in range(n_chunks):
                    if self.verbose:
                        print(f'\rgetting PC of chunk {i+1}/{n_chunks}', end='')
                    X_b = X_pc[set_ix][i * chunk: (i + 1) * chunk]

                    min_dist_pos, min_dist_val = get_closest_for_inputs(np.float32(X_b), W_pc[set_ix])

                    min_dist_args[set_ix].extend(list(min_dist_pos))
                    min_dist_vals[set_ix].extend(list(min_dist_val))
                if self.verbose:
                    print('')
            else:
                X_pc[set_ix] = X[set_ix]
                W_pc[set_ix] = self.W[set_ix]
                if len(X[set_ix].shape) == 1:
                    X = np.array([X[set_ix]])

                chunk = 1000
                n_chunks = X[set_ix].shape[0] // chunk + 1
                for i in range(n_chunks):
                    if self.verbose:
                        print(f'\rgetting PC of chunk {i+1}/{n_chunks}', end='')
                    X_b = X[set_ix][i * chunk: (i + 1) * chunk].toarray() if issparse(X[set_ix]) else X[set_ix][i * chunk: (i + 1) * chunk]
                    min_dist_pos, min_dist_val = get_closest_for_inputs(np.float32(X_b), self.W[set_ix])
                    min_dist_args.extend(list(min_dist_pos))
                    min_dist_vals.extend(list(min_dist_val))
                if self.verbose:
                    print('')
        return min_dist_args, min_dist_vals, X_pc, W_pc

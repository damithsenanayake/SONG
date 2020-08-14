import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin
# from scipy.spatial.distance import  cdist
from sklearn.manifold import SpectralEmbedding
from sklearn.base import BaseEstimator
from util import find_spread_tightness, \
    train_for_input, train_for_batch_online, bulk_grow_sans_drifters, embed_batch_epochs, \
    train_for_batch_batch, sq_eucl_opt

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


class SONG(BaseEstimator):

    def __init__(self, out_dim=2, n_neighbors=3,
                 lr=1., spread_factor=0.9,
                 spread=1., min_dist=0.1, ns_rate=5,
                 gamma=None,
                 init=None,
                 agility=0.8, verbose=0,
                 max_age=1,
                 random_seed=1, epsilon=1e-10, a=None, b=None, final_vector_count=None):
        ''' Initialize a SONG to reduce data.

        === General Hyperparameters ===
        :param out_dim : Dimensionality of the required output. Default set at 2 dimensions

        :param n_neighbors : Neighborhood size of the Neural (Coding Vector) Graph. Keep it low (1 or 2) for clear
         cluster separation. Higher values result in highly connected graphs, providing poor cluster separation. High
         values are more tolerant to noise.

        :param lr : Learning Rate starting value. Set at 1.0. For large data sets, smaller values may be required
        to prevent cluster drifts

        :param gamma : This variable governs the growth rate of the map. 0 provides fastest growth, which may result in
        noisy representations. Default set at 2 .

        :param so_steps : How many self organization steps are required. Higher
        :param its : Number of learning iterations (Including the self-organizing steps)

        :param spread_factor : Spread Factor as defined in the work by Alahakoon et al (2000). Values in (0, 1). Higher
        values result in higher resolutions, and finer grain clusterings, but inhibits time consumption performance.

        :param lr_decay_const : Learning rate decay governed by this hyperparameter. set at 1 for a linear decay. 2
        provides a quadratic decay and so on.

        :param verbose : Print out learning process on command line.


        === Incremental Learning Hyperparameters ===
        :param agility : A more agile map will adopt the map to new data (growth and all) faster than an inert map.
        Setting agility == 1. will treat all new data as equally weighed to old data. agility = (0, 1].

        === Visualization Hyperparameters ===
        :param spread : Hyperparameter to govern how spread-out the visualization needs to be.

        :param min_dist : Hyperparameter to govern how close together neighbors should be.

        '''

        self.ns_rate = ns_rate
        self.dim = out_dim
        self.lrst = lr
        self.map_ratio = 1.5
        self.sf = spread_factor
        self.spread = spread
        self.min_dist = min_dist
        self.n_neighbors = n_neighbors
        self.init = init
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

    def fit(self, X):
        '''
        :param X: The input dataset normalized over the dataset to range [0, 1].
        :param L: If needed, provide labels for the intermediate visualizations. Must be same length as input array and
        same order
        :return: Y : The Mapped Coordinates in the desired output space (default = 2 ).
        '''
        if X.__class__.__name__ == 'csr_matrix':
            sparse = True
        verbose = self.verbose
        min_dist = self.min_dist
        spread = self.spread
        self.ss = 25
        min_batch = 10
        self.max_its = self.ss * 2
        if self.alpha == None and self.beta == None:
            alpha, beta = find_spread_tightness(spread, min_dist)
        else:
            alpha = self.alpha
            beta = self.beta
        scale = np.median(np.linalg.norm(X, axis=1)) ** 2.
        if verbose:
            print("alpha : {} beta : {}".format(alpha, beta))
        if self.sf == None:
            self.sf = np.log(4) / (2 * self.ss)
        thresh_g = -np.log(X.shape[1]) * np.log(self.sf) * (scale)

        so_lr_st = self.lrst
        print('prototypes {}'.format(self.prototypes))
        if not self.fvc == None and self.prototypes == None:
            self.prototypes = self.fvc
        else:
            self.prototypes = int(np.exp(np.log(X.shape[0])/1.5))
        if verbose:
            print('starting lr for self organization : ' + str(so_lr_st))
        ''' Initialize coding vector weights and output coordinate system  '''
        ''' index of the last nearest neighbor : depends on output dimensionality. Higher the output dimensionality, 
                       higher the connectedness '''

        im_neix = self.dim + self.n_neighbors

        X = X.astype(np.float32)
        self.max_epochs_per_sample = np.float32(1)
        try:
            ''' When reusing a trained model, this block will be executed.'''
            W = self.W
            G = self.G
            Y = self.Y
            E_q = self.E_q

            if verbose:
                print('Using Trained Map')
        except:
            ''' If the model is not already initialized ...'''
            if self.init == None:
                init_size = im_neix
                W = X[self.random_state.choice(X.shape[0], init_size)] + self.random_state.random_sample(
                    (init_size, X.shape[1])).astype(np.float32) * 0.0001
                Y = self.random_state.random_sample((init_size, self.dim)).astype(np.float32)  # / self.min_dist
                if verbose:
                    print('random map initialization')

            elif self.init == 'Spectral':
                W = X[:6000].astype(np.float32)
                '''Initializing with Spectral Embedding '''
                Y = SpectralEmbedding(n_components=self.dim).fit_transform(W).astype(np.float32)

            ''' Initialize Graph Adjacency and Weight Matrices '''
            G = np.identity(W.shape[0]).astype(np.float32)  # np.zeros((W.shape[0], W.shape[0])).astype(np.float32)
            E_q = np.zeros(W.shape[0]).astype(np.float32)

        lrst = self.lrst

        presented_len = X.shape[0]
        soed = 0
        lrdec = 1.
        soeds = np.arange(self.ss)
        sratios = ((soeds) * 1. / (self.ss - 1))
        batch_sizes = (X.shape[0] - min_batch) * sratios ** np.log10(X.shape[0]+100) + min_batch
        epsilon = self.epsilon
        lr_sigma = np.float32(np.log10(X.shape[0]) / 2.)
        non_growing_iter = 0
        for i in range(self.max_its):
            order = self.random_state.permutation(X.shape[0])
            self.min_strength = epsilon ** ((self.dim + self.max_age))
            so_iter = 0 if self.ss <= self.max_its else 1
            if not i % 2:
                presented_len = int(batch_sizes[soed])
                soed += 1
                so_iter = 1
            X_presented = X[order[:presented_len]]
            non_growing_iter += 1
            if so_iter:
                non_growing_iter = 0
                k = 0
                shp = np.arange(G.shape[0]).astype(np.int32)

                if not self.prototypes == None and self.prototypes >= G.shape[0]:
                    W, G, Y, E_q = bulk_grow_sans_drifters(shp, E_q, thresh_g, G, W, Y, X_presented)
                shp = np.arange(G.shape[0], dtype=np.int32)


                if E_q.sum() == 0:
                    for x in X_presented:
                        W, Y, G, E_q, k, b, neilist, neighbors, lr = train_for_input(x, X_presented, i, k, self.max_its,
                                                                                     lrst, lrdec, im_neix, W,
                                                                                     self.max_epochs_per_sample,
                                                                                     G, epsilon, self.min_strength, shp,
                                                                                     Y, self.ns_rate, alpha, beta,
                                                                                     self.rng_state, E_q, lr_sigma, self.reduced_lr)

                        if not np.mod(k, 500) and verbose:
                            print(
                                '\r |G|= %i , alpha: %.4f, beta: %.5f , iter = %i , X_i = %i/%i, Neighbors: %i, lr : %.4f' % (
                                    G.shape[0], alpha, beta, i + 1, k, X_presented.shape[0], neighbors.shape[0],
                                    float(lr)), end='')

                else:
                    '''(X_presented, i, max_its,  lrst, lrdec, im_neix, W, order, G, epsilon, min_strength, shp, Y, ns_rate, alpha, beta, rng_state, E_q)'''
                    if verbose:
                        print(
                            '\r Training with Self Organization for all presented inputs in this batch i = {} , |X| = {} , |G| = {}  '.format(
                                i + 1, presented_len, G.shape[0]), end='')
                    if (i * 1./self.max_its <= 0.8 or X.shape[0] < 10000) and presented_len < 100000:
                        W, Y, G, E_q = train_for_batch_online(X_presented, i, self.max_its, lrst, lrdec, im_neix, W,
                                                          self.max_epochs_per_sample, G, epsilon, self.min_strength, shp, Y,
                                                          self.ns_rate, alpha, beta, self.rng_state, E_q, lr_sigma, self.reduced_lr)

                    else:
                        too_big = False
                        try:
                            pdists = sq_eucl_opt(X_presented, W).astype(np.float32)
                        except MemoryError:
                            too_big = True

                        if not too_big:
                            W, Y, G, E_q = train_for_batch_batch(X_presented, pdists, i, self.max_its, lrst, lrdec, im_neix, W,
                                                              self.max_epochs_per_sample, G, epsilon, self.min_strength,
                                                              shp, Y,
                                                              self.ns_rate, alpha, beta, self.rng_state, E_q, lr_sigma, self.reduced_lr)

                        else:
                            chunk_size = 1000
                            chunks = X_presented.shape[0] // chunk_size

                            for chunk in range(chunks + 1):
                                chunk_st = chunk * chunk_size
                                chunk_en = chunk_st + chunk_size
                                X_chunk = X_presented[chunk_st:chunk_en]
                                pdists = sq_eucl_opt(X_chunk, W).astype(np.float32)
                                W, Y, G, E_q = train_for_batch_batch(X_chunk, pdists, i, self.max_its, lrst, lrdec, im_neix, W,
                                                                     self.max_epochs_per_sample, G, epsilon, self.min_strength,
                                                                     shp, Y,
                                                                     self.ns_rate, alpha, beta, self.rng_state, E_q, lr_sigma, self.reduced_lr)


            else:
                embed_length = len(X_presented)
                G[G < self.min_strength] = 0
                if verbose:
                    print(
                        '\rTraining iteration %i without self organization :  Map size : %i, SOED : %i , Batch Size : %i' % (
                            i + 1, G.shape[0], soed, embed_length), end='')
                repeats = 1 if not soed == self.ss else 1
                for repeat in range(repeats):
                    Y = embed_batch_epochs(Y, G, self.max_its, i, alpha, beta, self.rng_state, self.reduced_lr)
                non_growing_iter += 1

        self.reduced_lr =self.reduced_lr * self.agility
        self.W = W
        self.Y = Y
        self.G = G
        self.E_q = E_q * 0
        self.so_lr_st = so_lr_st
        if verbose:
            print('\n Done ...')
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        min_dist_args = pairwise_distances_argmin(X, self.W)
        return self.Y[min_dist_args]

    def get_representatives(self, X):
        min_dist_args = pairwise_distances_argmin(X, self.W)
        return min_dist_args

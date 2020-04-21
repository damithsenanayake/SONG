import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.manifold import SpectralEmbedding
from sklearn.base import BaseEstimator
from util import  find_spread_tightness, train_for_input, train_for_batch, bulk_grow, embed_batch_epochs

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


class SONG(BaseEstimator):

    def __init__(self, out_dim=2, n_neighbors=1,
                 lr=1., spread_factor=0.9,
                 spread=1., min_dist=0.1, ns_rate=5,
                 gamma=None,
                 init=None,
                 agility=0.8, verbose=0,
                 random_seed=1, epsilon=0.05, a = None, b = None, final_vector_count = 20000):
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
        self.prototypes = final_vector_count

    def fit(self, X):
        '''
        :param X: The input dataset normalized over the dataset to range [0, 1].
        :param L: If needed, provide labels for the intermediate visualizations. Must be same length as input array and
        same order
        :return: Y : The Mapped Coordinates in the desired output space (default = 2 ).
        '''
        verbose = self.verbose
        min_dist = self.min_dist
        spread = self.spread
        self.ss = 10
        self.max_its = self.ss * 10
        if self.alpha == None and self.beta == None :
            alpha, beta = find_spread_tightness(spread, min_dist)
        else:
            alpha = self.alpha
            beta = self.beta
        scale = np.median(np.linalg.norm(X, axis=1))**2.
        if verbose:
            print("alpha : {} beta : {}".format(alpha, beta))
        if self.sf == None:
            self.sf = np.log(4) / (2 * self.ss)
        thresh_g = -X.shape[1] * np.log(self.sf) * (scale)

        so_lr_st = self.lrst#self.so_lr_st

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
                W = X[self.random_state.choice(X.shape[0], init_size)] + self.random_state.random_sample((init_size, X.shape[1])).astype(np.float32)*0.0001
                Y = self.random_state.random_sample((init_size, self.dim)).astype(np.float32) #/ self.min_dist
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

        presented_len = 0

        order = self.random_state.permutation(X.shape[0])

        soed = 0

        t_normalized = np.arange(self.max_its).astype(float)
        t_normalized /= (t_normalized.max())

        step_size = self.max_its *1. / self.ss

        t_ixs = (np.arange(self.ss) * step_size).astype(np.int)
        lrdec = 1

        if self.gamma == None:
            gamma = - np.log(min(500., X.shape[0]) * 1. / X.shape[0])
        else:
            gamma = self.gamma
        so_sched = np.unique((t_normalized[t_ixs] * self.max_its).astype(int))[:self.ss]
        soeds = np.arange(self.ss)
        sratios = 1 - ((soeds) * 1. / (self.ss - 1))
        batch_sizes = X.shape[0] * np.exp(-gamma * sratios**2)
        epsilon = self.epsilon
        self.min_strength = epsilon ** (self.dim + 2)
        no_so_ss = so_sched[1]-so_sched[0]
        next_so = so_sched[soed]
        non_growing_iter = 0
        drifters = []

        for i in range(self.max_its):

                so_iter = 0 if self.ss <= self.max_its else 1

                if i == next_so :
                    order = self.random_state.permutation(X.shape[0])

                    presented_len = int(batch_sizes[soed])

                    soed += 1

                    if soed  < self.ss:
                        try:
                            next_so = so_sched[soed]
                        except:
                            pass
                    so_iter = 1

                growing_iter = i == 0


                X_presented = X[order[:presented_len]]
                non_growing_iter += 1
                if so_iter:
                    non_growing_iter = 0

                    k = 0

                    shp = np.arange(G.shape[0]).astype(np.int32)
                    if self.prototypes == None:
                        max_nodes = 2000
                    else:
                        max_nodes = self.prototypes
                    if not self.prototypes == None and self.prototypes >= G.shape[0]:
                        W, G, Y, E_q, drifters = bulk_grow(shp, E_q, thresh_g, drifters, G, W, Y, X_presented)
                    shp = np.arange(G.shape[0])

                    if E_q.sum() == 0 :

                        for x in X_presented:

                            W, Y, G, E_q, k, b, neilist, neighbors, lr = train_for_input(x, X_presented, i, k, self.max_its, lrst, lrdec, im_neix, W, self.max_epochs_per_sample,
                                                G, epsilon, self.min_strength, shp, Y, self.ns_rate, alpha, beta, self.rng_state, E_q)

                            if (growing_iter or X_presented.shape[0] < 2500) and G.shape[0] < max_nodes and soed < self.ss:
                                growth_size = 1
                                if G.shape[0] < presented_len and thresh_g <= E_q[b]:
                                    oldG = G
                                    oldW = W
                                    oldY = Y
                                    oldE = E_q
                                    closests = neilist
                                    W_n = W[closests].sum(axis=0) / len(closests)
                                    Y_n = Y[closests].sum(axis=0) / len(closests)
                                    W = np.zeros((W.shape[0] + growth_size, W.shape[1]), dtype=np.float32)
                                    W[:-growth_size] = oldW
                                    W[-growth_size:] = W_n
                                    Y = np.zeros((Y.shape[0] + growth_size, Y.shape[1]), dtype=np.float32)
                                    Y[: -growth_size] = oldY

                                    Y[-growth_size:] = Y_n
                                    G = np.zeros((G.shape[0] + growth_size, G.shape[1] + growth_size), dtype=np.float32)
                                    G[:-growth_size][:, :-growth_size] = oldG
                                    ''' connect neighbors to the new node '''
                                    G[-growth_size:][:, b] = 1
                                    G[b][-growth_size:] = 1
                                    G[-growth_size:][:, closests] = 1
                                    G[closests][-growth_size:] = 1
                                    G[b][closests] = 0
                                    G[closests][:, b] = 0
                                    ''' Append new error. '''
                                    E_q = np.zeros(E_q.shape[0] + growth_size, dtype=np.float32)
                                    E_q[:-growth_size] = oldE
                                    E_q[-growth_size:] = 0
                                    E_q[closests] *= 0.5
                                    shp = np.arange(G.shape[0]).astype(np.int32)
                                    if G.shape[0] >= 2000:
                                        continue
                            if not np.mod(k, 500) and verbose:
                                print('\r |G|= %i , alpha: %.4f, beta: %.5f , iter = %i , X_i = %i/%i, Neighbors: %i, lr : %.4f' % (
                                    G.shape[0], alpha, beta, i + 1, k, X_presented.shape[0], neighbors.shape[0],
                                    float(lr))),

                    else:
                        '''(X_presented, i, max_its,  lrst, lrdec, im_neix, W, order, G, epsilon, min_strength, shp, Y, ns_rate, alpha, beta, rng_state, E_q)'''
                        if verbose :
                            print ('\r Training with Self Organization for all presented inputs in this batch i = {} , |X| = {} , |G| = {}  '.format(i+1, presented_len, G.shape[0] ), end='')
                        W, Y, G, E_q = train_for_batch(X_presented, i, self.max_its, lrst, lrdec, im_neix, W, self.max_epochs_per_sample, G, epsilon, self.min_strength, shp, Y, self.ns_rate, alpha, beta, self.rng_state, E_q)

                    drifters = shp[(G > 0).sum(axis=0) < im_neix]
                    skip = 0
                else:
                    if not skip > 0:
                        skip += 1
                        embed_length = len(X_presented)
                        if verbose:
                            print('\rTraining iteration %i without self organization :  Map size : %i, SOED : %i , Batch Size : %i' % (
                            i + 1, G.shape[0], soed, embed_length), end='')

                        Y = embed_batch_epochs(lrst, Y, G, self.max_its, i, no_so_ss + i, alpha, beta, self.rng_state)

                        non_growing_iter += 1

        self.lrst = self.lrst * self.agility
        self.W = W
        self.Y = Y
        self.G = G
        self.E_q = E_q *0
        self.so_lr_st = so_lr_st
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        min_dist_args = pairwise_distances_argmin(X, self.W)
        return self.Y[min_dist_args]

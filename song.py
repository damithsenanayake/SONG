import numpy as np
from sklearn.metrics.pairwise import  pairwise_distances_argmin
from sklearn.manifold import SpectralEmbedding
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from util import  bcolors
from util import train_neighborhood, grow_map_at_node, find_spread_tightness,  embed_batch, numba_ind_pairwise

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1



class SONG(BaseEstimator):

    def __init__(self, out_dim=2, n_neighbors = 1,
                 lr=1., its = 20, spread_factor = 0.9,
                 spread = 1., min_dist = 0.1, ns_rate = 5,
                 push_strength = 1., gamma=2,
                 init=None, so_steps = 8,
                 lr_decay_const = 1., agility = 0.5, verbose = 0):
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
        self.max_its = its
        self.sf = spread_factor
        self.spread = spread
        self.min_dist = min_dist
        self.n_neighbors = n_neighbors
        self.repulsion = push_strength*10
        self.init = init
        self.lrdec = lr_decay_const
        self.ss = so_steps
        random_state = np.random.RandomState()
        self.rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
        self.agility = agility
        self.gamma = gamma
        self.verbose = verbose



    def fit(self, X, L=np.array([])):
        '''
        :param X: The input dataset normalized over the dataset to range [0, 1].
        :param L: If needed, provide labels for the intermediate visualizations. Must be same length as input array and
        same order
        :return: Y : The Mapped Coordinates in the desired output space (default = 2 ).
        '''
        verbose = self.verbose
        ''' Initialize Weight and Coordinate System  '''
        im_neix = self.dim + self.n_neighbors
        X = X.astype(np.float32)
        try:
            W = self.W
            G = self.G
            Y = self.Y
            print 'Using Trained Map'
        except:

            if self.init == None:
                W = np.random.random((im_neix, X.shape[1])).astype(np.float32)
                Y = np.random.random((im_neix, self.dim)).astype(np.float32)
                print 'random map initialization'
            elif self.init == 'Spectral':
                W = X[:6000].astype(np.float32)
                '''Initializing with Spectral Embedding '''
                Y = SpectralEmbedding(n_components=self.dim).fit_transform(W).astype(np.float32)
            ''' Initialize Graph Adjacency and Weight Matrices '''
            G = np.ones((W.shape[0], W.shape[0])).astype(np.float32)
        E_q= np.zeros(W.shape[0]).astype(np.float32)
        lrst = self.lrst
        ''' index of the last nearest neighbor : depends on output dimensionality. Higher the output dimensionality, 
        higher the connectedness '''

        knn_recorded = np.zeros((X.shape[0], im_neix)).astype(np.int32)
        min_dist = self.min_dist
        spread = self.spread
        alpha, beta = find_spread_tightness(spread, min_dist)
        if verbose :
            print "alpha : {} beta : {}".format(alpha, beta)
        thresh_g = -X.shape[1] ** .5 * np.log(self.sf)
        presented_len = 0
        order = np.random.permutation(X.shape[0])
        soed = 0
        lrdec = self.lrdec
        self.min_strength = 0.5**self.n_neighbors
        try:
            for i in range(self.max_its):
                grown = 0
                t = i* 1./self.max_its
                so_iter = not(i%(self.max_its/self.ss))
                growing_iter = t <= 0.5
                if so_iter:
                    soed += 1
                    order = np.random.permutation(X.shape[0])
                    presented_len = int(X.shape[0] * (soed * 1./self.ss) ** self.gamma )

                X_presented = X[order[:presented_len]]

                if so_iter:
                    k = 0

                    for x in X_presented:
                        tau = (X.shape[0] * i + k) * 1. / (X.shape[0] * self.max_its)
                        lr = np.float32(lrst * (1 - tau) ** lrdec)


                        nei_len = np.int32(min(im_neix, W.shape[0]))
                        dist_H= numba_ind_pairwise(x, W)
                        neilist = np.argpartition(dist_H,range(nei_len))[:nei_len].astype(np.int32)
                        b = neilist[0]
                        knn_rec = knn_recorded[order[k]]
                        knn_rec[:len(neilist)] = neilist
                        knn_recorded[order[k]] = knn_rec
                        k+=1

                        G[b] *= 0.5
                        G[b, neilist] = 1.
                        G[b][G[b] < self.min_strength] = 0
                        G[:, b][G[:, b] < self.min_strength] = 0
                        neighbors = np.where(G[b] + G[:, b])[0].astype(np.int32)
                        denom = dist_H[neilist[-1]]

                        neg_samples = np.random.randint(0, G.shape[0], size=int(self.ns_rate*np.sum(G[b] * G[:, b]))).astype(np.int32)
                        W[neighbors], Y[neighbors], Y[neg_samples] = train_neighborhood(x, Y[b], (W[neighbors]), dist_H[neighbors]/denom, (Y[neighbors]), (Y[neg_samples]), alpha, beta, min_dist, lr, b, neg_samples)
                        if growing_iter and G.shape[0] < 10000 and np.random.binomial(1, 1./np.log10(X.shape[0])):
                            e = (dist_H[0]) / denom
                            E_q[b] += dist_H[0] ** 0.5 * (1 + np.nan_to_num(1. / (0.0001 + e))) if X.shape[0] > 10000 else 1 + np.nan_to_num(1. / (0.0001 + e))
                            if G.shape[0] < presented_len and thresh_g <= E_q[b] :
                                G, W, Y, E_q = grow_map_at_node(b, neilist, E_q, thresh_g, self.dim + 1 , W, Y, G)

                                grown += 1

                        if not np.mod(k , 500) and verbose:
                            print(bcolors.OKGREEN + '\r |G|= %i , alpha: %.4f, beta: %.5f , iter = %i , X_i = %i/%i, Neighbors: %i, lr : %.4f'%(G.shape[0], alpha, beta, i+1, k,X_presented.shape[0], neighbors.shape[0], float(lr))+bcolors.ENDC),
                    E_q *= 0



                else :
                    if verbose :
                        print '\rTraining iteration %i without self organization '%(i+1),
                    Y = embed_batch(X_presented, np.float32(lrst), Y,  G, knn_recorded[order[:presented_len]], np.int32(self.max_its), np.float32(lrdec), np.int32(i), np.int32(self.ns_rate), alpha, beta, self.rng_state, self.min_strength)
                if L.shape[0]:
                    Y_plt = Y[pairwise_distances_argmin(X, W)]
                    f = plt.figure(figsize=(10, 10))
                    plt.scatter(Y_plt.T[0], Y_plt.T[1], c=L, alpha=0.6, s=8)
                    plt.savefig('./images/{}.png'.format(i))
                    plt.close(f)


        except KeyboardInterrupt:

            pass

        self.lrst = self.lrst *self.agility
        self.W = W
        self.Y = Y
        self.G = G
        preds = Y[pairwise_distances_argmin(X, W)]


        return self

    def fit_transform(self, X, L = np.array([])):
        self.fit(X, L)
        return self.transform(X)




    def transform(self, X):

        return self.Y[pairwise_distances_argmin(X, self.W)]
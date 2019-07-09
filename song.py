import numpy as np
from sklearn.metrics.pairwise import  pairwise_distances_argmin
from sklearn.manifold import SpectralEmbedding
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from util import  bcolors

from util import pairwise_and_neighbors, train_neighborhood, grow_map_at_node, find_spread_tightness, delete_mult_nodes, delete_node, embed_batch

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1



class SONG(BaseEstimator):

    def __init__(self, out_dim=2, n_neighbors = 20,  lr=1., its = 60, spread_factor = 0.9, spread = 1., min_dist = 0.01,
                 ns_rate = 5, push_strength = 1., init=None, self_organizing_stride = 12, lr_decay_const = 1):
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
        self.ss = self_organizing_stride
        random_state = np.random.RandomState()
        self.rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        if push_strength < 1. :
            raise ValueError('The repulsion strength must be greater than 1.')


    def fit(self, X, L=np.array([])):
        '''
        :param X: The input dataset normalized over the dataset to range [0, 1].
        :return: Y : The Mapped Coordinates in the desired output space (default = 2 ).
        '''
        ''' Initialize Weight and Coordinate System  '''
        im_neix = self.dim + self.n_neighbors
        min_conn = self.dim+1
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
            G = np.ones((W.shape[0], W.shape[0])).astype(np.int8)
        E_q= np.zeros(W.shape[0]).astype(np.float32)
        lrst = self.lrst
        min_growth = 0.01
        ''' index of the last nearest neighbor : depends on output dimensionality. Higher the output dimensionality, 
        higher the connectedness '''

        knn_recorded = np.zeros((X.shape[0], im_neix)).astype(np.int32)
        dist_H_recorded = np.zeros((X.shape[0], im_neix)).astype(np.float32)
        min_dist = self.min_dist # Output minimum distance between nodes
        spread = self.spread
        alpha, beta = find_spread_tightness(spread, min_dist)
        print "alpha : {} beta : {}".format(alpha, beta)
        thresh_g = -X.shape[1] ** .5 * np.log(self.sf)
        bmus = []
        stable = 0
        tau = 1
        lr = lrst
        lrdec = self.lrdec
        order = np.random.permutation(X.shape[0])
        try:
            for i in range(self.max_its):
                grown = 0
                t = i* 1./self.max_its
                so_iter = not(i%self.ss)
                growing_iter = t<= 0.5
                if i == 0:
                    X_presented = X[order]
                else:
                    presented_len = int(X.shape[0] * np.exp(-4 * (1 - t) ** 2))
                    X_presented = X[order[:presented_len]]

                if so_iter:
                    k = 0

                    for x in X_presented:
                        tau = (X.shape[0] * i + k) * 1. / (X.shape[0] * self.max_its)
                        lr = np.float32(lrst * (1 - tau) ** lrdec)

                        nei_len = np.int32(min(im_neix, W.shape[0]))
                        dist_H, neilist = pairwise_and_neighbors(x, W, nei_len)
                        b = neilist[0]
                        denom = dist_H[neilist[-1]]
                        knn_rec = knn_recorded[order[k]]
                        knn_rec[:len(neilist)] = neilist
                        knn_recorded[order[k]] = knn_rec
                        k+=1
                        G[b] *= 0
                        G[b, neilist] = 1
                        neighbors = np.where(G[b] * G[:, b])[0].astype(np.int32)

                        neg_samples = np.random.randint(0, G.shape[0], size=int(self.ns_rate*neighbors.shape[0])).astype(np.int32)

                        W[neighbors], Y[neighbors], Y[neg_samples] = train_neighborhood(x, Y[b], (W[neighbors]), dist_H[neighbors]/denom, (Y[neighbors]), (Y[neg_samples]), alpha, beta, min_dist, lr, b, neg_samples)
                        #/ (1+ e**.5)
                        if growing_iter:
                            e = dist_H[b] / denom

                            E_q[b] += dist_H[b]**.5 * (1 + 1 /( 0.0001+ e))
                            if np.ceil(G[b]).sum() >= min_conn:
                                if G.shape[0] < self.map_ratio * X.shape[0] and thresh_g <= E_q[b] :
                                    G, W, Y, E_q = grow_map_at_node(b, neilist, E_q, thresh_g, self.dim + 1 , W, Y, G)
                                    grown += 1
                            else :
                                G, W, Y, E_q = delete_node(b, G, W, Y, E_q)
                        if not np.mod(k , 500):
                            print(bcolors.OKGREEN + '\r |G|= %i , alpha: %.4f, beta: %.5f , iter = %i , X_i = %i, Neighbors: %i, lr : %.4f'%(G.shape[0], alpha, beta, i+1, k, neighbors.shape[0], float(lr))+bcolors.ENDC),
                    E_q *= 0
                    delinds = np.logical_or(np.sum(G, axis=1)<min_conn , np.sum(G, axis=0) < min_conn)
                    delinds = np.where(delinds)[0]
                    G, W, Y, E_q = delete_mult_nodes(delinds, G, W, Y, E_q)

                else :

                    if knn_recorded.sum() ==0 :
                        p = 0
                        for x in X:
                            dist_H, neilist = pairwise_and_neighbors(x, W, im_neix)
                            knn_recorded[p] = neilist
                            dist_H_recorded[p] = dist_H[im_neix]
                            p += 1
                    ''' train without growth '''
                    print '\rTraining iteration %i without self organization '%(i+1),
                    Y = embed_batch(X_presented, np.float32(lrst), Y, G, knn_recorded[order[:presented_len]], np.int32(self.max_its), np.float32(lrdec), np.int32(i), np.int32(self.ns_rate), alpha, beta, self.rng_state)
                if L.shape[0]:
                    Y_plt = Y[pairwise_distances_argmin(X, W)]
                    f = plt.figure(figsize=(10, 10))
                    plt.scatter(Y_plt.T[0], Y_plt.T[1], c=plt.cm.jet(L / 10.), alpha=0.6, s=8)
                    plt.savefig('./images/{}.png'.format(i))
                    plt.close(f)


        except KeyboardInterrupt:

            pass
        self.W = W
        self.Y = Y
        self.G = G
        preds = Y[pairwise_distances_argmin(X, W)]
        self.lrst = self.lrst *0.75
        # self.max_its = int(self.max_its * 0.9)

        return self

    def fit_transform(self, X, L = np.array([])):
        self.fit(X, L)
        return self.transform(X)




    def transform(self, X):

        return self.Y[pairwise_distances_argmin(X, self.W)]
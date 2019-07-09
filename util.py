import numba
import numpy as np
from scipy.optimize import curve_fit
@numba.njit("i4(i8[:])")
def tau_rand_int(state):
    """A fast (pseudo)-random number generator.

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random int32 value
    """
    state[0] = (((state[0] & 4294967294) << 12) & 0xffffffff) ^ (
        (((state[0] << 13) & 0xffffffff) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xffffffff) ^ (
        (((state[1] << 2) & 0xffffffff) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xffffffff) ^ (
        (((state[2] << 3) & 0xffffffff) ^ state[2]) >> 11
    )

    return state[0] ^ state[1] ^ state[2]


@numba.njit("Tuple((f4[:], i4[:]))(f4[:],f4[:,:], i4)", fastmath=True, parallel=True)
def pairwise_and_neighbors(x, W, n_neis):
    dists_2 = []
    l = [np.float32(np.inf) for i in range(n_neis)]
    neinds = [-1 for i in range(n_neis)]
    for i in range(W.shape[0]):
        w = W[i]
        dist2 = np.float32(0)
        for k in range(w.shape[0]):
            dist2 += pow(w[k] - x[k], 2)
        dists_2.append(dist2)

        ''' place dist2 in the neighbor list'''
        e = dist2
        k = len(l)
        if l[-1] < e:
            l= l
        else:
            mid = len(l) // 2
            end = len(l) - 1
            beg = 0
            while end - beg and not (mid == beg or mid == end):
                if l[mid] > e:
                    end = mid
                else:
                    beg = mid
                mid = beg + (end - beg) // 2
            offset = l[mid] < e
            l_ret = (l[:mid + offset])
            n_ret = (neinds[:mid + offset])
            l_ret.append(e)
            n_ret.append(i)
            l_ret.extend(l[mid + offset:])
            n_ret.extend(neinds[mid + offset:])
            l = (l_ret[:k])
            neinds = n_ret[:k]

    return np.array(dists_2).astype(np.float32), np.array(neinds, dtype=np.int32)

@numba.njit('i4[:](i4[:], i4[:])')
def get_mutual_neighborhood(im_neix, rev_neis):
    return_neis = []
    for i in im_neix:
        for r in rev_neis:
            if i == r:
                return_neis.append(i)
                break

    return np.array(return_neis)

@numba.njit('f4(f4,f4)', fastmath=True)
def positive_clip(x, v):
    if x >= v:
        return v
    elif x <= -v:
        return -v
    else :
        return x

def delete_mult_nodes(node_list, G, W, Y, E_q):
    keepinds = np.setdiff1d(np.arange(G.shape[0]), node_list)
    G = G[keepinds][:, keepinds]
    W = W[keepinds]
    Y = Y[keepinds]
    E_q = E_q[keepinds]
    return G, W, Y, E_q

def delete_node(node, G, W, Y, E_q):
    keepinds = []#np.where(np.abs(np.array(range(G.shape[0]), dtype=np.int32)-node))[0]
    for i in range(G.shape[0]):
        if not(i==node):
            keepinds.append(i)
    G = G[keepinds][:, keepinds]
    W = W[keepinds]
    Y = Y[keepinds]
    E_q = E_q[keepinds]
    return G, W, Y, E_q

def remove_duplicates(self):
    for j in np.where(np.isnan(self.Y.sum(axis=0))):
        self.delete_node(j)
def find_spread_tightness(spread, min_dist):
    def curve (x, a, b):
        return 1./(1. + a * x ** (2*b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.
    yv[xv>=min_dist] = np.exp( - (xv[xv>=min_dist] - min_dist)/ spread)
    params, covar = curve_fit(curve, xv, yv)
    params = params.astype(np.float32)
    if params[1]<1:
        ''' Needed to avoid negative powers'''
        params[1] = max(params[1], 1.)
    return params[0], params[1]


def grow_map_at_node(b, neighbors, E_q, thresh_g, im_neix, W, Y, G):
    e_ratio = E_q[b] / thresh_g
    # for k in range(int(e_ratio)):
    if e_ratio>1:
        ''' position of the new node is the centroid of the k-simplex closest to the growing node'''

        closests = neighbors[:im_neix]#[E_q[neighbors].argsort()][-(im_neix):]
        W_n = W[closests].mean(axis=0)
        Y_n = Y[closests].mean(axis=0)
        new_W_size = np.array(W.shape)+np.array([1, 0])
        W.resize(new_W_size, refcheck=False)
        W[-1] = W_n
        Y.resize(np.array(Y.shape) + np.array([1, 0]), refcheck=False)
        Y[-1] = Y_n
        oldG = G
        G = np.zeros(np.array(G.shape)+1).astype(np.int8)
        G[:-1, :-1] = oldG
        ''' connect neighbors to the new node '''
        G[-1, closests] = 1
        G[closests, -1] = 1

        ''' Append new error. '''
        E_q.resize(E_q.shape[0]+1, refcheck=False)
        E_q[-1] = 0
        E_q[closests] = 0

    return G, W, Y, E_q


@numba.njit('f4(f4[:], f4[:])', fastmath=True, parallel=True)
def rdist(x, y):
    dist = 0
    for i in range(len(x)):
        dist += pow(x[i] - y[i], 2)
    return dist

@numba.njit('UniTuple(f4[:, :], 3)(f4[:], f4[:], f4[:,:],f4[:], f4[:, :], f4[:,:], f4, f4, f4, f4, i4, i4[:])', parallel = True, fastmath=True)
def train_neighborhood(x, y_b, W, hdist_nei, Y_nei, Y_oth, alpha, beta , mindist, lr, b, negs):
    hdists = hdist_nei

    ''' Self Organizing '''
    for j in range(W.shape[0]):
        hdist = hdists[j]
        h_pull_grad =  np.exp(-2*hdist)
        ldist_sq = rdist(y_b, Y_nei[j])
        '''deltas of the embedding are multiplied by a regularization term equal to the euclidean distance'''
        pull_grad = (2 * alpha * beta * pow(ldist_sq, beta - 1))
        denom = (1 + alpha * pow(ldist_sq, beta))
        pull_grad/= denom

        for i in range(x.shape[0]):
            W[j][i] += h_pull_grad * lr * (x[i] - W[j][i])
            if i < y_b.shape[0]:
                Y_nei[j][i] += positive_clip(pull_grad * (y_b[i]-Y_nei[j][i]), 4) * lr

    '''negative embedding'''
    for j in range(Y_oth.shape[0]):
        ldist_sq = rdist(y_b, Y_oth[j])
        push_grad = (2  * beta)
        denom = 1 + alpha * pow(ldist_sq, beta)
        push_grad /= denom
        for i in range(y_b.shape[0]):
            if ldist_sq >0.:
                Y_oth[j][i] -= positive_clip(push_grad / ldist_sq * (y_b[i]-Y_oth[j][i]) , 4)* lr
            elif b == negs[j]:
                continue
            else :
                Y_oth[j][i] += lr * 4

    return W, Y_nei, Y_oth



@numba.njit('f4[:, :](f4[:,:], i4[:], f4, f4, f4, i4, i4, i8[:])', parallel = True, fastmath=True)
def train_embedding(Y, neighbors, alpha, beta ,lr, b, ns_rate, rng_state):
    y_b = Y[b]
    ''' Self Organizing '''
    for j in range(neighbors.shape[0]):
        ldist_sq = rdist(y_b, Y[neighbors[j]])
        '''deltas of the embedding are multiplied by a regularization term equal to the euclidean distance'''
        pull_grad = (2 * alpha * beta * pow(ldist_sq, beta - 1))
        denom = (1 + alpha * pow(ldist_sq, beta))
        pull_grad/= denom

        for i in range(y_b.shape[0]):
            Y[neighbors[j]][i] += positive_clip(pull_grad * (y_b[i]-Y[neighbors[j]][i]) , 4)* lr

    '''negative embedding'''
    for p in range(neighbors.shape[0] * ns_rate):
        j = tau_rand_int(rng_state) % Y.shape[0]
        ldist_sq = rdist(y_b, Y[j])
        push_grad = (2  * beta)
        denom = 1 + alpha * pow(ldist_sq, beta)
        push_grad /= denom
        for i in range(y_b.shape[0]):
            if ldist_sq >0.:
                Y[j][i] -= positive_clip(push_grad / ldist_sq * (y_b[i]-Y[j][i]) , 4)* lr
            elif b == j:
                continue
            else :
                Y[j][i] += lr * 4

    return Y

@numba.njit('f4[:,:](f4[:,:], f4, f4[:,:], i1[:, :], i4[:,:], i4, f4, i4, i4, f4, f4, i8[:])', fastmath = True, parallel = True)
def embed_batch(X, lrst, Y, G, knn_recorded, max_its, lrdec, i, ns_rate, alpha, beta, rng_state):
    for j in range(X.shape[0]):
        tau = (X.shape[0] * i + j) * 1. / (max_its * X.shape[0])
        lr = np.float32(lrst * (1 - tau) ** lrdec)
        im_neis = knn_recorded[j]
        b = im_neis[0]
        rev_neis = np.where(G[:, b])[0].astype(np.int32)
        # G[b] *= 0
        for imnei in im_neis:
            G[b, imnei] = 1

        neighbors = get_mutual_neighborhood(im_neis, rev_neis)  # (x, W[im_neis], im_neis, rev_neis)
        Y = train_embedding(Y, neighbors, alpha, beta, lr, b, ns_rate, rng_state)
        # if not np.mod(j, 500):
            # print('\r |G|= '+(G.shape[0])+' , iter = '+(i+1) +' , X_i = ' + (j)),

    return Y




@numba.njit('Tuple((i4[:], f4, f4[:]))(f4[:], f4[:,:], i4[:], i4[:])', fastmath=True, parallel=True)
def get_neighbor_hdists(x, W, imneis, revneis):
    max_val = -1
    dists =[]
    mut_neis = []
    for i in range(len(W)):
        ret = 0
        for r in revneis:
            if r == imneis[i]:
                ret = 1
                mut_neis.append(r)
                break
        dist = np.float32(0)
        for j in range(len(x)):
            dist += pow(x[j]-W[i][j], 2)
        if max_val<dist:
            max_val = dist
        if ret:
            dists.append(dist)
    return np.array(mut_neis).astype(np.int32), max_val, np.array(dists).astype(np.float32)

@numba.njit('f4[:,:](i4, i4, f4[:,:], f4, f4[:,:], i1[:, :], i4[:,:], i4, f4, i4, f4, f4, i8[:])', fastmath = True, parallel = True)
def embed_batch_for_iters (start_iter, end_iter, X, lrst, Y, G, knn_recorded, max_its, lrdec, ns_rate, alpha, beta, rng_state):
    for p in range(start_iter, end_iter):
       Y =  embed_batch(X, lrst, Y, G, knn_recorded, max_its, lrdec, np.int32(p), ns_rate, alpha, beta, rng_state)
    return Y


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
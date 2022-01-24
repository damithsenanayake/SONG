import numba
import numpy as np
from scipy.optimize import curve_fit
from pynndescent import NNDescent


@numba.njit("i4(i8[:])")
def fast_random_integer(random_state):
    """
    XORShift Pseudorandom Number Generator (inspired by use in UMAP)
    0xffffffff ensures 32bit truncation for faster operations

    """
    random_state[0] &= 6489292939
    random_state[0] ^= random_state[0]<<13
    random_state[0] ^= random_state[0]>>7
    random_state[0] ^= random_state[0]<<17
    random_state[0] &=  0xffffffff

    random_state[0] &= 6498787687
    random_state[1] ^= random_state[1] << 15
    random_state[1] ^= random_state[1] >> 9
    random_state[1] ^= random_state[1] << 17
    random_state[1] &= 0xffffffff

    random_state[0] &= 64654892939
    random_state[2] ^= random_state[2] << 17
    random_state[2] ^= random_state[2] >> 5
    random_state[2] ^= random_state[2] << 3
    random_state[2] &= 0xffffffff

    return random_state[0] ^ random_state[1] ^ random_state[2]

@numba.njit('f4[:,:](f4[:,:], f4[:,:])', fastmath=True, parallel=True)
def sq_eucl_opt(A, B):
    ''' function adapted from https://github.com/numba/numba-scipy/issues/38#issuecomment-623569703 '''
    C = np.empty((A.shape[0], B.shape[0]), np.float32)
    I_BLK = 32
    J_BLK = 32

    init_val_arr = np.zeros(1, A.dtype)
    init_val = init_val_arr[0]

    for ii in numba.prange(A.shape[0] // I_BLK):
        for jj in range(B.shape[0] // J_BLK):
            for i in range(I_BLK // 4):
                for j in range(J_BLK // 2):
                    acc_0 = init_val
                    acc_1 = init_val
                    acc_2 = init_val
                    acc_3 = init_val
                    acc_4 = init_val
                    acc_5 = init_val
                    acc_6 = init_val
                    acc_7 = init_val
                    for k in range(A.shape[1]):
                        acc_0 += (A[ii * I_BLK + i * 4 + 0, k] - B[jj * J_BLK + j * 2 + 0, k]) ** 2
                        acc_1 += (A[ii * I_BLK + i * 4 + 0, k] - B[jj * J_BLK + j * 2 + 1, k]) ** 2
                        acc_2 += (A[ii * I_BLK + i * 4 + 1, k] - B[jj * J_BLK + j * 2 + 0, k]) ** 2
                        acc_3 += (A[ii * I_BLK + i * 4 + 1, k] - B[jj * J_BLK + j * 2 + 1, k]) ** 2
                        acc_4 += (A[ii * I_BLK + i * 4 + 2, k] - B[jj * J_BLK + j * 2 + 0, k]) ** 2
                        acc_5 += (A[ii * I_BLK + i * 4 + 2, k] - B[jj * J_BLK + j * 2 + 1, k]) ** 2
                        acc_6 += (A[ii * I_BLK + i * 4 + 3, k] - B[jj * J_BLK + j * 2 + 0, k]) ** 2
                        acc_7 += (A[ii * I_BLK + i * 4 + 3, k] - B[jj * J_BLK + j * 2 + 1, k]) ** 2
                    C[ii * I_BLK + i * 4 + 0, jj * J_BLK + j * 2 + 0] = (acc_0)
                    C[ii * I_BLK + i * 4 + 0, jj * J_BLK + j * 2 + 1] =(acc_1)
                    C[ii * I_BLK + i * 4 + 1, jj * J_BLK + j * 2 + 0] =(acc_2)
                    C[ii * I_BLK + i * 4 + 1, jj * J_BLK + j * 2 + 1] = (acc_3)
                    C[ii * I_BLK + i * 4 + 2, jj * J_BLK + j * 2 + 0] =(acc_4)
                    C[ii * I_BLK + i * 4 + 2, jj * J_BLK + j * 2 + 1] = (acc_5)
                    C[ii * I_BLK + i * 4 + 3, jj * J_BLK + j * 2 + 0] = (acc_6)
                    C[ii * I_BLK + i * 4 + 3, jj * J_BLK + j * 2 + 1] = (acc_7)
        for i in range(I_BLK):
            for j in range((B.shape[0] // J_BLK) * J_BLK, B.shape[0]):
                acc_0 = init_val
                for k in range(A.shape[1]):
                    acc_0 += (A[ii * I_BLK + i, k] - B[j, k]) ** 2
                C[ii * I_BLK + i, j] = (acc_0)

    for i in range((A.shape[0] // I_BLK) * I_BLK, A.shape[0]):
        for j in range(B.shape[0]):
            acc_0 = init_val
            for k in range(A.shape[1]):
                acc_0 += (A[i, k] - B[j, k]) ** 2
            C[i, j] = (acc_0)

    return C



@numba.njit(fastmath=True)
def get_closest_for_inputs(X, W):
    min_dist_args = np.zeros(X.shape[0], dtype=np.int64)#
    min_dists = np.zeros(X.shape[0], dtype = np.int64)
    batch_len = 5000

    for b in range(len(X)//batch_len + 1):

        pdists = sq_eucl_opt(X[b * batch_len : (b+1) * batch_len], W)

        for i in range(pdists.shape[0]):

            min_dist_args[b*batch_len + i] = pdists[i].argmin()
            min_dists[b*batch_len + i] = pdists[i].min()
    return min_dist_args, min_dists



@numba.njit(fastmath=True)
def get_fast_pairwise(X, W, k):
    min_dist_args = np.zeros((X.shape[0], k), dtype=np.int64)  #
    min_dists = np.zeros((X.shape[0],k), dtype=np.float32)
    batch_len = 5000

    for b in range(len(X) // batch_len + 1):

        pdists = sq_eucl_opt(X[b * batch_len: (b + 1) * batch_len], W)

        for i in range(pdists.shape[0]):
            min_dist_args[b * batch_len + i] = get_closest(pdists[i],k)
            min_dists[b * batch_len + i] = pdists[i][min_dist_args[b * batch_len + i]]
    return min_dist_args, min_dists





# @numba.njit(fastmath=True)
def get_reverse_index(bmus, g_size):
    index = [[] for _ in range(g_size)]
    for i in range(len(bmus)):
        index[bmus[i]].append(i)
    return index

# @numba.jit(fastmath=True)
def combine_neighbors(index, neighbors):
    candidates = []

    for i in range(len(neighbors)):
        candidates.extend(index[neighbors[i]])
    return np.array(candidates)

def get_neighbor_lists(G):
    shp = np.arange(G.shape[0])

    neighbor_tree = []#[[] for _ in range(len(G))]

    for i in range(len(G)):
        bix = np.logical_or(G[i] == 1 , G.T[i] == 1)
        neighbor_tree.append(shp[bix])
    return neighbor_tree

# @numba.njit(fastmath=True)
def get_fast_knn(X_pc, W_pc, x_k, G):
    print('getting knns')
    closest_cvs, _ = get_closest_for_inputs(X_pc, W_pc)
    reverse_index = get_reverse_index(closest_cvs, G.shape[0])
    knns = np.zeros((X_pc.shape[0], x_k))
    knn_dists = np.zeros(knns.shape, dtype=np.float32)
    values = np.zeros(X_pc.shape[0] * x_k, dtype=np.float32)
    rows = np.zeros(X_pc.shape[0] * x_k, dtype=np.int32)
    cols = np.zeros(X_pc.shape[0] * x_k, dtype=np.int32)
    X_r = X_pc.reshape(X_pc.shape[0], 1, X_pc.shape[1])
    neighbor_tree = get_neighbor_lists(G)
    for i in range(X_pc.shape[0]):
        # G_neis = G[closest_cvs[i]]
        neighbors = neighbor_tree[closest_cvs[i]]
        candidates = combine_neighbors(reverse_index, neighbors)
        knn_entry, dists = get_fast_pairwise(X_r[i], X_pc[candidates], x_k)
        knns[i] = candidates[knn_entry[0]]
        knn_dists[i] = dists[0]
        start_ix = i * x_k
        end_ix = (i+1)  * x_k

        values[start_ix:end_ix] = knn_dists[i]/(knn_dists[i][-1]+0.0001)
        rows[start_ix:end_ix] = np.ones(x_k) * i
        cols[start_ix:end_ix] = knns[i]

    return values, rows, cols

@numba.njit('f4(f4,f4)', fastmath=True, )
def positive_clip(x, v):
    if x >= v:
        return v
    elif x <= -v:
        return -v
    else:
        return x


def find_spread_tightness(spread, min_dist):
    def curve(x, a, b):
        return 1. / (1. + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.
    yv[xv >= min_dist] = np.exp(- (xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    params = params.astype(np.float32)
    return params[0], params[1]


@numba.njit('f4(f4[:], f4[:])', fastmath=True, )
def rdist(x, y):
    dist = 0
    for i in range(len(x)):
        dist += pow(x[i] - y[i], 2)
    return dist


@numba.njit(fastmath = True)
def bulk_grow_with_drifters(shp, E_q, thresh_g, drifters, G, W, Y):
    growth_size = max(0, len(shp[E_q >= thresh_g]) - len(drifters))

    if growth_size:
        oldG = G
        oldW = W
        oldY = Y
        oldE = E_q
        old_size = oldW.shape[0]
        W = np.zeros((W.shape[0] + growth_size, W.shape[1]), dtype=np.float32)
        W[:-growth_size] = oldW

        Y = np.zeros((Y.shape[0] + growth_size, Y.shape[1]), dtype=np.float32)
        Y[: -growth_size] = oldY

        G = np.zeros((G.shape[0] + growth_size, G.shape[1] + growth_size), dtype=np.float32)
        G[:-growth_size][:, :-growth_size] = oldG

        E_q = np.zeros(E_q.shape[0] + growth_size, dtype=np.float32)
        E_q[:-growth_size] = oldE

        grown = 0
        growing_nodes = shp[oldE >= thresh_g]

        shp = np.arange(G.shape[0]).astype(np.int32)

        if len(growing_nodes) >= 1:
            for k in range(len(growing_nodes)):
                b = growing_nodes[k]
                ''' If no reusable nodes create new nodes'''
                closests = shp[G[b] == 1]
                if len(closests) == 0:
                    drifters = np.append(drifters, b)
                    continue
                h_bias = 1e-1
                l_bias = 1e-8
                W_n = (1-h_bias) * W[b] + h_bias * (W[closests].sum(axis=0) / len(closests))
                Y_n = (1-l_bias) * Y[b] + l_bias * (Y[closests].sum(axis=0) / len(closests))
                if grown >= len(drifters):
                    W[old_size + grown - len(drifters)] = W_n

                    Y[old_size + grown - len(drifters)] = Y_n
                    ''' connect neighbors to the new node '''
                    G[old_size + grown - len(drifters)][b] = 1
                    G[b][old_size + grown - len(drifters)] = 0
                    G[old_size + grown - len(drifters)][closests] = 1
                    G[closests][:, old_size + grown - len(drifters)] = 0

                    ''' Append new error. '''
                    E_q[old_size + grown - len(drifters)] = 0
                else:
                    '''replace unusable nodes with new nodes'''
                    W[drifters[grown]] = W_n
                    Y[drifters[grown]] = Y_n
                    G[drifters[grown]][b] = 1
                    G[b][drifters[grown]] = 0
                    G[drifters[grown]][closests] = 1
                    G[closests][:, drifters[grown]] = 0
                G[b][closests] = 0
                G[closests][:, b] = 0
                E_q[closests] *= 0.
                grown += 1

    return W, G, Y, E_q



@numba.njit('f4(f4, f4)', fastmath=True)
def get_so_rate(tau, sigma):
    return  np.exp(-sigma * tau ** 2)

@numba.njit('i4[:](f4[:], i8)')
def get_closest(dists_2, k):
    neinds = [1 for _ in range(k)]
    n_W = len(dists_2)
    for i in range(n_W):
        dist2 = dists_2[i]
        ''' place dist2 in the neighbor list'''
        e = dist2
        if i < k or dists_2[neinds[-1]] > e:
            end = min(k - 1, i)
            mid = (end + 1) // 2
            beg = 0
            while end - beg and not (mid == beg or mid == end):
                if dists_2[neinds[mid]] > e:
                    end = mid
                else:
                    beg = mid
                mid = beg + (end - beg) // 2
            offset = dists_2[neinds[mid]] < e
            neinds.insert(mid + offset, i)
            neinds = neinds[:k]

    return np.array(neinds, dtype=np.int32)


@numba.njit(fastmath=True, )
def train_for_batch_batch(X_presented, pdist_matrix, i, max_its, lrst, lrdec, im_neix, W, max_epochs_per_sample, G, epsilon, min_strength,
                           shp, Y, ns_rate, alpha, beta, rng_state, E_q, lr_sigma, reduced_lr):


    taus = ((i * X_presented.shape[0] + np.arange(len(X_presented)).astype(np.float32)) * 1. / (max_its * X_presented.shape[0]))
    lrs = (1-taus)* reduced_lr
    so_lr = lrst * get_so_rate(i * 1. / max_its, lr_sigma)
    nei_len = np.int32(min(im_neix, W.shape[0]))

    for k in range(len(X_presented)):
        x = X_presented[k]
        dist_H = pdist_matrix[k]
        neilist = get_closest(dist_H, nei_len)

        b = neilist[0]
        G[b] *= epsilon

        G[b][neilist] = 1.
        G[b][G[b] < min_strength] = 0
        G[:, b][G[:, b] < min_strength] = 0
        nei_bin = (G[b] + G[:, b]) > 0
        neighbors = shp[nei_bin]

        denom = dist_H[neilist[-1]]

        epoch_vector = max_epochs_per_sample * ((G[b] + G[:, b]) / 2. + 1)
        neg_epoch_vector = ns_rate* (1 - (G[b] + G[:, b]) / 2.) + 1

        '''x, so_lr, b, W, hdist_nei, Y, alpha, beta , lr,  rng_state):'''
        W[neighbors], Y = train_neighborhood(x, so_lr, b, neighbors.astype(np.int32), W[neighbors],
                                             dist_H[neighbors] / denom,
                                             Y, ns_rate, alpha, beta,
                                             lrs[k], rng_state, epoch_vector.astype(np.int32),
                                             neg_epoch_vector.astype(np.int32))
        E_q[b] += dist_H[b]#/denom

    return W, Y, G, E_q


@numba.njit(
    'UniTuple(f4[:, :], 2)(f4[:], f4,  i4, i4[:],  f4[:,:], f4[:], f4[:,:], i4, f4, f4, f4, i8[:], i4[:], i4[:])',
    fastmath=True, )
def train_neighborhood(x, so_lr, b, neighbors, W, hdist_nei, Y, ns_rate, alpha, beta, lr, rng_state, epoch_vector,
                       neg_epoch_vector):
    hdists = hdist_nei
    y_b = Y[b]
    ''' Self Organizing '''
    sigma = 1
    for j in range(W.shape[0]):
        hdist = hdists[j]

        h_pull_grad =  1. if neighbors[j] == b else np.exp(-sigma * hdist)
        for i in range(x.shape[0]):
            W[j][i] += h_pull_grad * so_lr * (x[i] - W[j][i])

    '''negative embedding'''

    for j in range(W.shape[0]):

        epochs = epoch_vector[neighbors[j]]
        for e in range(epochs):
            for p in range(neg_epoch_vector[neighbors[j]]):
                n = fast_random_integer(rng_state) % Y.shape[0]
                ldist_sq = rdist(y_b, Y[n])
                push_grad = (2 * beta)
                denom = 1 + alpha * pow(ldist_sq, beta)
                push_grad /= denom
                for i in range(y_b.shape[0]):
                    if ldist_sq > 0.:
                        y_b[i] += positive_clip(push_grad / (ldist_sq + 0.001) * (y_b[i] - Y[n][i]),
                                                4) * lr
                        Y[n][i] -= positive_clip(push_grad / (ldist_sq + 0.001) * (y_b[i] - Y[n][i]),
                                                4) * lr
                    elif b == n:
                        continue
                    else:
                        y_b[i] += lr * 4
            Y_j = Y[neighbors[j]]
            ldist_sq = rdist(y_b, Y_j)
            '''deltas of the embedding are multiplied by a regularization term equal to the euclidean distance'''
            pull_grad = (2 * alpha * beta * pow(ldist_sq, beta - 1))
            denom = (1 + alpha * pow(ldist_sq, beta))
            pull_grad /= denom
            for i in range(y_b.shape[0]):
                Y_j[i] += positive_clip(pull_grad * (y_b[i] - Y_j[i]), 4) * lr
                y_b[i] -= positive_clip(pull_grad * (y_b[i] - Y_j[i]), 4) * lr

    return W, Y

@numba.njit('f4[:,:]( f4[:,:], f4[:, :], i4, i4, f4, f4, i8[:], f4 )', fastmath=True, )
def embed_batch_epochs(Y, G, max_its, i_st, alpha, beta, rng_state, agility):
    shp = np.arange(G.shape[0]).astype(np.int32)
    P_matrix = (G + G.T) / 2.
    tau = (i_st) * 1. / (max_its)
    tau_end = 1.
    epochs_per_sample = ((P_matrix)).astype(np.int32)
    dampening_coefficient = 1 if G.shape[0] < 50 else 1
    starting_lr = (1 - tau) * dampening_coefficient
    ending_lr = (1 - tau_end) * dampening_coefficient
    epochs_per_negative_sample = ((P_matrix.max() - P_matrix) * 5).astype(np.int32)
    totits = np.sum(epochs_per_sample)
    updated = 0

    pos_edges = np.where(epochs_per_sample)
    n_edges = len(pos_edges[0])
    for p in range(n_edges):
        j = pos_edges[0][p]
        lr = (starting_lr + (ending_lr - starting_lr) * (updated * 1. / totits)) * agility
        neighbors = shp[G[j] + G[:, j] > 0]
        for k in range(neighbors.shape[0]):
            epochs = epochs_per_sample[j][neighbors[k]]
            epochs_per_sample[j][neighbors[k]] = 0
            neg_epochs = int(epochs_per_negative_sample[j][neighbors[k]])
            for ep in range(epochs):
                for negs in range(neg_epochs):
                    n = fast_random_integer(rng_state) % Y.shape[0]
                    current_neg_epochs = epochs_per_negative_sample[j][n]
                    epochs_per_negative_sample[j][n] -= 1
                    if current_neg_epochs > 0:
                        ldist_sq = rdist(Y[j], Y[n])
                        push_grad = (2 * beta)
                        denom = 1 + alpha * pow(ldist_sq, beta)
                        push_grad /= denom
                        for i in range(Y[j].shape[0]):
                            if ldist_sq > 0.:
                                Y[j][i] += positive_clip(push_grad / (ldist_sq + 0.001) * (Y[j][i] - Y[n][i]), 4) * lr
                                Y[n][i] -= positive_clip(push_grad / (ldist_sq + 0.001) * (Y[j][i] - Y[n][i]), 4) * lr
                            elif j == n:
                                continue
                            else:
                                Y[j][i] += lr * 4
                ldist_sq = rdist(Y[j], Y[neighbors[k]])

                pull_grad = (2 * alpha * beta * pow(ldist_sq, beta - 1))
                denom = (1 + alpha * pow(ldist_sq, beta))
                pull_grad /= denom

                for i in range(Y[j].shape[0]):
                    Y[neighbors[k]][i] += positive_clip(pull_grad * (Y[j][i] - Y[neighbors[k]][i]), 4) * lr
                    Y[j][i] -= positive_clip(pull_grad * (Y[j][i] - Y[neighbors[k]][i]), 4) * lr
                updated += 1

    return Y

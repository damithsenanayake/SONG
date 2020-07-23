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


@numba.njit("Tuple((f4[:], i4[:]))(f4[:],f4[:,:], i8)", fastmath=True)
def pairwise_and_neighbors(x, W, n_neis):
    dists_2 = np.ones(W.shape[0], dtype=np.float32) * np.float32(np.inf)
    neinds = np.ones(n_neis, dtype=np.int32)
    n_W = np.int64(W.shape[0])
    n_dim = np.int64(W.shape[1])
    for i in range(n_W):
        w = W[i]
        dist2 = np.float32(0)
        for k in range(n_dim):
            dist2 += pow(w[k] - x[k], 2)
        dists_2[i] = dist2

        ''' place dist2 in the neighbor list'''
        e = dist2

        if i < n_neis or dists_2[neinds[-1]] > e:
            end = min(n_neis - 1, i)
            mid = (end + 1) // 2
            beg = 0
            while end - beg and not (mid == beg or mid == end):
                if dists_2[neinds[mid]] > e:
                    end = mid
                else:
                    beg = mid
                mid = beg + (end - beg) // 2
            offset = dists_2[neinds[mid]] < e
            n_tail = (neinds[mid + offset:-1]).copy()

            neinds[mid + offset] = i
            neinds[mid + offset + 1:] = n_tail

    return dists_2, neinds.astype(np.int32)


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


def bulk_grow(shp, E_q, thresh_g, drifters, G, W, Y, X_presented):
    growth_size = max(0, len(shp[E_q >= thresh_g]) - len(drifters))

    if growth_size and G.shape[0] < 10000 and G.shape[0] < X_presented.shape[0]:
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
        shp = np.arange(G.shape[0]).astype(np.int32)
        growing_nodes = shp[E_q >= thresh_g]

        if len(growing_nodes) > 1:
            for k in range(len(growing_nodes)):
                b = growing_nodes[k]
                ''' If no reusable nodes create new nodes'''
                closests = shp[G[b] == 1]
                if len(closests) == 0:
                    drifters = np.append(drifters, b)
                    continue
                W_n = W[closests].sum(axis=0) / len(closests)
                Y_n = Y[closests].sum(axis=0) / len(closests)
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
                E_q[closests] = 0.5
                grown += 1

    return W, G, Y, E_q, drifters


def bulk_grow_with_gen(shp, E_q, thresh_g, drifters, G, W, Y, X_presented, birth_gen, last_gen):
    growth_size = max(0, len(shp[E_q >= thresh_g]) - len(drifters))

    if growth_size and G.shape[0] < 10000 and G.shape[0] < X_presented.shape[0]:
        oldG = G
        oldW = W
        oldY = Y
        oldE = E_q
        old_birth_gen = birth_gen
        old_size = oldW.shape[0]
        W = np.zeros((W.shape[0] + growth_size, W.shape[1]), dtype=np.float32)
        W[:-growth_size] = oldW

        Y = np.zeros((Y.shape[0] + growth_size, Y.shape[1]), dtype=np.float32)
        Y[: -growth_size] = oldY

        G = np.zeros((G.shape[0] + growth_size, G.shape[1] + growth_size), dtype=np.float32)
        G[:-growth_size][:, :-growth_size] = oldG

        E_q = np.zeros(E_q.shape[0] + growth_size, dtype=np.float32)
        E_q[:-growth_size] = oldE
        birth_gen = np.zeros(birth_gen.shape[0] + growth_size)
        birth_gen[:-growth_size] = old_birth_gen
        birth_gen[-growth_size:] = last_gen
        grown = 0
        shp = np.arange(G.shape[0]).astype(np.int32)
        growing_nodes = shp[E_q >= thresh_g]

        if len(growing_nodes) > 1:
            for k in range(len(growing_nodes)):

                b = growing_nodes[k]
                ''' If no reusable nodes create new nodes'''
                closests = shp[G[b] == 1]
                if len(closests) == 0:
                    drifters = np.append(drifters, b)
                    continue
                W_n = W[closests].sum(axis=0) / len(closests)
                Y_n = Y[closests].sum(axis=0) / len(closests)
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
                    birth_gen[old_size + grown - len(drifters)] = last_gen
                else:
                    '''replace unusable nodes with new nodes'''
                    W[drifters[grown]] = W_n
                    Y[drifters[grown]] = Y_n
                    G[drifters[grown]][b] = 1
                    G[b][drifters[grown]] = 0
                    G[drifters[grown]][closests] = 1
                    G[closests][:, drifters[grown]] = 0
                    birth_gen[drifters[grown]] = last_gen

                G[b][closests] = 0
                G[closests][:, b] = 0
                E_q[closests] = 0.5
                last_gen += 1

                grown += 1

    return W, G, Y, E_q, birth_gen, last_gen, drifters

@numba.njit('f4(f4)', fastmath=True)
def get_so_rate(tau):
    return  np.exp(-3 * tau**1) #if tau < 0.8 else np.exp(-5 * tau ** 2)

@numba.njit(fastmath=True, )
def train_for_batch(X_presented, i, max_its, lrst, lrdec, im_neix, W, max_epochs_per_sample, G, epsilon, min_strength,
                    shp, Y, ns_rate, alpha, beta, rng_state, E_q):

    dampen = 1

    if W.shape[0] <= im_neix ** 2:

        dampen = (W.shape[0] * 1./X_presented.shape[0])

    for k in range(len(X_presented)):
        x = X_presented[k]
        tau = np.float32((i * X_presented.shape[0] + k) * 1. / (max_its * X_presented.shape[0]))
        lr = np.float32(((1 - tau)) ** lrdec) * dampen
        so_lr = lrst * get_so_rate(tau)# * G.shape[0] *1./ X_presented.shape[0]#st * np.exp(-7 * tau ** 2)  # np.float32(so_lr_st * (1-tau)** lrdec)
        nei_len = np.int32(min(im_neix, W.shape[0]))

        dist_H, neilist = pairwise_and_neighbors(x, W, nei_len)

        b = neilist[0]
        G[b] *= epsilon

        G[b][neilist] = 1.
        G[b][G[b] < min_strength] = 0

        # G[:, b][G[:, b] < min_strength] = 0
        nei_bin = (G[b] + G[:, b]) > 0
        neighbors = shp[nei_bin]

        denom = dist_H[neilist[-1]]

        epoch_vector = max_epochs_per_sample * ((G[b] + G[:, b]) / 2. + 1)
        neg_epoch_vector = 1 * ns_rate * (1 - (G[b] + G[:, b]) / 2.)

        '''x, so_lr, b, W, hdist_nei, Y, alpha, beta , lr,  rng_state):'''
        W[neighbors], Y = train_neighborhood(x, so_lr, b, neighbors.astype(np.int32), W[neighbors],
                                             dist_H[neighbors] / denom,
                                             Y, ns_rate, alpha, beta,
                                             lr, rng_state, epoch_vector.astype(np.int32),
                                             neg_epoch_vector.astype(np.int32))
        E_q[b] += dist_H[0]

    return W, Y, G, E_q


@numba.njit(fastmath=True, )
def train_for_input(x, X_presented, i, k, max_its, lrst, lrdec, im_neix, W, max_epochs_per_sample, G, epsilon,
                    min_strength, shp, Y, ns_rate, alpha, beta, rng_state, E_q):
    dampen = 1
    if W.shape[0] <= im_neix ** 2:
        dampen = (W.shape[0] * 1. / X_presented.shape[0])

    tau = np.float32((i * X_presented.shape[0] + k) * 1. / (max_its * X_presented.shape[0]))
    lr = np.float32(((1 - tau)) ** lrdec) * dampen
    so_lr = lrst * get_so_rate(tau)
    nei_len = np.int64(min(im_neix, W.shape[0]))

    dist_H, neilist = pairwise_and_neighbors(x, W, nei_len)
    if not nei_len == len(np.unique(neilist)):
        raise Exception('non_unique_elements')
    b = neilist[0]

    k += 1

    G[b] *= epsilon

    G[b][neilist] = 1.

    G[b][G[b] < min_strength] = 0

    # G[:, b][G[:, b] < min_strength] = 0
    neighbors = shp[(G[b] + G[:, b]) > 0]

    denom = dist_H[neilist[-1]]

    dist_H[neilist[0]] = 0

    epoch_vector = max_epochs_per_sample * ( (G[b] + G[:, b]) / 2. + 1)

    neg_epoch_vector = ns_rate * 1 * (1 -  (G[b] + G[:, b]) / 2.)

    '''x, so_lr, b, W, hdist_nei, Y, alpha, beta , lr,  rng_state):'''
    W[neighbors], Y = train_neighborhood(x, so_lr, b, neighbors.astype(np.int32), W[neighbors],
                                         dist_H[neighbors] / denom,
                                         Y, ns_rate, alpha, beta,
                                         lr, rng_state, epoch_vector.astype(np.int32),
                                         neg_epoch_vector.astype(np.int32))
    E_q[b] += dist_H[0]

    return W, Y, G, E_q, k, b, neilist, neighbors, lr


@numba.njit(
    'UniTuple(f4[:, :], 2)(f4[:], f4,  i4, i4[:],  f4[:,:], f4[:], f4[:,:], i4, f4, f4, f4, i8[:], i4[:], i4[:])',
    fastmath=True, )
def train_neighborhood(x, so_lr, b, neighbors, W, hdist_nei, Y, ns_rate, alpha, beta, lr, rng_state, epoch_vector,
                       neg_epoch_vector):
    hdists = hdist_nei
    y_b = Y[b]
    ''' Self Organizing '''
    sigma = 2.
    for j in range(W.shape[0]):
        hdist = hdists[j]
        if neighbors[j] == b:
            hdist = 0
        h_pull_grad = np.exp(-sigma * hdist)
        for i in range(x.shape[0]):
            W[j][i] += h_pull_grad * so_lr * (x[i] - W[j][i])

    '''negative embedding'''

    for j in range(W.shape[0]):

        epochs = epoch_vector[neighbors[j]]

        for p in range(neg_epoch_vector[neighbors[j]]):
            n = tau_rand_int(rng_state) % Y.shape[0]

            ldist_sq = rdist(y_b, Y[n])
            push_grad = (2 * beta)
            denom = 1 + alpha * pow(ldist_sq, beta)
            push_grad /= denom
            for i in range(y_b.shape[0]):
                if ldist_sq > 0.:
                    y_b[i] += positive_clip(push_grad / (ldist_sq + 0.001) * (y_b[i] - Y[n][i]),
                                            4) * lr
                elif b == n:
                    continue
                else:
                    y_b[i] += lr

        for e in range(epochs):

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

@numba.njit('f4[:,:](f4, f4[:,:], f4[:, :], i4, i4, i4, f4, f4, i8[:] )', fastmath=True, )
def embed_batch_epochs(lrst, Y, G, max_its, i_st, i_end, alpha, beta, rng_state):
    shp = np.arange(G.shape[0]).astype(np.int32)
    P_matrix = (G + G.T) / 2.
    tau = (i_st) * 1. / (max_its)
    tau_end = min(i_end, max_its) * 1. / max_its
    epochs_per_sample = ((P_matrix)).astype(np.int32)

    starting_lr = (1 - tau)
    ending_lr = (1 - tau_end)

    epochs_per_negative_sample = ((P_matrix.max() - P_matrix) * 5).astype(np.int32)
    totits = np.sum(epochs_per_sample)
    updated = 0

    pos_edges = np.where(epochs_per_sample)
    n_edges = len(pos_edges[0])
    for p in range(n_edges):
        j = pos_edges[0][p]
        lr = (starting_lr + (ending_lr - starting_lr) * (updated * 1. / totits))
        neighbors = shp[G[j] + G[:, j] > 0]
        for k in range(neighbors.shape[0]):
            epochs = epochs_per_sample[j][neighbors[k]]
            epochs_per_sample[j][neighbors[k]] = 0
            neg_epochs = int(epochs_per_negative_sample[j][neighbors[k]])

            for ep in range(epochs):
                for negs in range(neg_epochs):
                    n = tau_rand_int(rng_state) % Y.shape[0]
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
                            elif j == n:
                                continue
                            else:
                                Y[j][i] += lr
                ldist_sq = rdist(Y[j], Y[neighbors[k]])

                pull_grad = (2 * alpha * beta * pow(ldist_sq, beta - 1))
                denom = (1 + alpha * pow(ldist_sq, beta))
                pull_grad /= denom

                for i in range(Y[j].shape[0]):
                    Y[neighbors[k]][i] += positive_clip(pull_grad * (Y[j][i] - Y[neighbors[k]][i]), 4) * lr
                    Y[j][i] -= positive_clip(pull_grad * (Y[j][i] - Y[neighbors[k]][i]), 4) * lr
                updated += 1

    return Y

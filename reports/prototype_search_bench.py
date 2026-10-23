"""Compare exact nearest-prototype search methods on SONG's fitted MNIST prototypes."""
import json
from pathlib import Path
import sys
import time

import numba
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import pairwise_distances_argmin_min

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from song.song import SONG
from song.util import get_closest, get_closest_for_inputs, sq_eucl_opt

with np.load(ROOT / 'data' / 'mnist.npz') as archive:
    X_train = archive['x_train'].reshape(-1, 784).astype(np.float32) / 255
SONG(so_steps=2, final_vector_count=16, dispersion_method='none').fit_transform(X_train[:80])
model = SONG(so_steps=100, random_seed=42, dispersion_method='none')
model.fit(X_train)
X_pc = model._get_XPCA(X_train).astype(np.float32)
W_pc = model.pca.transform(model.W).astype(np.float32)
queries = X_pc[:10000]

@numba.njit
def top3_rows(distances):
    result = np.empty((len(distances), 3), np.int32)
    for i in range(len(distances)):
        result[i] = get_closest(distances[i], 3)
    return result

matrix = sq_eucl_opt(queries[:1000], W_pc)
top3_rows(matrix)

def current(Q):
    return get_closest_for_inputs(Q, W_pc)[0]

def blas(Q):
    w_norm = np.sum(W_pc * W_pc, axis=1)
    result = np.empty(len(Q), dtype=np.int64)
    for st in range(0, len(Q), 1000):
        chunk = Q[st:st + 1000]
        distances = np.sum(chunk * chunk, axis=1)[:, None] + w_norm[None, :] - 2 * (chunk @ W_pc.T)
        result[st:st + len(chunk)] = distances.argmin(axis=1)
    return result

tree = cKDTree(W_pc)
def ckdtree(Q):
    return tree.query(Q, k=1, workers=4)[1]

def sklearn_argmin(Q):
    return pairwise_distances_argmin_min(Q, W_pc, metric='euclidean')[0]

methods = {'SONG Numba brute force': current, 'BLAS matrix product': blas,
           'SciPy cKDTree (4 workers)': ckdtree,
           'scikit-learn ArgKmin': sklearn_argmin}
reference = current(queries)
rows = []
for name, method in methods.items():
    for _ in range(2):
        method(queries[:1000])
    times = []
    result = None
    for _ in range(3):
        start = time.perf_counter()
        result = method(queries)
        times.append(time.perf_counter() - start)
    mismatches = int(np.count_nonzero(result != reference))
    row = {'method': name, 'queries': len(queries), 'prototypes': len(W_pc),
           'features': W_pc.shape[1], 'median_seconds': float(np.median(times)),
           'mismatch_count': mismatches}
    rows.append(row)
    print(row, flush=True)

start = time.perf_counter()
top3_rows(matrix)
top3_seconds = time.perf_counter() - start
print('Training top-3 selection for 1,000 precomputed distance rows:', top3_seconds, 'seconds')

pd.DataFrame(rows).to_csv(ROOT / 'reports' / 'prototype_search_bench.csv', index=False)
(ROOT / 'reports' / 'prototype_search_bench.json').write_text(json.dumps({
    'train_size': 60000, 'song_steps': 100, 'numba_threads': numba.get_num_threads(),
    'queries_benchmarked': len(queries), 'prototypes': len(W_pc), 'features': W_pc.shape[1],
    'top3_precomputed_1000_rows_s': top3_seconds,
    'scipy_tree_build_s_excluded': True,
}, indent=2) + '\n')

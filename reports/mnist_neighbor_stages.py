"""Time SONG nearest-prototype lookup and UMAP neighbor graph stages on full MNIST."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import umap.umap_ as umap_module

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from song.song import SONG

with np.load(ROOT / 'data' / 'mnist.npz') as archive:
    X_train = archive['x_train'].reshape(-1, 784).astype(np.float32) / 255
SONG(so_steps=2, final_vector_count=16, dispersion_method='none').fit_transform(X_train[:80])
model = SONG(so_steps=100, random_seed=42, dispersion_method='none')
model.fit(X_train)

start = time.perf_counter()
Y_core = model.transform(X_train)
core_lookup = time.perf_counter() - start
start = time.perf_counter()
X_pc = model._get_XPCA(X_train).astype(np.float32)
pca_transform = time.perf_counter() - start

names = ['nearest_neighbors', 'fuzzy_simplicial_set', 'simplicial_set_embedding']
originals = {name: getattr(umap_module, name) for name in names}
totals = defaultdict(float)
counts = defaultdict(int)

def wrapper(name, function):
    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            totals[name] += time.perf_counter() - start
            counts[name] += 1
    return timed

for name in names:
    setattr(umap_module, name, wrapper(name, originals[name]))
try:
    start = time.perf_counter()
    model.get_umap_dispersion(Y_core, X_pc, np.empty((0, 50), dtype=np.float32))
    dispersion = time.perf_counter() - start
finally:
    for name, function in originals.items():
        setattr(umap_module, name, function)

result = {'train_size': len(X_train), 'song_steps': 100, 'core_lookup_s': core_lookup,
          'pca_transform_s': pca_transform, 'umap_dispersion_s': dispersion,
          'umap_stage_s': dict(totals), 'umap_stage_calls': dict(counts),
          'umap_other_s': dispersion - sum(totals.values())}
(ROOT / 'reports' / 'mnist_neighbor_stages.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))

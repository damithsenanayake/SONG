"""Benchmark the existing Numba distance kernel on MNIST-like matrix shapes."""
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from song.util import sq_eucl_opt

rng = np.random.default_rng(42)
thread_counts = [count for count in [1, 2, 4, 8] if count <= numba.config.NUMBA_NUM_THREADS]
rows = []
for n_samples, n_prototypes in [(100, 1050), (1000, 1050), (1000, 1500)]:
    A = rng.normal(size=(n_samples, 50)).astype(np.float32)
    B = rng.normal(size=(n_prototypes, 50)).astype(np.float32)
    numba.set_num_threads(1)
    reference = sq_eucl_opt(A, B)
    for threads in thread_counts:
        numba.set_num_threads(threads)
        for _ in range(3):
            sq_eucl_opt(A, B)
        timings = []
        for _ in range(35):
            start = time.perf_counter()
            result = sq_eucl_opt(A, B)
            timings.append(time.perf_counter() - start)
        np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-5)
        rows.append(dict(samples=n_samples, prototypes=n_prototypes, features=50,
                         threads=threads, median_ms=1000*np.median(timings),
                         p25_ms=1000*np.quantile(timings, .25),
                         p75_ms=1000*np.quantile(timings, .75)))
        print(rows[-1], flush=True)
results = pd.DataFrame(rows)
results.to_csv(ROOT / 'reports' / 'mnist_distance_threads.csv', index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, n_samples in zip(axes, [100, 1000]):
    subset = results[(results.samples == n_samples) & (results.prototypes == 1050)]
    ax.plot(subset.threads, subset.median_ms, marker='o')
    ax.set(xticks=thread_counts, xlabel='Numba threads', ylabel='Median milliseconds',
           title=f'{n_samples} samples × 1050 prototypes × 50D')
    ax.grid(alpha=.25)
fig.suptitle('Existing distance kernel: parallel row blocks')
fig.savefig(ROOT / 'reports' / 'figures' / 'mnist_distance_threads.png', dpi=170)
plt.close(fig)

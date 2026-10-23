"""Profile SONG on the canonical 60k/10k MNIST split; run from repo root.

NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python reports/mnist_profile.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
from urllib.request import urlretrieve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import song.song as song_module
from song.song import SONG

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'mnist.npz'
SOURCE = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
FIGURES = ROOT / 'reports' / 'figures'


def load_mnist(train_size: int, test_size: int):
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        urlretrieve(SOURCE, DATA)
    with np.load(DATA) as archive:
        assert archive['x_train'].shape == (60000, 28, 28)
        assert archive['x_test'].shape == (10000, 28, 28)
        X_train = archive['x_train'][:train_size].reshape(-1, 784).astype(np.float32) / 255
        y_train = archive['y_train'][:train_size].astype(np.int8)
        X_test = archive['x_test'][:test_size].reshape(-1, 784).astype(np.float32) / 255
        y_test = archive['y_test'][:test_size].astype(np.int8)
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    return X_train, y_train, X_test, y_test, digest


class PeakRSS:
    def __init__(self):
        self.process = psutil.Process()
        self.maximum = self.process.memory_info().rss
        self.active = True
        self.worker = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while self.active:
            self.maximum = max(self.maximum, self.process.memory_info().rss)
            time.sleep(0.05)

    def __enter__(self):
        self.worker.start()
        return self

    def __exit__(self, *_):
        self.active = False
        self.worker.join()
        self.maximum = max(self.maximum, self.process.memory_info().rss)


def timed_call(name, function, totals, counts):
    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            totals[name] += time.perf_counter() - start
            counts[name] += 1
    return timed


def save_scatter(path, train, test, y_train, y_test, title):
    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(train), min(12000, len(train)), replace=False)
    test_idx = rng.choice(len(test), min(5000, len(test)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, points, labels, subtitle in [(axes[0], train[train_idx], y_train[train_idx], 'Train'),
                                         (axes[1], test[test_idx], y_test[test_idx], 'Held-out test')]:
        ax.scatter(points[:, 0], points[:, 1], c=labels, cmap='tab10', vmin=0, vmax=9,
                   s=3, alpha=0.55, rasterized=True)
        ax.set_title(f'{subtitle} ({len(points):,} shown)')
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_confusion(path, y_true, y_pred, title):
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(10), normalize='true')
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap='Greens')
    ax.set(xticks=np.arange(10), yticks=np.arange(10), xlabel='Predicted digit',
           ylabel='True digit', title=title)
    fig.colorbar(image, ax=ax, label='Fraction of true class')
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-size', type=int, default=60000)
    parser.add_argument('--test-size', type=int, default=10000)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--skip-dispersion', action='store_true')
    parser.add_argument('--run-label', default='', help='suffix for output filenames')
    args = parser.parse_args()
    if args.run_label and not args.run_label.replace('_', '').isalnum():
        parser.error('run-label must contain only letters, digits, and underscores')
    def named(folder, filename):
        path = folder / filename
        return path.with_name(path.stem + ('_' + args.run_label if args.run_label else '') + path.suffix)
    assert 2 <= args.train_size <= 60000 and 1 <= args.test_size <= 10000
    FIGURES.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    X_train, y_train, X_test, y_test, data_hash = load_mnist(args.train_size, args.test_size)
    load_s = time.perf_counter() - t0
    print(f'Loaded MNIST: train={len(X_train):,}, test={len(X_test):,}, load={load_s:.1f}s', flush=True)

    # Compile the relevant kernels before timing the full fit.
    warm = SONG(so_steps=2, final_vector_count=16, dispersion_method='none', random_seed=42)
    warm.fit_transform(X_train[:80])
    del warm
    import gc
    gc.collect()
    print('Numba warm-up complete', flush=True)

    totals = defaultdict(float)
    counts = defaultdict(int)
    names = ['sq_eucl_opt', 'train_for_batch_batch', 'bulk_grow_with_drifters', 'embed_batch_epochs']
    originals = {name: getattr(song_module, name) for name in names}
    for name in names:
        setattr(song_module, name, timed_call(name, originals[name], totals, counts))

    model = SONG(so_steps=args.steps, random_seed=42, dispersion_method='none')
    model._train_pca = timed_call('PCA fit', model._train_pca, totals, counts)
    model._get_XPCA = timed_call('PCA transform', model._get_XPCA, totals, counts)
    metrics = {}
    with PeakRSS() as rss:
        start = time.perf_counter()
        model.fit(X_train)
        metrics['song_fit_s'] = time.perf_counter() - start
        print(f'SONG fit: {metrics["song_fit_s"]:.1f}s, prototypes={len(model.W):,}', flush=True)

        start = time.perf_counter()
        train_assignments, _ = model.get_representatives(X_train)
        test_assignments, _ = model.get_representatives(X_test)
        metrics['representative_assignments_s'] = time.perf_counter() - start
        train_assignments = np.asarray(train_assignments)
        test_assignments = np.asarray(test_assignments)
        votes = np.zeros((len(model.W), 10), dtype=np.int64)
        np.add.at(votes, (train_assignments, y_train), 1)
        default_label = np.bincount(y_train).argmax()
        prototype_labels = votes.argmax(axis=1)
        prototype_labels[votes.sum(axis=1) == 0] = default_label
        prototype_predictions = prototype_labels[test_assignments]
        metrics['prototype_test_accuracy'] = accuracy_score(y_test, prototype_predictions)
        metrics['prototype_test_balanced_accuracy'] = balanced_accuracy_score(y_test, prototype_predictions)
        metrics['occupied_prototypes'] = int(np.count_nonzero(votes.sum(axis=1)))
        save_confusion(named(FIGURES, 'mnist_prototype_confusion.png'), y_test, prototype_predictions,
                       f'SONG prototype labels, test accuracy {metrics["prototype_test_accuracy"]:.3f}')
        print(f'Prototype test accuracy: {metrics["prototype_test_accuracy"]:.3f}', flush=True)

        start = time.perf_counter()
        Y_train_core = model.transform(X_train)
        metrics['core_train_transform_s'] = time.perf_counter() - start
        start = time.perf_counter()
        Y_test_core = model.transform(X_test)
        metrics['core_test_transform_s'] = time.perf_counter() - start
        core_probe = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
        start = time.perf_counter()
        core_probe.fit(Y_train_core, y_train)
        core_predictions = core_probe.predict(Y_test_core)
        metrics['core_probe_s'] = time.perf_counter() - start
        metrics['core_2d_test_accuracy'] = accuracy_score(y_test, core_predictions)
        metrics['core_2d_test_balanced_accuracy'] = balanced_accuracy_score(y_test, core_predictions)
        save_scatter(named(FIGURES, 'mnist_song_core_train_test.png'), Y_train_core, Y_test_core,
                     y_train, y_test, 'SONG core: shared prototype coordinate frame')
        print(f'Core 2D test accuracy: {metrics["core_2d_test_accuracy"]:.3f}', flush=True)

        start = time.perf_counter()
        X_train_pc = model._get_XPCA(X_train).astype(np.float32)
        X_test_pc = model._get_XPCA(X_test).astype(np.float32)
        pca_probe = LogisticRegression(max_iter=200, random_state=42)
        pca_probe.fit(X_train_pc, y_train)
        pca_predictions = pca_probe.predict(X_test_pc)
        metrics['pca_linear_probe_s'] = time.perf_counter() - start
        metrics['pca_linear_test_accuracy'] = accuracy_score(y_test, pca_predictions)
        print(f'PCA-50 linear test accuracy: {metrics["pca_linear_test_accuracy"]:.3f}', flush=True)

        if not args.skip_dispersion:
            model.dispersion_method = 'UMAP'
            start = time.perf_counter()
            Y_train_disp = model.transform(X_train)
            metrics['dispersion_train_transform_s'] = time.perf_counter() - start
            print(f'Train dispersion: {metrics["dispersion_train_transform_s"]:.1f}s', flush=True)
            start = time.perf_counter()
            Y_test_disp = model.transform(X_test)
            metrics['dispersion_test_transform_s'] = time.perf_counter() - start
            print(f'Test dispersion: {metrics["dispersion_test_transform_s"]:.1f}s', flush=True)
            disp_probe = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
            start = time.perf_counter()
            disp_probe.fit(Y_train_disp, y_train)
            disp_predictions = disp_probe.predict(Y_test_disp)
            metrics['dispersion_cross_frame_probe_s'] = time.perf_counter() - start
            metrics['dispersion_cross_frame_test_accuracy'] = accuracy_score(y_test, disp_predictions)
            save_scatter(named(FIGURES, 'mnist_song_dispersion_train_test.png'), Y_train_disp, Y_test_disp,
                         y_train, y_test, 'SONG dispersion: train and test refit UMAP separately')
            print(f'Dispersion cross-frame accuracy (diagnostic): {metrics["dispersion_cross_frame_test_accuracy"]:.3f}', flush=True)
            np.savez_compressed(named(ROOT / 'data', 'mnist_profile_embeddings.npz'),
                                train_core=Y_train_core, test_core=Y_test_core,
                                train_dispersion=Y_train_disp, test_dispersion=Y_test_disp)
        else:
            np.savez_compressed(named(ROOT / 'data', 'mnist_profile_embeddings.npz'),
                                train_core=Y_train_core, test_core=Y_test_core)
        metrics['peak_rss_gib'] = rss.maximum / 2**30

    for name, original in originals.items():
        setattr(song_module, name, original)

    stage_rows = [{'stage': name, 'seconds': seconds, 'calls': counts[name]}
                  for name, seconds in sorted(totals.items(), key=lambda item: -item[1])]
    stage_rows.append({'stage': 'fit remainder', 'seconds': max(0, metrics['song_fit_s'] -
                       sum(totals.get(name, 0) for name in [*names, 'PCA fit', 'PCA transform'])), 'calls': 1})
    pd.DataFrame(stage_rows).to_csv(named(ROOT / 'reports', 'mnist_stage_profile.csv'), index=False)
    fit_stages = [row for row in stage_rows if row['stage'] in [*names, 'PCA fit', 'PCA transform', 'fit remainder']]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.barh([row['stage'] for row in fit_stages][::-1], [row['seconds'] for row in fit_stages][::-1], color='#31786c')
    ax.set(xlabel='Seconds', title=f'SONG fit time, MNIST {len(X_train):,} train images, {args.steps} steps')
    fig.savefig(named(FIGURES, 'mnist_song_fit_profile.png'), dpi=170)
    plt.close(fig)

    summary = {'data_source': SOURCE, 'data_sha256': data_hash,
               'train_size': len(X_train), 'test_size': len(X_test), 'features': 784,
               'normalization': 'float32 pixels / 255', 'song_steps': args.steps,
               'song_prototypes': len(model.W), 'pc_components': model.pc_components,
               'dispersion_epochs': model.um_epochs, 'numba_threads': os.environ.get('NUMBA_NUM_THREADS'),
               'blas_threads': os.environ.get('OPENBLAS_NUM_THREADS'),
               'platform': platform.platform(), 'metrics': metrics,
               'stage_times': dict(totals), 'stage_calls': dict(counts)}
    (named(ROOT / 'reports', 'mnist_profile.json')).write_text(json.dumps(summary, indent=2) + '\n')
    print('Saved MNIST profile and plots; peak RSS:', round(metrics['peak_rss_gib'], 2), 'GiB', flush=True)


if __name__ == '__main__':
    main()

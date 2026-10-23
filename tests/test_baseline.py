"""Small regression checks for the current UMAP_dispersion implementation."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from song.song import SONG
from song.util import get_closest_for_inputs


def sample_data():
    X, _ = make_blobs(n_samples=80, n_features=8, centers=3, random_state=7)
    return ((X - X.min(0)) / (X.max(0) - X.min(0))).astype(np.float32)


def test_static_fit_transform_returns_finite_coordinates():
    X = sample_data()
    model = SONG(so_steps=3, final_vector_count=16, dispersion_method='none')
    Y = model.fit_transform(X)
    assert Y.shape == (len(X), 2)
    assert np.isfinite(Y).all()


def test_incremental_fit_accepts_a_second_batch():
    X = sample_data()
    model = SONG(so_steps=2, final_vector_count=16, dispersion_method='none')
    model.fit(X[:40])
    model.fit(X[40:])
    assert model.trained


@pytest.mark.xfail(strict=True, reason='nearest distances are truncated to int64')
def test_nearest_distance_preserves_fractional_values():
    query = np.array([[0.3, 0.3]], dtype=np.float32)
    prototypes = np.array([[0.0, 0.0]], dtype=np.float32)
    _, distances = get_closest_for_inputs(query, prototypes)
    np.testing.assert_allclose(distances[0], 0.18, atol=1e-6)


@pytest.mark.parametrize('n_features', [51, 64, 100, 101])
def test_pca_distance_uses_projected_prototypes(monkeypatch, n_features):
    from sklearn.datasets import load_digits
    import song.song as song_module

    X = (load_digits().data[:80] / 16).astype(np.float32)
    if n_features > X.shape[1]:
        X = np.column_stack([X, np.random.default_rng(7).random(
            (len(X), n_features - X.shape[1]), dtype=np.float32)])
    else:
        X = X[:, :n_features]
    model = SONG(so_steps=2, final_vector_count=16, pc_components=50,
                 dispersion_method='none')
    original_distances = song_module.sq_eucl_opt
    original_train = song_module.train_for_batch_batch
    pending = []
    checked = []

    def record_distances(samples, prototypes):
        assert samples.shape[1] == prototypes.shape[1] == 50
        pending.append(prototypes.copy())
        return original_distances(samples, prototypes)

    def check_training_basis(*args, **kwargs):
        raw_weights = args[7]
        np.testing.assert_allclose(pending.pop(0), model.pca.transform(raw_weights),
                                   rtol=1e-5, atol=1e-5)
        checked.append(True)
        return original_train(*args, **kwargs)

    monkeypatch.setattr(song_module, 'sq_eucl_opt', record_distances)
    monkeypatch.setattr(song_module, 'train_for_batch_batch', check_training_basis)
    model.fit(X)
    assert checked and not pending
    assert model.pca.n_components_ == 50


def test_corrected_pc_requires_matching_pca_model():
    X = np.zeros((80, 64), dtype=np.float32)
    model = SONG(so_steps=2, pc_components=50)
    with pytest.raises(ValueError, match='fitted PCA model'):
        model.fit(X, corrected_PC=np.zeros((80, 50), dtype=np.float32))


def test_dispersion_respects_epoch_and_learning_rate_settings(monkeypatch):
    import song.song as song_module

    passed = {}

    class FakeUMAP:
        def __init__(self, **kwargs):
            passed.update(kwargs)

        def fit_transform(self, X):
            return X[:, :2]

    monkeypatch.setattr(song_module, 'UMAP', FakeUMAP)
    model = SONG(um_epochs=100, um_lr=0.02)
    X = np.ones((4, 3), dtype=np.float32)
    model.get_umap_dispersion(np.zeros((4, 2), dtype=np.float32), X, X)
    assert passed['n_epochs'] == 100
    assert passed['learning_rate'] == 0.02

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


@pytest.mark.xfail(strict=True, reason='second fit uses removed np.int alias')
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

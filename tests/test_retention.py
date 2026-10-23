"""Checks for the opt-in incremental layout KL objective."""
import numpy as np

from song.retention import neighborhood_temperature, position_kl_loss_and_grad
from song.song import SONG


def test_position_kl_gradient_matches_finite_difference():
    reference = np.array([[0., 0.], [2., 0.], [0., 1.], [2., 2.]], dtype=np.float64)
    current = reference + np.array([[.1, -.3], [-.2, .1], [.4, .2], [-.1, -.2]])
    tau = neighborhood_temperature(reference)
    loss, gradient, kl, position_kl = position_kl_loss_and_grad(
        current, reference, tau, kl_weight=.7, position_weight=.3)
    assert loss > 0 and kl > 0 and position_kl > 0
    step = 1e-6
    for i in range(len(current)):
        for j in range(current.shape[1]):
            plus = current.copy(); plus[i, j] += step
            minus = current.copy(); minus[i, j] -= step
            numerical = (position_kl_loss_and_grad(plus, reference, tau, .7, .3)[0] -
                         position_kl_loss_and_grad(minus, reference, tau, .7, .3)[0]) / (2*step)
            np.testing.assert_allclose(gradient[i, j], numerical, rtol=1e-5, atol=1e-7)


def test_incremental_fit_optimizes_retention_loss():
    rng = np.random.default_rng(7)
    old = rng.normal(0, .1, (80, 8)).astype(np.float32)
    new = rng.normal(.7, .1, (80, 8)).astype(np.float32)
    model = SONG(so_steps=3, final_vector_count=20, dispersion_method='none',
                 retention_kl_weight=1., retention_position_weight=1.,
                 retention_anchor_count=32, retention_maxiter=20)
    model.fit(old)
    assert len(model._retention_anchor_X) == 32
    model.fit(new)
    result = model.retention_result_
    assert result['anchor_count'] == 32
    assert result['loss_after'] < result['loss_before']
    assert np.isfinite(model.Y).all()


def test_retention_changes_coordinates_but_not_prototype_weights():
    rng = np.random.default_rng(19)
    old = rng.normal(0, .1, (90, 8)).astype(np.float32)
    new = rng.normal(.6, .1, (90, 8)).astype(np.float32)
    common = dict(so_steps=3, final_vector_count=20, random_seed=4,
                  dispersion_method='none')
    plain = SONG(**common)
    protected = SONG(**common, retention_kl_weight=1.,
                     retention_position_weight=1., retention_anchor_count=32)
    plain.fit(old)
    protected.fit(old)
    np.testing.assert_array_equal(plain.W, protected.W)
    np.testing.assert_array_equal(plain.Y, protected.Y)
    plain.fit(new)
    protected.fit(new)
    np.testing.assert_array_equal(plain.W, protected.W)
    assert not np.array_equal(plain.Y, protected.Y)

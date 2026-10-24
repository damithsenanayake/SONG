"""Distributional retention objective for incremental SONG layouts."""

import numpy as np
from scipy.special import logsumexp


def _squared_distances(positions):
    squared_norms = np.sum(positions * positions, axis=1)
    distances = squared_norms[:, None] + squared_norms[None, :] - 2 * positions @ positions.T
    return np.maximum(distances, 0.0)


def neighborhood_temperature(positions):
    """A fixed, global bandwidth from the reference anchor layout."""
    positions = np.asarray(positions, dtype=np.float64)
    distances = _squared_distances(positions)
    positive = distances[np.triu_indices(len(positions), 1)]
    positive = positive[positive > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def position_kl_loss_and_grad(current, reference, temperature, kl_weight=1.0,
                              position_weight=0.0):
    """Mean KL of anchor-neighbor distributions, plus Gaussian position KL.

    The reference and current rows represent the same old anchors. A fixed
    temperature prevents scale changes from silently changing the objective.
    The Gaussian term uses equal isotropic covariance ``temperature * I``.
    """
    current = np.asarray(current, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if current.shape != reference.shape or current.ndim != 2 or len(current) < 2:
        raise ValueError('current and reference need matching (n>=2, dimensions) arrays')
    if temperature <= 0 or kl_weight < 0 or position_weight < 0:
        raise ValueError('temperature must be positive and weights nonnegative')
    n = len(current)
    d_ref = _squared_distances(reference)
    d_cur = _squared_distances(current)
    np.fill_diagonal(d_ref, np.inf)
    np.fill_diagonal(d_cur, np.inf)
    logq = -d_ref / temperature
    logp = -d_cur / temperature
    logq -= logsumexp(logq, axis=1, keepdims=True)
    logp -= logsumexp(logp, axis=1, keepdims=True)
    q = np.exp(logq)
    p = np.exp(logp)
    mask = q > 0
    kl = float(np.sum(q[mask] * (logq[mask] - logp[mask])) / n)
    residual = p - q
    symmetric = residual + residual.T
    gradient = (-2.0 / (temperature * n)) * (
        symmetric.sum(axis=1)[:, None] * current - symmetric @ current
    )
    delta = current - reference
    position_kl = float(np.sum(delta * delta) / (2.0 * temperature * n))
    gradient = kl_weight * gradient + position_weight * delta / (temperature * n)
    loss = kl_weight * kl + position_weight * position_kl
    return float(loss), gradient, kl, position_kl

"""Risk-controlled simulation triage (Section 4.9).

A prediction is accepted when its uncertainty score ``u(x)`` is at most a threshold ``t``; otherwise the
rock is sent to pore-scale simulation. A *bad accept* is an accepted prediction whose error exceeds the
tolerance, ``|mu(x) - y| > log10(f)`` for a factor ``f`` on permeability. The risk of a threshold is
``R(t) = P(u(X) <= t, bad)`` (Eq. 12).

* ``ltt_threshold`` (Learn-then-Test) guarantees ``R(t_hat) <= eps`` with probability at least ``1 - delta``.
* ``crc_threshold`` (conformal risk control) bounds the expected risk instead.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def bad_accept_labels(mu, y, tol_factor: float) -> np.ndarray:
    """True where a log10 prediction misses by more than the multiplicative tolerance ``tol_factor``."""
    return np.abs(np.asarray(mu) - np.asarray(y)) > np.log10(tol_factor)


def threshold_grid(u_ref, n_grid: int = 200) -> np.ndarray:
    """Candidate thresholds fixed before calibration: quantiles of the score on a disjoint monitoring slice."""
    return np.unique(np.quantile(u_ref, np.linspace(0, 1, n_grid + 1)[1:]))


def ltt_threshold(u, bad, eps: float, delta: float, grid) -> float:
    """Learn-then-Test threshold with fixed-sequence testing over an ascending grid.

    For each candidate t, H0: R(t) > eps is tested with the exact binomial p-value. Testing stops at the
    first t not rejected; the largest rejected t is returned (or -inf if none).
    """
    u = np.asarray(u, float)
    bad = np.asarray(bad, bool)
    grid = np.asarray(grid, float)
    n = len(u)
    o = np.argsort(u)
    us, cb = u[o], np.cumsum(bad[o].astype(int))
    pos = np.searchsorted(us, grid, side="right")
    k = np.where(pos > 0, cb[np.maximum(pos - 1, 0)], 0)
    ok = stats.binom.cdf(k, n, eps) <= delta
    if not ok[0]:
        return -np.inf
    first_fail = int(np.argmin(ok)) if (~ok).any() else len(ok)
    return float(grid[first_fail - 1])


def crc_threshold(u, bad, eps: float) -> float:
    """Conformal risk control: largest t with (bad accepts + 1) / (n + 1) <= eps."""
    u = np.asarray(u, float)
    n = len(u)
    o = np.argsort(u)
    us, cb = u[o], np.cumsum(np.asarray(bad, bool)[o].astype(int))
    ok = (cb + 1) / (n + 1) <= eps
    return float(us[np.where(ok)[0].max()]) if ok.any() else -np.inf


def evaluate(u, bad, t: float) -> dict:
    """Share accepted, bad-accept rate (over all volumes) and error rate among accepted volumes."""
    u = np.asarray(u)
    bad = np.asarray(bad, bool)
    acc = u <= t
    return dict(accepted=float(acc.mean()), bad_accept=float((acc & bad).mean()),
                err_among_accepted=float(bad[acc].mean()) if acc.any() else 0.0)

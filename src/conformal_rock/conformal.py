"""Conformal predictors used in the paper (Section 4.2 and 4.4).

All intervals are property-wise: each property ``j`` gets its own marginal guarantee
``P(Y_j in C_j(X)) >= 1 - alpha`` (Eq. 1). Joint coverage over several properties is a separate
construction (``joint_box``).

Prediction dictionaries
-----------------------
Functions take dictionaries with NumPy arrays, as stored in the released ``preds/*.npz`` files:

``y``   (n, P)        reference values (transformed targets)
``mu``  (n, P)        point predictions
``sig`` (n, P)        predicted standard deviation (Gaussian head)
``q``   (n, P, Q)     predicted quantiles at ``Config.q_levels``
``mc_mu``, ``mc_vm``, ``mc_mv`` (n, P)  MC-dropout mean, variance of means, mean predicted variance
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .config import CFG


# ----------------------------------------------------------------------------------------------
# Finite-sample threshold
# ----------------------------------------------------------------------------------------------
def conformal_quantile(scores, alpha: float) -> float:
    """The k-th smallest calibration score with k = ceil((n+1)(1-alpha)); +inf if k > n (Eq. 2).

    Ties are kept as they are, which keeps the guarantee and makes it slightly conservative.
    """
    s = np.asarray(scores, float)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha) - 1e-9))
    if n == 0 or k > n:
        return float("inf")
    return float(np.partition(s, k - 1)[k - 1])


cq = conformal_quantile  # short alias used in the notebook


def normal_quantile(alpha: float) -> float:
    """Two-sided normal quantile z_{1-alpha/2}, used by the heuristic baselines."""
    return float(stats.norm.ppf(1 - alpha / 2))


def quantile_pair(alpha: float, q_lo=None) -> tuple[int, int]:
    """Indices of the quantile-head pair whose nominal coverage is closest to 1 - alpha."""
    q_lo = CFG.q_lo if q_lo is None else q_lo
    j = int(np.argmin([abs((1 - 2 * t) - (1 - alpha)) for t in q_lo]))
    return j, j + len(q_lo)


def group_edges(train_mu, n_groups: int) -> np.ndarray:
    """Mondrian group edges: quantiles of the training-set predictions (fixed before calibration)."""
    return np.quantile(train_mu, np.linspace(0, 1, n_groups + 1)[1:-1])


# ----------------------------------------------------------------------------------------------
# Property-wise intervals
# ----------------------------------------------------------------------------------------------
def _sorted_pair(d, j, a, b):
    lo = np.minimum(d["q"][:, j, a], d["q"][:, j, b])
    hi = np.maximum(d["q"][:, j, a], d["q"][:, j, b])
    return lo, hi


def interval(method: str, cal: dict, te: dict, j: int, alpha: float, tr: dict | None = None,
             eps_rel: float | None = None, n_groups: int | None = None):
    """Prediction interval for property ``j`` on the test dictionary ``te``.

    Parameters
    ----------
    method : one of ``split``, ``norm``, ``cqr``, ``cqr_fixed``, ``mondrian`` (conformal) or
             ``gauss``, ``mcd``, ``mcd_tot`` (uncalibrated heuristics).
    cal, te, tr : prediction dictionaries for calibration, test and training volumes.
             ``tr`` is needed by ``norm`` (scale floor) and ``mondrian`` (group edges).

    Returns
    -------
    lo, hi : arrays of interval endpoints.
    info : dict with bookkeeping (CQR inversions and crossings, Mondrian fallbacks).
    """
    eps_rel = CFG.eps_rel if eps_rel is None else eps_rel
    n_groups = CFG.n_groups if n_groups is None else n_groups
    yc, mc, mt = cal["y"][:, j], cal["mu"][:, j], te["mu"][:, j]
    info: dict = {}
    if method == "split":                                   # Eq. (3)
        q = cq(np.abs(yc - mc), alpha)
        lo, hi = mt - q, mt + q
    elif method == "norm":                                  # Eq. (4)
        eps = eps_rel * tr["y"][:, j].std()
        q = cq(np.abs(yc - mc) / (cal["sig"][:, j] + eps), alpha)
        w = q * (te["sig"][:, j] + eps)
        lo, hi = mt - w, mt + w
    elif method in ("cqr", "cqr_fixed"):                    # Eq. (5)
        a, b = quantile_pair(alpha if method == "cqr" else 0.10)
        cl, ch = _sorted_pair(cal, j, a, b)
        tl, th = _sorted_pair(te, j, a, b)
        q = cq(np.maximum(cl - yc, yc - ch), alpha)
        lo, hi = tl - q, th + q
        inv = lo > hi                                       # a negative q can invert the band
        mid = 0.5 * (lo + hi)
        lo, hi = np.where(inv, mid, lo), np.where(inv, mid, hi)
        info = dict(n_inverted=int(inv.sum()), n_crossed=int((te["q"][:, j, a] > te["q"][:, j, b]).sum()), qhat=q)
    elif method == "mondrian":                              # Eq. (6)
        e = group_edges(tr["mu"][:, j], n_groups)
        gc = np.searchsorted(e, mc, side="right")
        gt = np.searchsorted(e, mt, side="right")
        res = np.abs(yc - mc)
        pooled = cq(res, alpha)
        qg = np.empty(n_groups)
        fallback = 0
        for g in range(n_groups):
            qg[g] = cq(res[gc == g], alpha)
            if not np.isfinite(qg[g]):
                qg[g] = pooled
                fallback += 1
        lo, hi = mt - qg[gt], mt + qg[gt]
        info = dict(n_fallback_groups=fallback, min_group_n=int(np.bincount(gc, minlength=n_groups).min()))
    elif method == "gauss":
        w = normal_quantile(alpha) * te["sig"][:, j]
        lo, hi = mt - w, mt + w
    elif method == "mcd":                                   # Eq. (7), spread of means only
        w = normal_quantile(alpha) * np.sqrt(te["mc_vm"][:, j])
        lo, hi = te["mc_mu"][:, j] - w, te["mc_mu"][:, j] + w
    elif method == "mcd_tot":                               # Eq. (7), total variance
        w = normal_quantile(alpha) * np.sqrt(te["mc_vm"][:, j] + te["mc_mv"][:, j])
        lo, hi = te["mc_mu"][:, j] - w, te["mc_mu"][:, j] + w
    else:
        raise ValueError(f"unknown method {method!r}")
    return lo, hi, info


def coverage(y, lo, hi) -> float:
    """Share of values inside their interval."""
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


# ----------------------------------------------------------------------------------------------
# Exact finite-sample law of split-conformal coverage (Eq. 9)
# ----------------------------------------------------------------------------------------------
def coverage_law_params(n_cal: int, alpha: float) -> tuple[int, int]:
    """Beta parameters (a, b) of the coverage law: Beta(k, n+1-k) with k = ceil((n+1)(1-alpha))."""
    k = int(np.ceil((n_cal + 1) * (1 - alpha) - 1e-9))
    return k, n_cal + 1 - k


def coverage_law(n_cal: int, n_test: int, alpha: float) -> tuple[float, float]:
    """Exact mean and standard deviation of empirical test coverage (beta-binomial law)."""
    a, b = coverage_law_params(n_cal, alpha)
    m = a / (a + b)
    v = a * b / ((a + b) ** 2 * (a + b + 1))
    return float(m), float(np.sqrt(v + (m - (v + m ** 2)) / n_test))


def coverage_law_pmf(n_cal: int, n_test: int, alpha: float) -> np.ndarray:
    """Probability of h covered test points, h = 0..n_test, under the beta-binomial law."""
    a, b = coverage_law_params(n_cal, alpha)
    return stats.betabinom.pmf(np.arange(n_test + 1), n_test, a, b)


def discrete_ks_test(h_obs, n_cal: int, n_test: int, alpha: float, n_mc: int = 2000, seed: int = 0):
    """Kolmogorov-Smirnov statistic of observed covered counts against the exact law.

    The p-value is computed by Monte Carlo from the exact law (Phipson & Smyth, 2010), which is valid
    for discrete distributions where the classical KS p-value is not.
    """
    h_obs = np.asarray(h_obs, int)
    pmf = coverage_law_pmf(n_cal, n_test, alpha)
    cdf = np.cumsum(pmf)
    support = np.arange(n_test + 1)

    def ks(h):
        ecdf = np.searchsorted(np.sort(h), support, side="right") / len(h)
        return float(np.max(np.abs(ecdf - cdf)))

    d_obs = ks(h_obs)
    rng = np.random.default_rng(seed)
    a, b = coverage_law_params(n_cal, alpha)
    d_null = np.array([ks(stats.betabinom.rvs(n_test, a, b, size=len(h_obs), random_state=rng)) for _ in range(n_mc)])
    return d_obs, float((1 + np.sum(d_null >= d_obs)) / (n_mc + 1))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


# ----------------------------------------------------------------------------------------------
# Joint coverage across properties (Section 4.4, Eq. 8)
# ----------------------------------------------------------------------------------------------
def joint_box(cal: dict, te: dict, props, alpha: float, scale_cal=None, scale_te=None):
    """Max-score conformal box over the properties ``props``.

    The score is max_j |y_j - mu_j| / d_j. With ``scale_cal`` and ``scale_te`` both ``None`` the
    caller must pass fixed scales through ``scale_cal`` as a 1-D array (e.g. training residual sd).
    For adaptive scales pass (n, k) arrays for calibration and test.

    Returns (lo, hi, qhat) with lo/hi of shape (n_test, k).
    """
    props = list(props)
    yc, mc = cal["y"][:, props], cal["mu"][:, props]
    mt = te["mu"][:, props]
    dc = np.asarray(scale_cal, float)
    dt = dc if scale_te is None else np.asarray(scale_te, float)
    q = cq(np.max(np.abs(yc - mc) / dc, axis=1), alpha)
    return mt - q * dt, mt + q * dt, q


def bonferroni_box(cal: dict, te: dict, props, alpha: float):
    """Split conformal at level alpha / k for each of k properties."""
    props = list(props)
    k = len(props)
    hw = np.array([cq(np.abs(cal["y"][:, j] - cal["mu"][:, j]), alpha / k) for j in props])
    mt = te["mu"][:, props]
    return mt - hw, mt + hw


def joint_coverage(y, lo, hi) -> float:
    """Share of rows where every column lies inside its interval."""
    return float(np.mean(np.all((y >= lo) & (y <= hi), axis=1)))

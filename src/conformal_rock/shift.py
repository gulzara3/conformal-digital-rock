"""Tools for distribution shift (Sections 4.7 and 4.8).

* ``novelty_scores`` and ``conformal_pvalue`` form the label-free out-of-distribution guard (Eq. 10).
* ``weighted_threshold`` and ``density_ratio`` give weighted conformal prediction (Eq. 11).
"""
from __future__ import annotations

import numpy as np


def knn_score(fit_x, x, k: int = 10) -> np.ndarray:
    """Mean Euclidean distance from each row of ``x`` to its ``k`` nearest rows of ``fit_x``."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k).fit(fit_x)
    return nn.kneighbors(x)[0].mean(1)


def standardise_features(train_feat, *others):
    """Standardise network features with the training mean and sd; returns the training array first."""
    m, s = train_feat.mean(0), train_feat.std(0) + 1e-6
    return [(train_feat - m) / s] + [(o - m) / s for o in others]


def novelty_scores(train_feat, cal_feat, new_feat, k: int = 10, max_ref: int = 20000, seed: int = 0):
    """Novelty of calibration and new volumes relative to the training set (penultimate-layer features)."""
    z_tr, z_cal, z_new = standardise_features(train_feat, cal_feat, new_feat)
    if len(z_tr) > max_ref:
        z_tr = z_tr[np.random.default_rng(seed).choice(len(z_tr), max_ref, replace=False)]
    return knn_score(z_tr, z_cal, k), knn_score(z_tr, z_new, k)


def conformal_pvalue(score_cal, score_new) -> np.ndarray:
    """Conformal p-value p(x) = (1 + #{i: s_i >= s(x)}) / (n + 1)  (Eq. 10).

    On data exchangeable with the calibration set, P(p(X) <= delta) <= delta, so flagging
    p <= delta bounds the false-alarm rate by delta.
    """
    s_cal = np.sort(np.asarray(score_cal, float))
    s_new = np.asarray(score_new, float)
    n_ge = len(s_cal) - np.searchsorted(s_cal, s_new, side="left")
    return (1 + n_ge) / (len(s_cal) + 1)


def weighted_threshold(s_cal, w_cal, w_new, alpha: float) -> np.ndarray:
    """Weighted conformal threshold for each new point (Tibshirani et al., 2019); the test mass sits at +inf."""
    s_cal = np.asarray(s_cal, float)
    o = np.argsort(s_cal)
    ss, cw = s_cal[o], np.cumsum(np.asarray(w_cal, float)[o])
    total = cw[-1]
    idx = np.searchsorted(cw, (1 - alpha) * (total + np.asarray(w_new, float)) - 1e-12, side="left")
    return np.where(idx < len(ss), ss[np.minimum(idx, len(ss) - 1)], np.inf)


def density_ratio(x_cal, x_unlabelled, x_eval, C: float = 1.0):
    """Density ratio p_target / p_cal by logistic regression on label-free descriptors.

    Returns the ratio at the calibration points and at ``x_eval``.
    """
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, max_iter=2000).fit(
        np.r_[x_cal, x_unlabelled], np.r_[np.zeros(len(x_cal)), np.ones(len(x_unlabelled))])
    r = len(x_cal) / len(x_unlabelled)

    def g(x):
        p = np.clip(clf.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
        return p / (1 - p) * r

    return g(x_cal), g(x_eval)


def effective_sample_size(w) -> float:
    """Kish effective sample size of a weight vector."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum())

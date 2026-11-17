"""Point-accuracy metrics (Section 4.10)."""
from __future__ import annotations

import numpy as np


def regression_metrics(y, pred) -> dict:
    """MAE, RMSE and coefficient of determination."""
    y = np.asarray(y, float)
    e = np.asarray(pred, float) - y
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                R2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()))


def coverage_error_pp(coverages, target: float = 0.90) -> tuple[float, float]:
    """Mean and maximum absolute coverage error in percentage points.

    The absolute error is taken per property and seed *before* averaging, so over- and under-coverage
    cannot cancel (Reviewer 2, comment 7).
    """
    e = 100 * np.abs(np.asarray(coverages, float) - target)
    return float(e.mean()), float(e.max())


def permeability_factor(width_decades) -> np.ndarray:
    """Multiplicative factor around the prediction implied by a log10-permeability interval width."""
    return 10 ** (np.asarray(width_decades, float) / 2)

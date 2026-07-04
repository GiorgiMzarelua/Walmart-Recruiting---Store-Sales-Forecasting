"""WMAE and the holiday weighting. The single most-reused piece in the project.

Key fact used throughout: WMAE == a plain *weighted* mean absolute error with
weight 5 on holiday rows and 1 otherwise. So if you pass these same weights as
sample_weight to LightGBM/XGBoost with an L1 objective, the library's reported
weighted 'l1'/'mae' IS the competition metric, exactly.
"""
import numpy as np

HOLIDAY_WEIGHT = 5.0
NORMAL_WEIGHT = 1.0


def _np(x):
    return x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)


def holiday_weights(is_holiday) -> np.ndarray:
    """Map a boolean/0-1 holiday indicator to sample weights (5.0 / 1.0)."""
    h = _np(is_holiday).astype(bool)
    return np.where(h, HOLIDAY_WEIGHT, NORMAL_WEIGHT).astype(float)


def wmae(y_true, y_pred, is_holiday) -> float:
    """Competition metric: sum(w * |y - yhat|) / sum(w), w = 5 on holidays else 1."""
    yt, yp = _np(y_true).astype(float), _np(y_pred).astype(float)
    w = holiday_weights(is_holiday)
    return float(np.sum(w * np.abs(yt - yp)) / np.sum(w))


def wmae_weighted(y_true, y_pred, sample_weight) -> float:
    """Same as wmae but when you already hold the weight vector."""
    yt, yp, w = _np(y_true).astype(float), _np(y_pred).astype(float), _np(sample_weight).astype(float)
    return float(np.sum(w * np.abs(yt - yp)) / np.sum(w))
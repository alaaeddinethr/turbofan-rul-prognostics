"""Evaluation metrics for RUL prediction."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric scoring function of the PHM08 challenge (lower is better).

    Late predictions (overestimating RUL, d > 0) are penalised more than
    early ones, because predicting failure too late is the dangerous error
    in maintenance planning.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of true values falling inside the predicted interval."""
    y_true = np.asarray(y_true, float)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"rmse": round(rmse(y_true, y_pred), 2), "nasa_score": round(nasa_score(y_true, y_pred), 1)}

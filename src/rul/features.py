"""Turn per-cycle sensor time series into model inputs.

Two representations are produced from the same sliding windows:
- tabular features (level, mean, std, trend per sensor) for linear / tree models
- raw windows (cycles x sensors) for the 1D-CNN
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 30


def _pad_front(values: np.ndarray, window: int) -> np.ndarray:
    """Repeat the first row so short engines still yield a full window."""
    if len(values) >= window:
        return values
    pad = np.repeat(values[:1], window - len(values), axis=0)
    return np.vstack([pad, values])


def windows(
    df: pd.DataFrame, features: list[str], window: int = DEFAULT_WINDOW, last_only: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows over each engine.

    Returns X of shape (n, window, n_features), y (RUL at the window's last
    cycle) and the unit id of each window. With ``last_only`` a single
    window per engine is returned (its most recent cycles), which is how
    the official test set is scored.
    """
    xs, ys, units = [], [], []
    for unit, g in df.groupby("unit", sort=True):
        vals = _pad_front(g[features].to_numpy(dtype=np.float32), window)
        rul = _pad_front(g[["rul"]].to_numpy(dtype=np.float32), window)[:, 0]
        ends = [len(vals)] if last_only else range(window, len(vals) + 1)
        for end in ends:
            xs.append(vals[end - window : end])
            ys.append(rul[end - 1])
            units.append(unit)
    return np.stack(xs), np.asarray(ys, dtype=np.float32), np.asarray(units)


def tabular(x: np.ndarray) -> np.ndarray:
    """Summarise each window: last value, mean, std and linear slope per sensor."""
    t = np.arange(x.shape[1], dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-8)
    slope = np.einsum("t,ntf->nf", t, x - x.mean(axis=1, keepdims=True)) / len(t)
    return np.concatenate([x[:, -1, :], x.mean(axis=1), x.std(axis=1), slope], axis=1)


def tabular_names(features: list[str]) -> list[str]:
    return [f"{f}_{stat}" for stat in ("last", "mean", "std", "slope") for f in features]

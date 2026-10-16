"""RUL models: a linear baseline, gradient-boosted trees and a 1D-CNN.

Every model exposes ``fit(X, y)`` and ``predict(X) -> (mean, lo, hi)``
where ``lo``/``hi`` bound an 80 % prediction interval. Knowing how sure
the model is matters as much as the point estimate when a maintenance
planner has to decide whether to pull an engine.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .features import tabular

Q_LO, Q_HI = 0.1, 0.9


class LinearRUL:
    name = "linear"

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.resid_q: tuple[float, float] = (0.0, 0.0)

    def fit(self, x: np.ndarray, y: np.ndarray, x_val=None, y_val=None) -> LinearRUL:
        self.model.fit(tabular(x), y)
        # Interval from empirical residual quantiles on held-out engines
        xr, yr = (x_val, y_val) if x_val is not None else (x, y)
        resid = yr - self.model.predict(tabular(xr))
        self.resid_q = (float(np.quantile(resid, Q_LO)), float(np.quantile(resid, Q_HI)))
        return self

    def predict(self, x: np.ndarray):
        m = np.clip(self.model.predict(tabular(x)), 0, None)
        return m, np.clip(m + self.resid_q[0], 0, None), m + self.resid_q[1]


class XGBoostRUL:
    name = "xgboost"

    def __init__(self, n_estimators: int = 400, max_depth: int = 4, learning_rate: float = 0.05, seed: int = 0):
        import xgboost as xgb

        params = dict(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=2,
        )
        self.point = xgb.XGBRegressor(objective="reg:squarederror", **params)
        self.lo = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=Q_LO, **params)
        self.hi = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=Q_HI, **params)

    def fit(self, x: np.ndarray, y: np.ndarray, x_val=None, y_val=None) -> XGBoostRUL:
        f = tabular(x)
        for m in (self.point, self.lo, self.hi):
            m.fit(f, y)
        return self

    def predict(self, x: np.ndarray):
        f = tabular(x)
        m = np.clip(self.point.predict(f), 0, None)
        lo = np.minimum(np.clip(self.lo.predict(f), 0, None), m)
        hi = np.maximum(self.hi.predict(f), m)
        return m, lo, hi


class CNNRUL:
    """1D convolutional network over raw sensor windows.

    Uncertainty is estimated with Monte-Carlo dropout: dropout stays active
    at inference and the spread of repeated predictions gives the interval.
    """

    name = "cnn"

    def __init__(self, epochs: int = 100, lr: float = 1e-3, batch_size: int = 128,
                 dropout: float = 0.2, patience: int = 15, mc_samples: int = 50, seed: int = 0,
                 y_scale: float = 125.0):
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.dropout, self.patience, self.mc_samples, self.seed = dropout, patience, mc_samples, seed
        self.y_scale = y_scale  # targets are trained in [0, 1] for stable optimisation
        self.net = None
        self.history: list[tuple[float, float]] = []

    def _build(self, n_features: int):
        import torch.nn as nn

        d = self.dropout
        return nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(d), nn.Linear(32, 64), nn.ReLU(),
            nn.Dropout(d), nn.Linear(64, 1),
        )

    def fit(self, x: np.ndarray, y: np.ndarray, x_val=None, y_val=None) -> CNNRUL:
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net = self._build(x.shape[2])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        xt = torch.from_numpy(x).permute(0, 2, 1)
        yt = torch.from_numpy(y / self.y_scale).unsqueeze(1)
        xv = torch.from_numpy(x_val).permute(0, 2, 1) if x_val is not None else None
        yv = torch.from_numpy(y_val / self.y_scale).unsqueeze(1) if y_val is not None else None

        best, best_state, bad = float("inf"), None, 0
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(xt))
            total = 0.0
            for i in range(0, len(xt), self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            train_rmse = (total / len(xt)) ** 0.5 * self.y_scale
            if xv is None:
                self.history.append((train_rmse, float("nan")))
                continue
            self.net.eval()
            with torch.no_grad():
                val_rmse = loss_fn(self.net(xv), yv).item() ** 0.5 * self.y_scale
            self.history.append((train_rmse, val_rmse))
            if val_rmse < best - 1e-3:
                best, bad = val_rmse, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, x: np.ndarray):
        import torch

        xt = torch.from_numpy(x).permute(0, 2, 1)
        self.net.train()  # keep dropout on for MC sampling
        torch.manual_seed(self.seed)
        with torch.no_grad():
            draws = np.stack([self.net(xt).numpy()[:, 0] for _ in range(self.mc_samples)]) * self.y_scale
        draws = np.clip(draws, 0, None)
        m = draws.mean(axis=0)
        return m, np.quantile(draws, Q_LO, axis=0), np.quantile(draws, Q_HI, axis=0)


MODELS = {cls.name: cls for cls in (LinearRUL, XGBoostRUL, CNNRUL)}

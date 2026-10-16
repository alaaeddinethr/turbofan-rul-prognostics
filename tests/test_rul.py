"""Unit tests on synthetic engines: no dataset download needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rul import data as D
from src.rul.features import tabular, tabular_names, windows
from src.rul.metrics import interval_coverage, nasa_score, rmse
from src.rul.models import CNNRUL, LinearRUL, XGBoostRUL


def synthetic(n_units: int = 12, seed: int = 0) -> pd.DataFrame:
    """Engines whose sensors drift linearly towards failure, plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        life = int(rng.integers(80, 160))
        for c in range(1, life + 1):
            wear = c / life
            sensors = {s: 0.0 for s in D.SENSORS}
            sensors.update({"s2": wear + rng.normal(0, 0.02), "s3": 2 * wear + rng.normal(0, 0.02),
                            "s4": -wear + rng.normal(0, 0.02)})
            rows.append({"unit": u, "cycle": c, "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0, **sensors})
    return pd.DataFrame(rows)[D.COLUMNS]


# --- labelling ---------------------------------------------------------------

def test_train_rul_counts_down_to_zero_and_is_capped():
    df = D.add_train_rul(synthetic(3), cap=50)
    for _, g in df.groupby("unit"):
        assert g["rul"].iloc[-1] == 0
        assert g["rul"].max() == 50
        assert (np.diff(g["rul"].to_numpy()) <= 0).all()


def test_test_rul_uses_official_offset():
    df = synthetic(2)
    out = D.add_test_rul(df, np.array([10, 30]), cap=None)
    last = out.groupby("unit").tail(1)["rul"].tolist()
    assert last == [10, 30]


def test_test_rul_rejects_mismatched_units():
    with pytest.raises(ValueError):
        D.add_test_rul(synthetic(2), np.array([1, 2, 3]))


def test_constant_sensors_are_dropped():
    feats = D.informative_sensors(synthetic(4))
    assert feats == ["s2", "s3", "s4"]


def test_split_is_by_engine():
    tr, va = D.split_units(synthetic(10), val_fraction=0.3)
    assert set(tr["unit"]).isdisjoint(va["unit"])
    assert va["unit"].nunique() == 3


def test_scaler_uses_train_range_only():
    tr = pd.DataFrame({"s2": [0.0, 10.0]})
    sc = D.Scaler.fit(tr, ["s2"])
    assert sc.transform(pd.DataFrame({"s2": [20.0]}))["s2"].iloc[0] == 2.0


# --- features ----------------------------------------------------------------

def test_window_shapes_and_targets():
    df = D.add_train_rul(synthetic(3))
    x, y, u = windows(df, ["s2", "s3", "s4"], window=20)
    n_expected = sum(len(g) - 19 for _, g in df.groupby("unit"))
    assert x.shape == (n_expected, 20, 3)
    assert len(y) == len(u) == n_expected
    assert y.min() == 0


def test_short_engine_is_padded():
    df = D.add_train_rul(synthetic(1)).head(5)
    x, _, _ = windows(df, ["s2"], window=30, last_only=True)
    assert x.shape == (1, 30, 1)


def test_tabular_slope_sign():
    x = np.linspace(0, 1, 30, dtype=np.float32).reshape(1, 30, 1)
    f = tabular(x)
    assert f.shape == (1, 4)
    assert f[0, 3] > 0  # rising sensor -> positive slope
    assert len(tabular_names(["a", "b"])) == 8


# --- metrics -----------------------------------------------------------------

def test_rmse_known_value():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_nasa_score_penalises_late_predictions_more():
    early = nasa_score([50], [40])  # predicted failure 10 cycles too early
    late = nasa_score([50], [60])   # 10 cycles too late
    assert late > early > 0
    assert nasa_score([10, 20], [10, 20]) == 0


def test_interval_coverage():
    assert interval_coverage([1, 5, 9], np.array([0, 0, 0]), np.array([2, 4, 10])) == pytest.approx(2 / 3)


# --- models ------------------------------------------------------------------

@pytest.fixture(scope="module")
def xy():
    feats = ["s2", "s3", "s4"]
    tr, va = D.split_units(D.add_train_rul(synthetic(16)), seed=1)
    x, y, _ = windows(tr, feats, window=20)
    xv, yv, _ = windows(va, feats, window=20)
    return x, y, xv, yv


@pytest.mark.parametrize("model", [LinearRUL(), XGBoostRUL(n_estimators=60)])
def test_tabular_models_learn_and_bound_prediction(model, xy):
    x, y, xv, yv = xy
    m, lo, hi = model.fit(x, y, xv, yv).predict(xv)
    assert rmse(yv, m) < 0.5 * np.std(yv)  # clearly better than predicting the mean
    assert (lo <= m + 1e-6).all() and (m <= hi + 1e-6).all() and (m >= 0).all()


def test_cnn_trains_and_returns_intervals(xy):
    x, y, xv, yv = xy
    model = CNNRUL(epochs=8, patience=8, mc_samples=10).fit(x, y, xv, yv)
    m, lo, hi = model.predict(xv)
    assert m.shape == yv.shape
    assert (hi >= lo).all()
    assert model.history[-1][0] < model.history[0][0]  # training loss decreases

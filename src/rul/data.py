"""Loading and labelling of the NASA C-MAPSS turbofan degradation dataset.

Each file is a space-separated table: one row per engine cycle with
unit id, cycle number, 3 operating settings and 21 sensor readings.
Training engines run until failure; test engines stop at an unknown
point and the true remaining useful life (RUL) at their last cycle is
given in a separate RUL_FDxxx.txt file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SETTINGS = ["setting_1", "setting_2", "setting_3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]

# Above this value the engine is considered healthy: degradation is not
# yet observable in the sensors, so the target is capped (piecewise-linear
# RUL, the standard convention in the C-MAPSS literature).
DEFAULT_RUL_CAP = 125

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "CMAPSSData"


def read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS
    return df


def add_train_rul(df: pd.DataFrame, cap: int | None = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """Label run-to-failure data: RUL = last cycle of the unit - current cycle."""
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = last - out["cycle"]
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out


def add_test_rul(df: pd.DataFrame, final_rul: np.ndarray, cap: int | None = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """Label truncated test data using the true RUL at each unit's last cycle."""
    out = df.copy()
    units = np.sort(out["unit"].unique())
    if len(units) != len(final_rul):
        raise ValueError(f"{len(units)} test units but {len(final_rul)} RUL values")
    offset = dict(zip(units, final_rul))
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = last - out["cycle"] + out["unit"].map(offset)
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out


def informative_sensors(train: pd.DataFrame, tol: float = 1e-4) -> list[str]:
    """Sensors whose standard deviation is non-negligible on the training set."""
    std = train[SENSORS].std()
    return [s for s in SENSORS if std[s] > tol]


@dataclass
class Scaler:
    """Min-max scaler fitted on training data only (no test leakage)."""

    lo: pd.Series
    hi: pd.Series

    @classmethod
    def fit(cls, df: pd.DataFrame, cols: list[str]) -> Scaler:
        return cls(df[cols].min(), df[cols].max())

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cols = list(self.lo.index)
        span = (self.hi - self.lo).replace(0, 1.0)
        out[cols] = (out[cols] - self.lo) / span
        return out


@dataclass
class Dataset:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    features: list[str]


def load(name: str = "FD001", data_dir: Path = DATA_DIR, cap: int | None = DEFAULT_RUL_CAP) -> Dataset:
    """Load one C-MAPSS sub-dataset, labelled and scaled."""
    data_dir = Path(data_dir)
    files = {k: data_dir / f"{k}_{name}.txt" for k in ("train", "test", "RUL")}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "C-MAPSS files not found: " + ", ".join(missing) + "\nRun: python scripts/download_data.py"
        )
    train = add_train_rul(read_table(files["train"]), cap)
    final_rul = pd.read_csv(files["RUL"], header=None).iloc[:, 0].to_numpy()
    test = add_test_rul(read_table(files["test"]), final_rul, cap)

    features = informative_sensors(train)
    scaler = Scaler.fit(train, features)
    return Dataset(name, scaler.transform(train), scaler.transform(test), features)


def split_units(df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by engine (never by row) so validation engines are unseen."""
    rng = np.random.default_rng(seed)
    units = df["unit"].unique()
    n_val = max(1, int(round(len(units) * val_fraction)))
    val_units = set(rng.choice(units, size=n_val, replace=False))
    is_val = df["unit"].isin(val_units)
    return df[~is_val], df[is_val]

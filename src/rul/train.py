"""Train and evaluate RUL models on a C-MAPSS sub-dataset.

    python -m src.rul.train --dataset FD001 --models linear xgboost cnn

Writes results/metrics_<dataset>.json and results/predictions_<dataset>.csv.
Scoring follows the standard protocol: one prediction per test engine,
made at its last observed cycle, compared with the official true RUL.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .features import DEFAULT_WINDOW, windows
from .metrics import interval_coverage, summary
from .models import MODELS

RESULTS = Path(__file__).resolve().parents[2] / "results"


def run(dataset: str = "FD001", models: list[str] | None = None, window: int = DEFAULT_WINDOW,
        seed: int = 0, data_dir: Path = D.DATA_DIR, out_dir: Path = RESULTS) -> dict:
    models = models or list(MODELS)
    ds = D.load(dataset, data_dir)
    tr, va = D.split_units(ds.train, val_fraction=0.2, seed=seed)
    x_tr, y_tr, _ = windows(tr, ds.features, window)
    x_va, y_va, _ = windows(va, ds.features, window)
    x_te, y_te, u_te = windows(ds.test, ds.features, window, last_only=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"dataset": dataset, "window": window, "rul_cap": D.DEFAULT_RUL_CAP,
              "features": ds.features, "n_test_engines": len(u_te), "models": {}}
    preds = pd.DataFrame({"unit": u_te, "rul_true": y_te})

    for name in models:
        t0 = time.time()
        model = MODELS[name](seed=seed) if name != "linear" else MODELS[name]()
        # Same training engines for every model; held-out engines are used for
        # early stopping (CNN) and interval calibration (linear).
        model.fit(x_tr, y_tr, x_va, y_va)
        m, lo, hi = model.predict(x_te)
        res = summary(y_te, m)
        res["interval_80_coverage"] = round(interval_coverage(y_te, lo, hi), 2)
        res["mean_interval_width"] = round(float(np.mean(hi - lo)), 1)
        res["train_seconds"] = round(time.time() - t0, 1)
        report["models"][name] = res
        preds[f"{name}_pred"], preds[f"{name}_lo"], preds[f"{name}_hi"] = m, lo, hi
        print(f"{dataset} {name:8s} RMSE={res['rmse']:6.2f}  score={res['nasa_score']:8.1f}  "
              f"cov80={res['interval_80_coverage']:.2f}  ({res['train_seconds']}s)")

    (out_dir / f"metrics_{dataset}.json").write_text(json.dumps(report, indent=2))
    preds.round(2).to_csv(out_dir / f"predictions_{dataset}.csv", index=False)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run(a.dataset, a.models, a.window, a.seed)


if __name__ == "__main__":
    main()

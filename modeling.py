import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from evaluation import paired_difference, per_race_metrics, summarize
from form_features import FORM_FEATURES, GRID_FEATURES, add_form_features
from jolpica_data import fetch_race_training_data

PROJECT_DIR = Path(__file__).resolve().parent
TARGET = "FinishPosition"
TRAIN_START, TEST_SEASON, WARMUP_ROUNDS = 2018, 2025, 5
MODEL_PARAMS = dict(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42)

CONFIGS = {
    "grid": dict(features=GRID_FEATURES, model="position_model.joblib",
                 metrics="model_metrics.json", label="Grid + form (after qualifying)"),
    "form": dict(features=FORM_FEATURES, model="form_model.joblib",
                 metrics="form_metrics.json", label="Form only (before qualifying)"),
}
BASELINES = {"grid_order": "QualifyingPosition", "driver_form_order": "DriverFormFinish"}


def load_frame(refresh=False):
    end = max(TEST_SEASON, dt.date.today().year)
    history = fetch_race_training_data(TRAIN_START, end, refresh)

    races = history.groupby("Season")["Round"].nunique()
    print("\nRaces per season:\n" + races.to_string())
    for season, n in races.items():
        if season < end and n < 15:
            print(f"WARNING: {season} has only {n} races - data looks incomplete.")

    frame = add_form_features(history)
    frame = frame[~((frame["Season"] == TRAIN_START) & (frame["Round"] <= WARMUP_ROUNDS))]
    return frame


def train_and_evaluate(kind, refresh=False):
    cfg = CONFIGS[kind]
    features = cfg["features"]
    print(f"\n=== {cfg['label']} ===")

    frame = load_frame(refresh).dropna(subset=features + [TARGET])
    train = frame[frame["Season"] < TEST_SEASON]
    test = frame[frame["Season"] == TEST_SEASON].copy()
    if train.empty or test.empty:
        raise ValueError("Train or test split is empty - check the data.")
    print(f"Train rows: {len(train)} | Test rows: {len(test)}")

    model = GradientBoostingRegressor(**MODEL_PARAMS).fit(train[features], train[TARGET])
    test["ModelScore"] = model.predict(test[features])

    per_race = {"model": per_race_metrics(test, "ModelScore")}
    for name, col in BASELINES.items():
        per_race[name] = per_race_metrics(test, col)
    summary = {name: summarize(df) for name, df in per_race.items()}
    paired = {name: {m: paired_difference(per_race["model"], per_race[name], m)
                     for m in ("spearman", "winner_hit", "top3_hit")}
              for name in BASELINES}

    model_mae = float(np.abs(test["ModelScore"] - test[TARGET]).mean())
    baseline_mae = float(np.abs(test["QualifyingPosition"] - test[TARGET]).mean())

    table = pd.DataFrame(summary).T[["races", "winner_accuracy", "top3_hit_rate",
                                     "spearman", "rank_mae"]]
    print(f"\nTest season {TEST_SEASON} (per-race metrics, averaged over races):")
    print(table.round(3).to_string())
    print("\nPaired difference model - grid_order (mean [95% CI]):")
    for m, d in paired["grid_order"].items():
        print(f"  {m:11s} {d['mean']:+.3f} [{d['ci95_low']:+.3f}, {d['ci95_high']:+.3f}] over {d['n']} races")

    final = GradientBoostingRegressor(**MODEL_PARAMS).fit(frame[features], frame[TARGET])
    joblib.dump(final, PROJECT_DIR / cfg["model"])

    metrics = {
        "model": "Gradient Boosting Regressor", "variant": cfg["label"],
        "features": features, "target": TARGET,
        "training_period": f"{TRAIN_START}-{TEST_SEASON - 1}",
        "testing_period": str(TEST_SEASON),
        "final_model_training_period":
            f"{int(frame['Season'].min())}-{int(frame['Season'].max())}",
        "training_records": int(len(train)), "testing_records": int(len(test)),
        "model_mae": model_mae, "baseline_mae": baseline_mae,
        "relative_improvement_percent": (baseline_mae - model_mae) / baseline_mae * 100,
        "per_race_summary": summary, "paired_vs_baselines": paired,
        "model_params": MODEL_PARAMS,
    }
    with open(PROJECT_DIR / cfg["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {cfg['model']} and {cfg['metrics']}")
    return metrics
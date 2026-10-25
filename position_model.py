"""
F1 Race Predictor 2026

1. Fetch historical qualifying and race results.
2. Train Gradient Boosting using 2018-2024.
3. Evaluate on the separate 2025 season.
4. Compare against a qualifying-position baseline.
5. Train the final model on all available historical data.
6. Save the model and metrics for Streamlit.
"""

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from jolpica_data import fetch_race_training_data


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

FEATURES = ["QualifyingPosition"]
TARGET = "FinishPosition"

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "position_model.joblib"
METRICS_PATH = PROJECT_DIR / "model_metrics.json"


def create_model():
    """Create the Gradient Boosting regression model."""

    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    )


def clean_data(data):
    """Convert required columns to numeric values and clean records."""

    required_columns = FEATURES + [TARGET, "Season"]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Historical data is missing columns: {missing_columns}"
        )

    data = data.copy()

    # Convert values before dropping missing records.
    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=required_columns
    ).copy()

    data["Season"] = data["Season"].astype(int)

    return data


def fetch_with_retry(start_year, end_year, attempts=3):
    """Fetch historical data with a small retry allowance."""

    for attempt in range(1, attempts + 1):
        try:
            print(
                f"Fetching seasons {start_year}-{end_year} "
                f"(attempt {attempt}/{attempts})..."
            )

            data = fetch_race_training_data(
                start_year=start_year,
                end_year=end_year,
            )

            data = clean_data(data)

            if not data.empty:
                return data

            print("No usable records returned.")

        except (ValueError, KeyError, TypeError) as error:
            print(f"Data retrieval issue: {error}")

        if attempt < attempts:
            print("Retrying in 5 seconds...")
            time.sleep(5)

    raise ValueError(
        f"Could not retrieve usable data for "
        f"{start_year}-{end_year} after {attempts} attempts."
    )


def train_and_evaluate():
    """Train, evaluate, and save the F1 finishing-position model."""

    print("=" * 55)
    print("F1 RACE PREDICTOR 2026 — MODEL TRAINING")
    print("=" * 55)

    # -----------------------------------------------------
    # 1. Fetch training and testing data separately
    # -----------------------------------------------------

    print("\n[1/5] Fetching historical data...")

    training_history = fetch_with_retry(
        start_year=2018,
        end_year=2024,
    )

    testing_history = fetch_with_retry(
        start_year=2025,
        end_year=2025,
    )

    # Keep only the intended seasons in each dataset.
    training_history = training_history[
        training_history["Season"].between(2018, 2024)
    ].copy()

    testing_history = testing_history[
        testing_history["Season"] == 2025
    ].copy()

    if training_history.empty:
        raise ValueError(
            "No usable training records remain for 2018-2024."
        )

    if testing_history.empty:
        raise ValueError(
            "The 2025 API response contained no usable 2025 records."
        )

    # Combine data only after checking each period separately.
    all_data = pd.concat(
        [training_history, testing_history],
        ignore_index=True,
    )

    print("\nUsable records by season:")
    print(all_data.groupby("Season").size().to_string())

    print(f"\nTraining records: {len(training_history)}")
    print(f"Testing records:  {len(testing_history)}")

    # -----------------------------------------------------
    # 2. Prepare the chronological evaluation
    # -----------------------------------------------------

    print("\n[2/5] Preparing model evaluation...")

    X_train = training_history[FEATURES]
    y_train = training_history[TARGET]

    X_test = testing_history[FEATURES]
    y_test = testing_history[TARGET]

    # -----------------------------------------------------
    # 3. Train and evaluate the validation model
    # -----------------------------------------------------

    print("[3/5] Training Gradient Boosting Regressor...")

    validation_model = create_model()

    validation_model.fit(
        X_train,
        y_train,
    )

    predictions = validation_model.predict(X_test)

    model_mae = mean_absolute_error(
        y_test,
        predictions,
    )

    # Baseline: predict race finishing position using qualifying position.
    baseline_predictions = X_test["QualifyingPosition"]

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions,
    )

    improvement = (
        (baseline_mae - model_mae) / baseline_mae * 100
        if baseline_mae > 0
        else 0.0
    )

    print("\n" + "=" * 55)
    print("CHRONOLOGICAL EVALUATION RESULTS")
    print("=" * 55)

    print("Training period:             2018-2024")
    print("Testing period:              2025")
    print(f"Training records:             {len(training_history)}")
    print(f"Testing records:              {len(testing_history)}")
    print(f"Gradient Boosting MAE:        {model_mae:.3f}")
    print(f"Qualifying baseline MAE:      {baseline_mae:.3f}")
    print(f"Relative MAE improvement:     {improvement:.2f}%")

    if model_mae < baseline_mae:
        print("\nThe model outperformed the baseline.")
    elif model_mae > baseline_mae:
        print("\nThe baseline outperformed the model.")
    else:
        print("\nThe model and baseline have equal MAE.")

    # -----------------------------------------------------
    # 4. Train the final model on all available seasons
    # -----------------------------------------------------

    print("\n[4/5] Training final model on 2018-2025 data...")

    final_model = create_model()

    final_model.fit(
        all_data[FEATURES],
        all_data[TARGET],
    )

    # -----------------------------------------------------
    # 5. Save model and metrics
    # -----------------------------------------------------

    print("[5/5] Saving model and evaluation metrics...")

    joblib.dump(
        final_model,
        MODEL_PATH,
    )

    metrics = {
        "model": "Gradient Boosting Regressor",
        "features": FEATURES,
        "target": TARGET,
        "training_period": "2018-2024",
        "testing_period": "2025",
        "final_model_training_period": "2018-2025",
        "training_records": int(len(training_history)),
        "testing_records": int(len(testing_history)),
        "model_mae": float(model_mae),
        "baseline_mae": float(baseline_mae),
        "relative_improvement_percent": float(improvement),
        "random_state": 42,
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 2,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 55)

    print(f"Model saved:   {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")

    return final_model, model_mae, baseline_mae, all_data


if __name__ == "__main__":
    train_and_evaluate()
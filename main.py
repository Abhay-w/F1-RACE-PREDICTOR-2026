"""
2026 F1 Predictor - entry point.
Pipeline:
  1. Fetch previous-season race lap times for the same Grand Prix (training target)
  2. Fetch current-season qualifying times for that Grand Prix (prediction input)
  3. Engineer features (qualifying time + team average pace)
  4. Train a Gradient Boosting model
  5. Predict and rank 2026 race pace
  6. Report Mean Absolute Error
Usage:
      python main.py
Edit RACE_NAME / TRAIN_ROUNDS / PREDICT_YEAR / PREDICT_ROUND to target a different race.
"""

import pandas as pd
from drivers_2026 import DRIVER_CODES
from fetch_data import get_season_avg_pace, get_qualifying_times
from features import build_feature_set
from train_model import train, predict, FEATURE_COLUMNS

# ---- Configuration: change these for a different race ----
RACE_NAME = "Italian GP (Monza)"
TRAIN_YEAR = 2026
TRAIN_ROUNDS = list(range(1, 16))  # 1-15, up to Monza (completed rounds)
PREDICT_YEAR, PREDICT_ROUND = 2026, 11 # currently set to round 11 for testing
# ------------------------------------------------------------
# NOTE: Madrid is a brand-new circuit in 2026, so there's no "same
# race last year" to train on. Instead we train on this season's
# average race pace across every round run so far, which also
# captures each driver's current form better than a single old race.


def main():
    print(f"\nFetching {TRAIN_YEAR} season race data (rounds {TRAIN_ROUNDS[0]}-{TRAIN_ROUNDS[-1]}) for training...")
    race_laps = get_season_avg_pace(TRAIN_YEAR, TRAIN_ROUNDS)

    print(f"\nFetching {PREDICT_YEAR} {RACE_NAME} qualifying data...")
    qualifying = get_qualifying_times(PREDICT_YEAR, PREDICT_ROUND)

    # Map FastF1 3-letter codes back to full driver names for readability
    code_to_name = {v: k for k, v in DRIVER_CODES.items()}
    qualifying["Driver"] = qualifying["DriverCode"].map(code_to_name)
    qualifying.dropna(subset=["Driver"], inplace=True)

    # Feature engineering (adds Team + TeamAvgTime columns)
    qualifying_features = build_feature_set(qualifying)

    # Merge training data: match by FastF1 driver code
    merged = qualifying_features.merge(
        race_laps, left_on="DriverCode", right_on="Driver", suffixes=("", "_race")
    )
    merged = merged.dropna()

    print("\nTraining model...")
    model, mae = train(merged)

    print("Predicting 2026 race pace...")
    qualifying_features = qualifying_features.dropna(subset=FEATURE_COLUMNS)
    qualifying_features["PredictedRaceTime (s)"] = predict(model, qualifying_features)
    results = qualifying_features.sort_values("PredictedRaceTime (s)")

    print(f"\nPredicted {PREDICT_YEAR} {RACE_NAME} Result\n")
    print(results[["Driver", "Team", "PredictedRaceTime (s)"]].to_string(index=False))
    
    winner = results.iloc[0]
    print(f"\n Predicted Winner: {winner['Driver']} ({winner['Team']})")
    print(f"\nModel Error (MAE): {mae:.2f} seconds")


if __name__ == "__main__":
    main()
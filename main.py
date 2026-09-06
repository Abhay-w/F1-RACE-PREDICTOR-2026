"""
2026 F1 Predictor - entry point.

Pipeline:
  1. Auto-select training rounds: every completed round BEFORE the target round
  2. Fetch current-season qualifying times for the target round (prediction input)
  3. Engineer features (qualifying time + team average pace)
  4. Train a Gradient Boosting model on prior-round average race pace
  5. Predict and rank future race pace for the target round
  6. Report Mean Absolute Error
  7. If the target round has ALREADY happened, also compare predictions
     against the real race result (accuracy check / demo for grading)

Usage:
      python main.py

ONLY THING YOU EVER NEED TO CHANGE:
      TARGET_ROUND   -> the round number you want to predict

Everything else (training window, leakage prevention, comparison) is automatic.
"""

import pandas as pd
from drivers_2026 import DRIVER_CODES
from fetch_data import get_season_avg_pace, get_qualifying_times
from features import build_feature_set
from train_model import train, predict, FEATURE_COLUMNS
 
TARGET_ROUND = 13          # <-- set this to the round you want predicted


TARGET_YEAR = 2026
RACE_NAME_LOOKUP = {
    # optional: fill in as you go so printouts look nice.
    # round_number: "Race Name"
    13: "Spanish GP (Madrid)",
    16: "Azerbaijan GP (Baku)",
}

# Training always = every round BEFORE the target round.
# This guarantees the target race can never leak into training data.
TRAIN_YEAR = TARGET_YEAR
TRAIN_ROUNDS = list(range(1, TARGET_ROUND))


def get_race_name(round_num: int) -> str:
    return RACE_NAME_LOOKUP.get(round_num, f"Round {round_num}")


def main():
    if not TRAIN_ROUNDS:
        raise ValueError(
            f"TARGET_ROUND is {TARGET_ROUND}, so there are no prior rounds "
            f"to train on. Pick a later round."
        )

    print(f"\nFetching {TRAIN_YEAR} race data (rounds {TRAIN_ROUNDS[0]}-{TRAIN_ROUNDS[-1]}) for training...")
    try:
        race_laps = get_season_avg_pace(TRAIN_YEAR, TRAIN_ROUNDS)
    except Exception as e:
        print(f"ERROR fetching training data: {e}")
        return

    print(f"\nFetching {TARGET_YEAR} {get_race_name(TARGET_ROUND)} qualifying data...")
    try:
        qualifying = get_qualifying_times(TARGET_YEAR, TARGET_ROUND)
    except Exception as e:
        print(f"ERROR fetching qualifying data: {e}")
        print("This usually means qualifying for this round hasn't happened yet.")
        return

    if qualifying is None or qualifying.empty:
        print("No qualifying data available yet for this round. Try again after quali.")
        return

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

    if merged.empty:
        print("ERROR: No overlapping drivers between qualifying and training data. Check driver codes.")
        return

    print("\nTraining model...")
    model, mae = train(merged)

    print("Predicting race pace...")
    qualifying_features = qualifying_features.dropna(subset=FEATURE_COLUMNS).copy()
    qualifying_features["PredictedRaceTime (s)"] = predict(model, qualifying_features)
    results = qualifying_features.sort_values("PredictedRaceTime (s)")

    print(f"\nPredicted {TARGET_YEAR} {get_race_name(TARGET_ROUND)} Result\n")
    print(results[["Driver", "Team", "PredictedRaceTime (s)"]].to_string(index=False))

    winner = results.iloc[0]
    print(f"\nPredicted Winner: {winner['Driver']} ({winner['Team']})")
    print(f"Model Error (MAE): {mae:.2f} seconds")

    # --------------------------------------------------------
    # Optional: if the target race has ALREADY happened,
    # compare prediction against the real result.
    # --------------------------------------------------------
    print(f"\nChecking whether {get_race_name(TARGET_ROUND)} has actually run yet...")
    try:
        actual = get_season_avg_pace(TARGET_YEAR, [TARGET_ROUND])
    except Exception:
        actual = None

    if actual is not None and not actual.empty:
        actual_renamed = actual.rename(columns={"LapTime (s)": "ActualRaceTime (s)"})
        comparison = results.merge(
            actual_renamed, left_on="DriverCode", right_on="Driver", suffixes=("", "_actual")
        )
        comparison = comparison.sort_values("PredictedRaceTime (s)")
        comparison["Error (s)"] = (
            comparison["PredictedRaceTime (s)"] - comparison["ActualRaceTime (s)"]
        ).abs()

        print(f"\n{get_race_name(TARGET_ROUND)} ALREADY RAN — Predicted vs Actual:\n")
        print(comparison[["Driver", "Team", "PredictedRaceTime (s)", "ActualRaceTime (s)", "Error (s)"]].to_string(index=False))

        actual_sorted = actual.sort_values("LapTime (s)")
        actual_winner_code = actual_sorted.iloc[0]["Driver"]
        actual_winner_name = code_to_name.get(actual_winner_code, actual_winner_code)
        print(f"\nActual winner: {actual_winner_name}")
        print(f"Predicted winner: {winner['Driver']}")
        print(f"Correct prediction: {actual_winner_name == winner['Driver']}")
    else:
        print("Race hasn't happened yet — this is a true future prediction.")


if __name__ == "__main__":
    main()
"""
F1 Race Predictor 2026

1 = Predict next upcoming race
2 = Analyze a completed race
"""

import pandas as pd
import fastf1

from drivers_2026 import DRIVER_CODES
from fetch_data import get_season_avg_pace, get_qualifying_times
from features import build_feature_set
from train_model import train, predict, FEATURE_COLUMNS


YEAR = 2026


def calendar():
    schedule = fastf1.get_event_schedule(YEAR)

    schedule = schedule[
        pd.to_numeric(
            schedule["RoundNumber"],
            errors="coerce"
        ).notna()
    ].copy()

    schedule["RoundNumber"] = schedule["RoundNumber"].astype(int)

    schedule["EventDate"] = pd.to_datetime(
        schedule["EventDate"],
        utc=True,
        errors="coerce"
    )

    return schedule[schedule["RoundNumber"] > 0]


def prepare(qualifying, race_data):
    code_to_name = {
        v: k for k, v in DRIVER_CODES.items()
    }

    qualifying = qualifying.copy()

    qualifying["Driver"] = qualifying[
        "DriverCode"
    ].map(code_to_name)

    qualifying.dropna(
        subset=["Driver"],
        inplace=True
    )

    features = build_feature_set(qualifying)

    features.dropna(
        subset=FEATURE_COLUMNS,
        inplace=True
    )

    merged = features.merge(
        race_data,
        left_on="DriverCode",
        right_on="Driver",
        suffixes=("", "_race")
    ).dropna()

    return features, merged, code_to_name


def predict_race(
    round_no,
    race_name,
    schedule,
    compare=False
):
    print(f"\n{race_name} — Round {round_no}")

    # Use only races before the target race for training.
    train_rounds = schedule[
        schedule["RoundNumber"] < round_no
    ]["RoundNumber"].tolist()

    if not train_rounds:
        print(
            "Not enough previous races "
            "to train the model."
        )
        return None

    print(
        f"Training on rounds 1-{round_no - 1}..."
    )

    try:
        race_data = get_season_avg_pace(
            YEAR,
            train_rounds
        )

        qualifying = get_qualifying_times(
            YEAR,
            round_no
        )

    except Exception as e:
        print(
            f"\nQualifying/data is not available: {e}"
        )
        return None

    if qualifying.empty:
        print(
            "\nNo qualifying data available yet."
        )
        return None

    try:
        features, merged, code_to_name = prepare(
            qualifying,
            race_data
        )

    except Exception as e:
        print(
            f"\nData preparation error: {e}"
        )
        return None

    if merged.empty:
        print(
            "\nNo matching driver data."
        )
        return None

    print(
        "Training Gradient Boosting model..."
    )

    model, mae = train(merged)

    features["PredictedRaceTime (s)"] = predict(
        model,
        features
    )

    results = features.sort_values(
        "PredictedRaceTime (s)"
    ).reset_index(drop=True)

    results["Predicted Position"] = (
        results.index + 1
    )

    print(
        f"\nPREDICTED "
        f"{race_name.upper()} RESULT\n"
    )

    print(
        results[
            [
                "Predicted Position",
                "Driver",
                "Team",
                "PredictedRaceTime (s)"
            ]
        ].to_string(index=False)
    )

    winner = results.iloc[0]

    print(
        f"\nPredicted Winner: "
        f"{winner['Driver']}"
    )

    print(
        f"Team: {winner['Team']}"
    )

    print(
        f"Model MAE: {mae:.2f} seconds"
    )

    # ------------------------------------------------
    # Streamlit result
    # ------------------------------------------------

    prediction_result = {
        "Race": race_name,
        "Predicted Winner": winner["Driver"],
        "Team": winner["Team"],
        "Model MAE": f"{mae:.2f} seconds",
        "Results": results[
            [
                "Predicted Position",
                "Driver",
                "Team",
                "PredictedRaceTime (s)"
            ]
        ].copy()
    }

    # Compare with actual result when requested.
    if compare:

        try:
            actual = get_season_avg_pace(
                YEAR,
                [round_no]
            )

        except Exception:
            print(
                "\nActual race data unavailable."
            )

            return prediction_result

        actual = actual.rename(
            columns={
                "LapTime (s)": "ActualRaceTime (s)"
            }
        )

        comparison = results.merge(
            actual,
            left_on="DriverCode",
            right_on="Driver",
            suffixes=("", "_actual")
        )

        comparison["Error (s)"] = (
            comparison["PredictedRaceTime (s)"]
            - comparison["ActualRaceTime (s)"]
        ).abs()

        print(
            "\nPREDICTED VS ACTUAL\n"
        )

        print(
            comparison[
                [
                    "Predicted Position",
                    "Driver",
                    "Team",
                    "PredictedRaceTime (s)",
                    "ActualRaceTime (s)",
                    "Error (s)"
                ]
            ].to_string(index=False)
        )

        actual_winner_code = (
            actual.sort_values(
                "ActualRaceTime (s)"
            ).iloc[0]["Driver"]
        )

        actual_winner = code_to_name.get(
            actual_winner_code,
            actual_winner_code
        )

        print(
            f"\nPredicted Winner: "
            f"{winner['Driver']}"
        )

        print(
            f"Actual Winner: "
            f"{actual_winner}"
        )

        correct_prediction = (
            winner["Driver"] == actual_winner
        )

        print(
            f"Correct Prediction: "
            f"{correct_prediction}"
        )

        prediction_result[
            "Actual Winner"
        ] = actual_winner

        prediction_result[
            "Correct Prediction"
        ] = correct_prediction

    # IMPORTANT:
    # Return the result to Streamlit.
    return prediction_result


def main():

    print(
        "\nF1 RACE PREDICTOR 2026"
    )

    print(
        "\n1. Predict next upcoming race"
    )

    print(
        "2. Analyze a completed race"
    )

    choice = input(
        "\nEnter 1 or 2: "
    ).strip()

    try:
        schedule = calendar()

    except Exception as e:
        print(
            f"\nCould not load F1 calendar: {e}"
        )
        return

    today = pd.Timestamp.now(tz="UTC")

    if choice == "1":

        upcoming = schedule[
            schedule["EventDate"] >= today
        ].sort_values("RoundNumber")

        if upcoming.empty:
            print(
                "\nNo upcoming races found."
            )
            return

        race = upcoming.iloc[0]

        predict_race(
            int(race["RoundNumber"]),
            race["EventName"],
            schedule,
            compare=False
        )

    elif choice == "2":

        completed = schedule[
            schedule["EventDate"] < today
        ].sort_values("RoundNumber")

        print(
            "\nCompleted races:"
        )

        for _, race in completed.iterrows():

            print(
                f"{int(race['RoundNumber'])} - "
                f"{race['EventName']}"
            )

        try:
            round_no = int(
                input(
                    "\nEnter round number: "
                )
            )

        except ValueError:
            print(
                "\nEnter a valid round number."
            )
            return

        if round_no not in (
            completed["RoundNumber"].values
        ):
            print(
                "\nThat race is not completed."
            )
            return

        race_name = completed.loc[
            completed["RoundNumber"] == round_no,
            "EventName"
        ].iloc[0]

        predict_race(
            round_no,
            race_name,
            schedule,
            compare=True
        )

    else:

        print(
            "\nInvalid choice. "
            "Enter 1 or 2."
        )


if __name__ == "__main__":
    main()
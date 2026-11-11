"""
F1 Race Predictor 2026

1 = Predict next upcoming race
2 = Analyze a completed race
"""

import time
import pandas as pd
import fastf1

from drivers_2026 import DRIVER_CODES
from fetch_data import (
    get_season_avg_pace,
    get_qualifying_times
)
from features import build_feature_set
from train_model import train, predict, FEATURE_COLUMNS


YEAR = 2026


def calendar():
    """Load and clean the F1 season calendar."""

    schedule = fastf1.get_event_schedule(YEAR)

    schedule = schedule[
        pd.to_numeric(
            schedule["RoundNumber"],
            errors="coerce"
        ).notna()
    ].copy()

    schedule["RoundNumber"] = (
        schedule["RoundNumber"].astype(int)
    )

    schedule["EventDate"] = pd.to_datetime(
        schedule["EventDate"],
        utc=True,
        errors="coerce"
    )

    return schedule[
        schedule["RoundNumber"] > 0
    ]


def prepare(qualifying, race_data):
    """Prepare qualifying and historical data for prediction."""

    code_to_name = {
        code: name
        for name, code in DRIVER_CODES.items()
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
    """Predict a race or analyze a completed race."""

    print(f"\n{race_name} — Round {round_no}")

    # Only use rounds before the target race.
    train_rounds = tuple(
        schedule.loc[
            schedule["RoundNumber"] < round_no,
            "RoundNumber"
        ].astype(int).tolist()
    )

    if not train_rounds:
        message = (
            "Not enough previous races "
            "to train the model."
        )

        print(message)
        return {"error": message}

    print(
        f"Training data rounds: {train_rounds}"
    )

    start = time.perf_counter()

    try:
        race_data = get_season_avg_pace(
            YEAR,
            train_rounds
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start

        message = (
            "Historical race data failed to load: "
            f"{exc}"
        )

        print(f"\n{message}")
        print(
            f"Historical loading time: "
            f"{elapsed:.2f} seconds"
        )

        return {"error": message}

    print(
        f"Historical data loading: "
        f"{time.perf_counter() - start:.2f} seconds"
    )

    if race_data.empty:
        message = "Historical race data is empty."
        print(message)
        return {"error": message}


    start = time.perf_counter()

    try:
        qualifying = get_qualifying_times(
            YEAR,
            int(round_no)
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start

        message = (
            f"Qualifying data failed for "
            f"Round {round_no}: {exc}"
        )

        print(f"\n{message}")
        print(
            f"Qualifying loading time: "
            f"{elapsed:.2f} seconds"
        )

        return {"error": message}

    print(
        f"Qualifying data loading: "
        f"{time.perf_counter() - start:.2f} seconds"
    )

    if qualifying.empty:
        message = (
            "No qualifying lap data is available "
            f"for Round {round_no}."
        )

        print(message)
        return {"error": message}


    start = time.perf_counter()

    try:
        features, merged, code_to_name = prepare(
            qualifying,
            race_data
        )

    except Exception as exc:
        message = f"Data preparation failed: {exc}"

        print(f"\n{message}")
        return {"error": message}

    print(
        f"Feature preparation: "
        f"{time.perf_counter() - start:.2f} seconds"
    )

    if features.empty:
        message = (
            "No usable qualifying features remain "
            "after data preparation."
        )

        print(message)
        return {"error": message}

    if merged.empty:
        message = (
            "No matching drivers between qualifying "
            "and historical race data."
        )

        print(message)
        return {"error": message}


    start = time.perf_counter()

    try:
        print(
            "\nTraining Gradient Boosting model..."
        )

        model, mae = train(merged)

        features["PredictedRaceTime (s)"] = predict(
            model,
            features
        )

    except Exception as exc:
        message = f"Model training or prediction failed: {exc}"

        print(f"\n{message}")
        return {"error": message}

    print(
        f"Model training and prediction: "
        f"{time.perf_counter() - start:.2f} seconds"
    )


    results = features.sort_values(
        "PredictedRaceTime (s)"
    ).reset_index(drop=True)

    results["Predicted Position"] = (
        results.index + 1
    )

    result_columns = [
        "Predicted Position",
        "Driver",
        "Team",
        "PredictedRaceTime (s)"
    ]

    print(
        f"\nPREDICTED {race_name.upper()} RESULT\n"
    )

    print(
        results[result_columns].to_string(
            index=False
        )
    )

    winner = results.iloc[0]

    print(
        f"\nPredicted Winner: {winner['Driver']}"
    )

    print(f"Team: {winner['Team']}")
    print(f"Model MAE: {mae:.2f} seconds")

    prediction_result = {
        "Race": race_name,
        "Predicted Winner": winner["Driver"],
        "Team": winner["Team"],
        "Model MAE": f"{mae:.2f} seconds",
        "Results": results[result_columns].copy()
    }


    if compare:

        try:
            actual = get_season_avg_pace(
                YEAR,
                (int(round_no),)
            )

        except Exception as exc:
            print(
                f"\nActual race data unavailable: {exc}"
            )

            prediction_result[
                "Comparison Error"
            ] = str(exc)

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

        if not comparison.empty:

            comparison["Error (s)"] = (
                comparison["PredictedRaceTime (s)"]
                - comparison["ActualRaceTime (s)"]
            ).abs()

            print(
                "\nPREDICTED VS ACTUAL AVERAGE LAP TIME\n"
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

            fastest_actual_code = (
                actual.sort_values(
                    "ActualRaceTime (s)"
                ).iloc[0]["Driver"]
            )

            fastest_actual_driver = code_to_name.get(
                fastest_actual_code,
                fastest_actual_code
            )

            print(
                "\nDriver with lowest average race lap time: "
                f"{fastest_actual_driver}"
            )

            # This is NOT necessarily the official race winner.
            prediction_result[
                "Fastest Average Lap Driver"
            ] = fastest_actual_driver

            prediction_result[
                "Comparison"
            ] = comparison[
                [
                    "Predicted Position",
                    "Driver",
                    "Team",
                    "PredictedRaceTime (s)",
                    "ActualRaceTime (s)",
                    "Error (s)"
                ]
            ].copy()

        else:
            print(
                "\nNo matching drivers for comparison."
            )

            prediction_result[
                "Comparison Error"
            ] = "No matching drivers for comparison."

    return prediction_result


def main():

    print("\nF1 RACE PREDICTOR 2026")

    print("\n1. Predict next upcoming race")
    print("2. Analyze a completed race")

    choice = input(
        "\nEnter 1 or 2: "
    ).strip()

    try:
        schedule = calendar()

    except Exception as exc:
        print(
            f"\nCould not load F1 calendar: {exc}"
        )
        return

    today = pd.Timestamp.now(tz="UTC")

    if choice == "1":

        upcoming = schedule[
            schedule["EventDate"] >= today
        ].sort_values("RoundNumber")

        if upcoming.empty:
            print("\nNo upcoming races found.")
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

        print("\nCompleted races:")

        for _, race in completed.iterrows():
            print(
                f"{int(race['RoundNumber'])} - "
                f"{race['EventName']}"
            )

        try:
            round_no = int(
                input("\nEnter round number: ")
            )

        except ValueError:
            print("\nEnter a valid round number.")
            return

        if round_no not in completed[
            "RoundNumber"
        ].values:

            print("\nThat race is not completed.")
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
            "\nInvalid choice. Enter 1 or 2."
        )


if __name__ == "__main__":
    main()
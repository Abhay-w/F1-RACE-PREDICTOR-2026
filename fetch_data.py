"""
fetch_data.py

Handles FastF1 API calls for:
- historical race data used for training
- qualifying data used for future prediction
"""

import fastf1
import pandas as pd


fastf1.Cache.enable_cache("f1_cache")


def get_race_lap_times(
    year: int,
    round_no: int
) -> pd.DataFrame:

    """
    Get average race lap time per driver.
    """

    session = fastf1.get_session(
        year,
        round_no,
        "R"
    )

    session.load()

    laps = session.laps[
        ["Driver", "LapTime"]
    ].copy()

    laps.dropna(
        subset=["LapTime"],
        inplace=True
    )

    laps["LapTime (s)"] = (
        laps["LapTime"]
        .dt.total_seconds()
    )

    avg_laps = (
        laps
        .groupby(
            "Driver",
            as_index=False
        )["LapTime (s)"]
        .mean()
    )

    return avg_laps


def get_season_avg_pace(
    year: int,
    rounds: list
) -> pd.DataFrame:

    """
    Calculate each driver's average race pace
    across all completed training rounds.
    """

    all_laps = []

    for round_no in rounds:

        try:

            laps = get_race_lap_times(
                year,
                round_no
            )

            if not laps.empty:
                all_laps.append(laps)

        except Exception as e:

            print(
                f"  Skipping round "
                f"{round_no}: {e}"
            )

    if not all_laps:

        raise ValueError(
            "No race data could be loaded."
        )

    combined = pd.concat(
        all_laps,
        ignore_index=True
    )

    season_avg = (
        combined
        .groupby(
            "Driver",
            as_index=False
        )["LapTime (s)"]
        .mean()
    )

    return season_avg


def get_qualifying_times(
    year: int,
    round_no: int
) -> pd.DataFrame:

    """
    Get each driver's fastest qualifying lap.

    This is the prediction input for the target race.
    """

    session = fastf1.get_session(
        year,
        round_no,
        "Q"
    )

    session.load()

    quali = session.laps[
        ["Driver", "LapTime"]
    ].copy()

    quali.dropna(
        subset=["LapTime"],
        inplace=True
    )

    quali["QualifyingTime (s)"] = (
        quali["LapTime"]
        .dt.total_seconds()
    )

    best_quali = (
        quali
        .groupby(
            "Driver",
            as_index=False
        )["QualifyingTime (s)"]
        .min()
    )

    best_quali.rename(
        columns={
            "Driver": "DriverCode"
        },
        inplace=True
    )

    return best_quali
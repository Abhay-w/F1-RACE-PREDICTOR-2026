"""
fetch_data.py

Handles FastF1 API calls for:
- Historical race data used for training
- Qualifying data used for predictions
"""

import fastf1
import pandas as pd
import streamlit as st


fastf1.Cache.enable_cache("f1_cache")


@st.cache_data(show_spinner=False)
def get_race_lap_times(
    year: int,
    round_no: int
) -> pd.DataFrame:
    """Get average race lap time per driver."""

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
        laps["LapTime"].dt.total_seconds()
    )

    return (
        laps.groupby(
            "Driver",
            as_index=False
        )["LapTime (s)"]
        .mean()
    )


@st.cache_data(show_spinner=False)
def get_season_avg_pace(
    year: int,
    rounds: tuple
) -> pd.DataFrame:
    """
    Calculate average driver race pace across
    the specified training rounds.

    The rounds argument is a tuple so that it can
    be used consistently as a cache key.
    """

    all_laps = []
    failed_rounds = []

    for round_no in rounds:
        try:
            laps = get_race_lap_times(
                year,
                round_no
            )

            if not laps.empty:
                all_laps.append(laps)
            else:
                failed_rounds.append(round_no)

        except Exception as exc:
            failed_rounds.append(round_no)

            print(
                f"Round {round_no} failed: {exc}"
            )

    if not all_laps:
        raise ValueError(
            "No historical race data could be loaded."
        )

    if failed_rounds:
        print(
            "Historical rounds that failed to load: "
            f"{failed_rounds}"
        )

    combined = pd.concat(
        all_laps,
        ignore_index=True
    )

    season_avg = (
        combined.groupby(
            "Driver",
            as_index=False
        )["LapTime (s)"]
        .mean()
    )

    return season_avg


@st.cache_data(show_spinner=False)
def get_qualifying_times(
    year: int,
    round_no: int
) -> pd.DataFrame:
    """Get each driver's fastest qualifying lap."""

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
        quali["LapTime"].dt.total_seconds()
    )

    best_quali = (
        quali.groupby(
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
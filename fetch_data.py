"""
fetch_data.py
Handles all FastF1 API calls: pulling historical race lap times (for
training) and qualifying results (for prediction input).
"""
import fastf1
import pandas as pd

fastf1.Cache.enable_cache("f1_cache")


def get_race_lap_times(year: int, round_no: int) -> pd.DataFrame:
    """
    Pull average clean lap time per driver for a given race.
    Used as the training target (y).
    """
    session = fastf1.get_session(year, round_no, "R")
    session.load()

    laps = session.laps[["Driver", "LapTime"]].copy()
    laps.dropna(subset=["LapTime"], inplace=True)
    laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()

    # Average per driver instead of using every single lap - reduces
    # noise from pit stops, safety cars, and outlier laps.
    avg_laps = laps.groupby("Driver", as_index=False)["LapTime (s)"].mean()
    return avg_laps


def get_season_avg_pace(year: int, rounds: list) -> pd.DataFrame:
    """
    Average a driver's race pace across multiple rounds already run
    this season. Used instead of "same race last year" when the target
    race is new to the calendar (e.g. Madrid in 2026), or simply to get
    a more stable training signal from recent form.
    """
    all_laps = []
    for r in rounds:
        try:
            laps = get_race_lap_times(year, r)
            all_laps.append(laps)
        except Exception as e:
            print(f"  Skipping round {r}: {e}")

    if not all_laps:
        raise ValueError("No race data could be loaded for the given rounds.")

    combined = pd.concat(all_laps, ignore_index=True)
    season_avg = combined.groupby("Driver", as_index=False)["LapTime (s)"].mean()
    return season_avg


def get_qualifying_times(year: int, round_no: int) -> pd.DataFrame:
    """
    Pull best qualifying lap per driver for a given race.
    Used as the model's input feature (X) at prediction time.
    """
    session = fastf1.get_session(year, round_no, "Q")
    session.load()

    quali = session.laps[["Driver", "LapTime"]].copy()
    quali.dropna(subset=["LapTime"], inplace=True)
    quali["QualifyingTime (s)"] = quali["LapTime"].dt.total_seconds()

    # Best lap per driver (their fastest qualifying lap)
    best_quali = quali.groupby("Driver", as_index=False)["QualifyingTime (s)"].min()
    best_quali.rename(columns={"Driver": "DriverCode"}, inplace=True)
    return best_quali
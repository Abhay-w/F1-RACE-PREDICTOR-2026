import numpy as np
import pandas as pd

from jolpica_data import fetch_schedule

DRIVER_WINDOW = 5
TEAM_WINDOW = 5
DNF_WINDOW = 10
DEFAULT_FINISH = 10.5
DEFAULT_DNF = 0.10

FORM_FEATURES = ["DriverFormFinish", "TeamFormFinish",
                 "CircuitFormFinish", "DriverDnfRate"]
GRID_FEATURES = ["QualifyingPosition"] + FORM_FEATURES


def is_dnf(status):
    s = str(status)
    return not (s == "Finished" or s == "Lapped" or s.startswith("+"))


def add_form_features(df):
    df = df.sort_values(["Season", "Round"], kind="stable").reset_index(drop=True)
    df["Dnf"] = df["Status"].map(lambda s: np.nan if pd.isna(s) else float(is_dnf(s)))

    by_driver = df.groupby("DriverCode", sort=False)
    df["DriverFormFinish"] = by_driver["FinishPosition"].transform(
        lambda s: s.shift(1).rolling(DRIVER_WINDOW, min_periods=1).mean())
    df["DriverDnfRate"] = by_driver["Dnf"].transform(
        lambda s: s.shift(1).rolling(DNF_WINDOW, min_periods=1).mean())
    df["CircuitFormFinish"] = df.groupby(["DriverCode", "CircuitId"], sort=False)[
        "FinishPosition"].transform(lambda s: s.shift(1).expanding().mean())

    team = (df.groupby(["Season", "Round", "Team"], sort=False)["FinishPosition"]
              .mean().rename("TeamRaceFinish").reset_index()
              .sort_values(["Season", "Round"], kind="stable"))
    team["TeamFormFinish"] = team.groupby("Team", sort=False)["TeamRaceFinish"].transform(
        lambda s: s.shift(1).rolling(TEAM_WINDOW, min_periods=1).mean())
    df = df.merge(team[["Season", "Round", "Team", "TeamFormFinish"]],
                  on=["Season", "Round", "Team"], how="left")

    df["TeamFormFinish"] = df["TeamFormFinish"].fillna(DEFAULT_FINISH)
    df["DriverFormFinish"] = df["DriverFormFinish"].fillna(df["TeamFormFinish"])
    df["CircuitFormFinish"] = df["CircuitFormFinish"].fillna(df["DriverFormFinish"])
    df["DriverDnfRate"] = df["DriverDnfRate"].fillna(DEFAULT_DNF)
    return df


def _features_for_round(history, roster, season, round_no, circuit_id):
    before = history[(history["Season"] < season)
                     | ((history["Season"] == season) & (history["Round"] < round_no))]
    if before.empty:
        raise ValueError("No earlier races to learn form from.")

    new = roster.copy()
    new["Season"], new["Round"] = season, round_no
    new["CircuitId"] = circuit_id or "unknown"
    new["FinishPosition"], new["Status"] = np.nan, np.nan
    if "QualifyingPosition" not in new:
        new["QualifyingPosition"] = np.nan

    combined = add_form_features(pd.concat([before, new], ignore_index=True))
    mask = (combined["Season"] == season) & (combined["Round"] == round_no)
    return combined[mask].reset_index(drop=True)


def _circuit_for(season, round_no):
    try:
        schedule = fetch_schedule(season)
        match = schedule.loc[schedule["Round"] == round_no, "CircuitId"]
        return match.iloc[0] if len(match) else None
    except Exception:
        return None


def next_race_features(history, season, round_no, circuit_id=None):
    before = history[(history["Season"] < season)
                     | ((history["Season"] == season) & (history["Round"] < round_no))]
    if before.empty:
        raise ValueError("No earlier races to learn form from.")
    last = before.sort_values(["Season", "Round"]).iloc[-1]
    roster = before[(before["Season"] == last["Season"])
                    & (before["Round"] == last["Round"])][["DriverCode", "Driver", "Team"]]
    circuit_id = circuit_id or _circuit_for(season, round_no)
    return _features_for_round(history, roster, season, round_no, circuit_id)


def grid_race_features(history, season, round_no, qualifying, circuit_id=None):
    roster = qualifying[["DriverCode", "Driver", "Team", "QualifyingPosition"]]
    circuit_id = circuit_id or _circuit_for(season, round_no)
    return _features_for_round(history, roster, season, round_no, circuit_id)
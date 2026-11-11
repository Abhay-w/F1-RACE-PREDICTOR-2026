"""
features.py
feature engineering beyound raw qaulifying time. this is one of the 
main upgrades over the original single-feature model: it adds team
performance as a second signal, since car pace matters as much as
a single qualifying lap.
"""
import pandas as pd
from drivers_2026 import DRIVER_TEAMS

def add_team_column(df: pd.DataFrame, driver_col: str = "Driver") -> pd.DataFrame:
    """
    Attach each driver's 2026 team as a new column.
    """
    df = df.copy()
    df["Team"] = df[driver_col].map(DRIVER_TEAMS)
    return df

def add_team_avg_pace(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Add a 'TeamAvgTime' feature: the average of both teammates' times.
    this gives the model a sense of car performance independent of
    one driver's individual form on a given lap.
    """
    df = df.copy()
    team_avg = df.groupby("Team")[time_col].transform("mean")
    df["TeamAvgTime (s)"] = team_avg
    return df

def build_feature_set(qualifying_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature pipeline applied to a qualifying results dataframe.
    Expect columns: ['Driver', 'QualifyingTime (s)']
    """
    df = add_team_column(qualifying_df, driver_col="Driver")
    df = add_team_avg_pace(df, time_col="QualifyingTime (s)")
    return df
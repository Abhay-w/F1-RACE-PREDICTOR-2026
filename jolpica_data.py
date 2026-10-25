import requests
import pandas as pd
import streamlit as st

BASE_URL = "https://api.jolpi.ca/ergast/f1"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_qualifying(year: int, round_no: int) -> pd.DataFrame:
    """Fetch qualifying positions for a Grand Prix."""

    url = f"{BASE_URL}/{year}/{round_no}/qualifying.json"
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    races = response.json()["MRData"]["RaceTable"]["Races"]

    if not races:
        return pd.DataFrame()

    rows = []

    for item in races[0].get("QualifyingResults", []):
        driver = item["Driver"]
        team = item["Constructor"]

        rows.append({
            "Season": year,
            "Round": round_no,
            "DriverCode": driver.get("code", ""),
            "Driver": (
                f"{driver.get('givenName', '')} "
                f"{driver.get('familyName', '')}"
            ).strip(),
            "Team": team.get("name", ""),
            "QualifyingPosition": int(item["position"]),
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_results(year: int, round_no: int) -> pd.DataFrame:
    """Fetch official race finishing positions."""

    url = f"{BASE_URL}/{year}/{round_no}/results.json"
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    races = response.json()["MRData"]["RaceTable"]["Races"]

    if not races:
        return pd.DataFrame()

    rows = []

    for item in races[0].get("Results", []):
        driver = item["Driver"]
        team = item["Constructor"]

        try:
            finish_position = int(item["position"])
        except (ValueError, TypeError, KeyError):
            finish_position = None

        rows.append({
            "Season": year,
            "Round": round_no,
            "DriverCode": driver.get("code", ""),
            "Driver": (
                f"{driver.get('givenName', '')} "
                f"{driver.get('familyName', '')}"
            ).strip(),
            "Team": team.get("name", ""),
            "FinishPosition": finish_position,
            "Status": item.get("status", ""),
        })

    return pd.DataFrame(rows)


def fetch_race_training_data(
    start_year: int = 2018,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Combine historical qualifying and race results."""

    combined = []

    for year in range(start_year, end_year + 1):
        for round_no in range(1, 30):
            try:
                qualifying = fetch_qualifying(year, round_no)

                if qualifying.empty:
                    continue

                results = fetch_results(year, round_no)

                if results.empty:
                    continue

                merged = qualifying.merge(
                    results[
                        [
                            "Season",
                            "Round",
                            "DriverCode",
                            "FinishPosition",
                        ]
                    ],
                    on=["Season", "Round", "DriverCode"],
                    how="inner",
                )

                merged = merged.dropna(subset=["FinishPosition"])

                if not merged.empty:
                    combined.append(merged)

            except requests.RequestException:
                continue
            except (KeyError, ValueError, TypeError):
                continue

    if not combined:
        raise ValueError(
            "No historical data retrieved. Check your internet connection "
            "or the Jolpica API."
        )

    return pd.concat(combined, ignore_index=True)
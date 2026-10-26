"""
"Next Race Forecast" page.

Predicts a race BEFORE qualifying, using only the results of earlier races
(driver form and team form). Can also back-test a completed race: the model
is shown only the races before it, and its forecast is compared with what
really happened.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

from form_features import FORM_FEATURES, next_race_features
from jolpica_data import fetch_race_training_data

PROJECT_DIR = Path(__file__).resolve().parent
FORM_MODEL_PATH = PROJECT_DIR / "form_model.joblib"
FORM_METRICS_PATH = PROJECT_DIR / "form_metrics.json"

# Five races of form are needed, so the previous season is enough history.
HISTORY_FROM_SEASON = 2025


@st.cache_resource
def load_form_model():
    if not FORM_MODEL_PATH.exists():
        return None
    return joblib.load(FORM_MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_form_metrics():
    if not FORM_METRICS_PATH.exists():
        return {}
    with open(FORM_METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(ttl=3600, show_spinner=False)
def load_recent_history(season):
    return fetch_race_training_data(HISTORY_FROM_SEASON, season)


def rank_drivers(features, model):
    """Score every driver and assign predicted positions 1..N."""

    ranked = features.copy()
    ranked["Score"] = model.predict(ranked[FORM_FEATURES])
    ranked = ranked.sort_values(["Score", "DriverCode"]).reset_index(drop=True)
    ranked["PredictedPosition"] = range(1, len(ranked) + 1)
    return ranked


def show_podium(ranked, make_podium_card):
    labels = ["P1 • RACE WINNER", "P2 • SECOND PLACE", "P3 • THIRD PLACE"]
    columns = st.columns(3)

    for index, column in enumerate(columns):
        if index >= len(ranked):
            break
        row = ranked.iloc[index]
        with column:
            make_podium_card(index + 1, labels[index], row["Driver"], row["Team"])


def render_forecast_page(season, fetch_schedule, make_podium_card):
    st.markdown('<div class="section-title">NEXT RACE FORECAST</div>',
                unsafe_allow_html=True)

    st.caption(
        "Predicts a race before qualifying, using only the results of earlier "
        "races: each driver's and each team's average qualifying and finishing "
        "position over the last five races."
    )

    model = load_form_model()

    if model is None:
        st.error("The pre-race model could not be found. "
                 "Please run form_model.py first.")
        return

    try:
        with st.spinner("Loading the schedule and recent results..."):
            schedule = fetch_schedule(season)
            history = load_recent_history(season)
    except (requests.RequestException, ValueError, KeyError) as error:
        st.error(f"Could not load the race data: {error}")
        return

    if history.empty or schedule.empty:
        st.warning("No race data was returned by the API.")
        return

    completed = sorted(
        int(r) for r in history.loc[history["Season"] == season, "Round"].unique())
    last_done = completed[-1] if completed else 0
    next_round = last_done + 1
    last_round = int(schedule["Round"].max())

    modes = []
    if next_round <= last_round:
        modes.append("Next race")
    if completed:
        modes.append("Back-test a completed race")

    mode = st.radio("FORECAST MODE", modes, horizontal=True)
    names = dict(zip(schedule["Round"], schedule["Race"]))

    # ------------------------------------------------------------ next race
    if mode == "Next race":
        st.markdown(f"**Round {next_round} — {names.get(next_round, 'Next race')}**")
        st.caption(f"Uses the results of rounds 1 to {last_done} of {season} "
                   "and the end of the previous season. Qualifying is not used.")

        try:
            features = next_race_features(history, season, next_round)
        except ValueError as error:
            st.error(str(error))
            return

        ranked = rank_drivers(features, model)

        st.markdown('<div class="section-title">FORECAST PODIUM</div>',
                    unsafe_allow_html=True)
        show_podium(ranked, make_podium_card)

        st.markdown('<div class="section-title">FULL FORECAST</div>',
                    unsafe_allow_html=True)

        table = ranked[["PredictedPosition", "Driver", "Team",
                        "DriverFormFinish", "TeamFormFinish", "Score"]].rename(columns={
            "PredictedPosition": "Predicted Position",
            "Team": "Constructor",
            "DriverFormFinish": "Driver form (avg finish, last 5)",
            "TeamFormFinish": "Team form (avg finish, last 5)",
            "Score": "Regression Score",
        })
        st.dataframe(table.round(2), use_container_width=True, hide_index=True)

        st.download_button(
            "Download Forecast CSV",
            table.to_csv(index=False).encode("utf-8"),
            file_name=f"f1_{season}_round_{next_round}_forecast.csv",
            mime="text/csv",
        )

    # ------------------------------------------------------------- back-test
    else:
        round_no = st.selectbox(
            "SELECT A COMPLETED ROUND",
            completed,
            index=len(completed) - 1,
            format_func=lambda r: f"Round {r} — {names.get(r, 'Grand Prix')}",
        )

        before = history[
            (history["Season"] < season)
            | ((history["Season"] == season) & (history["Round"] < round_no))
        ]

        if before.empty:
            st.info("There are no earlier races to learn the form from.")
            return

        st.caption(f"The model sees only the races before round {round_no}.")

        features = next_race_features(before, season, round_no)
        ranked = rank_drivers(features, model)

        actual = history[
            (history["Season"] == season) & (history["Round"] == round_no)
        ][["DriverCode", "FinishPosition", "QualifyingPosition"]]

        compared = ranked.drop(
            columns=["FinishPosition", "QualifyingPosition"], errors="ignore"
        ).merge(actual, on="DriverCode", how="inner")
        compared["AbsoluteError"] = (
            compared["PredictedPosition"] - compared["FinishPosition"]).abs()

        st.markdown('<div class="section-title">FORECAST PODIUM</div>',
                    unsafe_allow_html=True)
        show_podium(ranked, make_podium_card)

        if compared.empty:
            st.info("The drivers of the forecast did not match the official result.")
            return

        winner = compared.loc[compared["PredictedPosition"] == 1, "FinishPosition"]
        podium_hits = int(((compared["PredictedPosition"] <= 3)
                           & (compared["FinishPosition"] <= 3)).sum())

        st.markdown('<div class="section-title">FORECAST VS ACTUAL</div>',
                    unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Race MAE", f"{compared['AbsoluteError'].mean():.2f}")
        c2.metric("Drivers compared", len(compared))
        c3.metric("Podium drivers found (of 3)", podium_hits)

        if not winner.empty and int(winner.iloc[0]) == 1:
            st.success("The forecast winner won the race.")

        table = compared[["PredictedPosition", "Driver", "Team",
                          "FinishPosition", "AbsoluteError"]].rename(columns={
            "PredictedPosition": "Predicted Position",
            "Team": "Constructor",
            "FinishPosition": "Actual Position",
            "AbsoluteError": "Absolute Error",
        })
        st.dataframe(table, use_container_width=True, hide_index=True)

    metrics = load_form_metrics()
    if metrics:
        with st.expander("How good is this forecast?"):
            st.write(
                f"Trained on {metrics.get('training_period')} and tested on "
                f"{metrics.get('testing_period')}: see the comparison table "
                "printed by form_model.py and in the project report."
            )
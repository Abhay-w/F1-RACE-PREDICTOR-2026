import json
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

from jolpica_data import fetch_qualifying, fetch_results
from forecast_page import render_forecast_page

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "position_model.joblib"
METRICS_PATH = PROJECT_DIR / "model_metrics.json"

SEASON = 2026


st.set_page_config(
    page_title="F1 Race Predictor | 2026",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background: #0b0d12;
    color: #f5f6fa;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: #11141c;
    border-right: 1px solid #272b36;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
h1, h2, h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 0.5px;
}
.hero {
    background: linear-gradient(115deg, #1b202b, #11141b 70%);
    border: 1px solid #303644;
    border-left: 5px solid #e10600;
    border-radius: 15px;
    padding: 28px;
    margin-bottom: 22px;
}
.eyebrow {
    color: #ff625d;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}
.hero-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 46px;
    font-weight: 800;
    line-height: 1.05;
    margin: 8px 0;
}
.hero-subtitle {
    color: #aeb5c3;
    font-size: 14px;
}
.metric-card {
    background: #151923;
    border: 1px solid #2b303d;
    border-radius: 12px;
    padding: 18px;
    min-height: 112px;
}
.metric-label {
    color: #aeb5c3;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 29px;
    font-weight: 800;
    margin-top: 6px;
}
.section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 27px;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 10px;
}
.podium-card {
    background: #151923;
    border: 1px solid #303644;
    border-radius: 12px;
    text-align: center;
    padding: 22px 10px;
    min-height: 150px;
}
.podium-position {
    color: #ff625d;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}
.podium-driver {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 24px;
    font-weight: 800;
    margin: 8px 0 4px;
}
.podium-team {
    color: #aeb5c3;
    font-size: 12px;
}
.stButton > button {
    background: #e10600;
    color: white;
    border: none;
    border-radius: 8px;
    min-height: 44px;
    font-weight: 700;
    width: 100%;
}
.stButton > button:hover {
    background: #ff302a;
    color: white;
}
[data-testid="stDataFrame"] {
    border: 1px solid #2b303d;
    border-radius: 10px;
}
.small-note {
    color: #929aaa;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_schedule(year):
    """Retrieve the race schedule from Jolpica."""

    url = f"{BASE_URL}/{year}.json?limit=100"

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    races = response.json()["MRData"]["RaceTable"]["Races"]

    schedule = []

    for race in races:
        try:
            round_number = int(race["round"])
        except (KeyError, TypeError, ValueError):
            continue

        circuit = race.get("Circuit", {})

        schedule.append({
            "Round": round_number,
            "Race": race.get("raceName", "Unknown race"),
            "Date": race.get("date", "Date unavailable"),
            "Circuit": circuit.get("circuitName", "Unknown circuit"),
            "Location": (
                f"{circuit.get('Location', {}).get('locality', '')}, "
                f"{circuit.get('Location', {}).get('country', '')}"
            ).strip(", "),
        })

    return pd.DataFrame(schedule).sort_values("Round")


@st.cache_resource
def load_model():
    """Load the trained model."""

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics():
    """Load the saved evaluation metrics."""

    if not METRICS_PATH.exists():
        return {}

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def predict_race(qualifying_data, model):
    """
    Predict finishing positions using qualifying position.

    The regression output is used to rank drivers.
    It is not a probability or a confidence score.
    """

    if qualifying_data.empty:
        return pd.DataFrame()

    required_columns = [
        "Driver",
        "Team",
        "QualifyingPosition",
    ]

    if not all(
        column in qualifying_data.columns
        for column in required_columns
    ):
        raise ValueError(
            "Qualifying data is missing required columns."
        )

    predictions = qualifying_data.copy()

    predictions["ModelScore"] = model.predict(
        predictions[["QualifyingPosition"]]
    )

    predictions = predictions.sort_values(
        ["ModelScore", "QualifyingPosition"],
        ascending=[True, True],
    ).reset_index(drop=True)

    predictions["PredictedPosition"] = (
        range(1, len(predictions) + 1)
    )

    return predictions


def make_podium_card(position, label, driver, team):
    """Render a podium card."""

    st.markdown(f"""
    <div class="podium-card">
        <div class="podium-position">{label}</div>
        <div class="podium-driver">{driver}</div>
        <div class="podium-team">{team}</div>
        <div class="podium-team">Predicted P{position}</div>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🏎️ F1 ANALYTICS")
    st.caption("RACE PREDICTOR • 2026")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Race Predictor",
            "Next Race Forecast",
            "Race Calendar",
            "Model Analytics",
            "About Project",
        ],
    )

    st.divider()

    st.markdown("**DATA SOURCE**")
    st.caption("Jolpica-F1 API")
    st.caption("Historical qualifying + race results")

    st.divider()

    st.markdown("**MODEL**")
    st.caption("Gradient Boosting Regressor")

    if MODEL_PATH.exists():
        st.success("Trained model loaded")
    else:
        st.warning("Model file not found")

st.markdown("""
<div class="hero">
    <div class="eyebrow">THE GRID • THE DATA • THE PREDICTION</div>
    <div class="hero-title">
        F1 RACE PREDICTOR
        <span style="color:#e10600">2026</span>
    </div>
    <div class="hero-subtitle">
        A machine-learning project for analysing qualifying
        performance and predicting Formula 1 race finishing positions.
    </div>
</div>
""", unsafe_allow_html=True)

model = load_model()
metrics = load_metrics()

if page == "Race Predictor":

    st.markdown(
        '<div class="section-title">RACE CONTROL</div>',
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(
            "The trained model could not be found. "
            "Please run position_model.py first."
        )
        st.stop()

    try:
        with st.spinner("Loading the 2026 race calendar..."):
            schedule = fetch_schedule(SEASON)

    except (requests.RequestException, ValueError, KeyError) as error:
        st.error(f"Could not load the race calendar: {error}")
        st.stop()

    if schedule.empty:
        st.warning("No 2026 race schedule was returned by the API.")
        st.stop()

    race_options = schedule.to_dict("records")

    selected_race = st.selectbox(
        "SELECT GRAND PRIX",
        options=race_options,
        format_func=lambda race: (
            f"Round {race['Round']} — {race['Race']}"
        ),
    )

    st.caption(
        f"📍 {selected_race['Circuit']} • "
        f"{selected_race['Location']} • "
        f"Scheduled date: {selected_race['Date']}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Season</div>
            <div class="metric-value">2026</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Prediction target</div>
            <div class="metric-value">Finish Position</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">Gradient Boosting</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">PREDICTED PODIUM</div>',
        unsafe_allow_html=True,
    )

    try:
        with st.spinner(
            "Fetching qualifying data and generating predictions..."
        ):
            qualifying = fetch_qualifying(
                SEASON,
                selected_race["Round"],
            )

    except (requests.RequestException, ValueError, KeyError) as error:
        st.error(f"Could not retrieve qualifying data: {error}")
        st.stop()

    if qualifying.empty:
        st.info(
            "Qualifying results are not available from the data source "
            "for this round yet. Select another round with qualifying data."
        )
        st.stop()

    try:
        predictions = predict_race(qualifying, model)

    except (ValueError, KeyError, TypeError) as error:
        st.error(f"Could not generate predictions: {error}")
        st.stop()

    if predictions.empty:
        st.warning("No predictions were generated.")
        st.stop()

    podium_columns = st.columns(3)

    podium_labels = [
        (1, "P1 • RACE WINNER"),
        (2, "P2 • SECOND PLACE"),
        (3, "P3 • THIRD PLACE"),
    ]

    for column, (position, label) in zip(
        podium_columns,
        podium_labels,
    ):
        with column:
            if len(predictions) >= position:
                driver = predictions.iloc[position - 1]

                make_podium_card(
                    position,
                    label,
                    driver["Driver"],
                    driver["Team"],
                )
            else:
                st.info(f"Not enough drivers to display {label}.")

    st.markdown(
        '<div class="section-title">FULL RACE PREDICTIONS</div>',
        unsafe_allow_html=True,
    )

    display_predictions = predictions[
        [
            "PredictedPosition",
            "Driver",
            "Team",
            "QualifyingPosition",
            "ModelScore",
        ]
    ].copy()

    display_predictions.columns = [
        "Predicted Position",
        "Driver",
        "Constructor",
        "Qualifying Position",
        "Regression Score",
    ]

    display_predictions["Regression Score"] = (
        display_predictions["Regression Score"].round(3)
    )

    st.dataframe(
        display_predictions,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The regression score is an internal model output, not a "
        "probability. Predicted positions are rankings of those outputs."
    )

    csv_data = display_predictions.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Predictions CSV",
        data=csv_data,
        file_name=(
            f"f1_2026_round_{selected_race['Round']}_predictions.csv"
        ),
        mime="text/csv",
    )

    # Compare with official results when available
    st.markdown(
        '<div class="section-title">PREDICTION VS ACTUAL</div>',
        unsafe_allow_html=True,
    )

    try:
        actual_results = fetch_results(
            SEASON,
            selected_race["Round"],
        )

    except (requests.RequestException, ValueError, KeyError) as error:
        actual_results = pd.DataFrame()
        st.caption(f"Official results could not be retrieved: {error}")

    if actual_results.empty:
        st.info(
            "Official race results are not available from the data source "
            "for this round yet. Once results are published, this section "
            "can compare predicted and actual finishing positions."
        )

    else:
        comparison = predictions.merge(
            actual_results[
                ["DriverCode", "FinishPosition", "Status"]
            ],
            on="DriverCode",
            how="inner",
        )

        if not comparison.empty:
            comparison = comparison.dropna(
                subset=["FinishPosition"]
            ).copy()

            if not comparison.empty:
                comparison["Position Error"] = (
                    comparison["PredictedPosition"]
                    - comparison["FinishPosition"]
                ).abs()

                race_mae = comparison["Position Error"].mean()

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric(
                        "Race MAE",
                        f"{race_mae:.2f}",
                    )

                with metric_col2:
                    st.metric(
                        "Drivers compared",
                        len(comparison),
                    )

                st.dataframe(
                    comparison[
                        [
                            "PredictedPosition",
                            "Driver",
                            "Team",
                            "FinishPosition",
                            "Position Error",
                            "Status",
                        ]
                    ].rename(columns={
                        "PredictedPosition": "Predicted Position",
                        "Driver": "Driver",
                        "Team": "Constructor",
                        "FinishPosition": "Actual Position",
                        "Position Error": "Absolute Error",
                        "Status": "Race Status",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "This is a post-race comparison for the selected "
                    "round. It is not the same as the held-out 2025 "
                    "validation result."
                )
            else:
                st.info(
                    "Official results were returned, but no comparable "
                    "finishing positions were available."
                )
        else:
            st.info(
                "The available qualifying and race-result records "
                "could not be matched by driver code."
            )

elif page == "Next Race Forecast":

    render_forecast_page(SEASON, fetch_schedule, make_podium_card)

elif page == "Race Calendar":

    st.markdown(
        '<div class="section-title">2026 RACE CALENDAR</div>',
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Loading the official schedule..."):
            schedule = fetch_schedule(SEASON)

        if schedule.empty:
            st.info("No schedule information is currently available.")
        else:
            st.dataframe(
                schedule.rename(columns={
                    "Round": "Round",
                    "Race": "Grand Prix",
                    "Date": "Scheduled Date",
                    "Circuit": "Circuit",
                    "Location": "Location",
                }),
                use_container_width=True,
                hide_index=True,
            )

    except (requests.RequestException, ValueError, KeyError) as error:
        st.error(f"Could not load the race calendar: {error}")


# =========================================================
# MODEL ANALYTICS
# =========================================================

elif page == "Model Analytics":

    st.markdown(
        '<div class="section-title">MODEL PERFORMANCE</div>',
        unsafe_allow_html=True,
    )

    if not metrics:
        st.warning(
            "The evaluation metrics file was not found. "
            "Run position_model.py to generate it."
        )

    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Validation MAE</div>
                <div class="metric-value">
                    {metrics.get('model_mae', float('nan')):.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Baseline MAE</div>
                <div class="metric-value">
                    {metrics.get('baseline_mae', float('nan')):.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Relative MAE improvement</div>
                <div class="metric-value">
                    {metrics.get('relative_improvement_percent', 0):.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Validation setup")

        st.write(
            f"**Algorithm:** {metrics.get('model', 'Unknown')}"
        )
        st.write(
            f"**Training period:** "
            f"{metrics.get('training_period', 'Not specified')}"
        )
        st.write(
            f"**Testing period:** "
            f"{metrics.get('testing_period', 'Not specified')}"
        )
        st.write(
            f"**Training records:** "
            f"{metrics.get('training_records', 'Not recorded')}"
        )
        st.write(
            f"**Testing records:** "
            f"{metrics.get('testing_records', 'Not recorded')}"
        )
        st.write(
            f"**Features:** {', '.join(metrics.get('features', []))}"
        )
        st.write(
            f"**Target:** {metrics.get('target', 'Not specified')}"
        )

        model_mae = metrics.get("model_mae")
        baseline_mae = metrics.get("baseline_mae")

        if model_mae is not None and baseline_mae is not None:
            if model_mae < baseline_mae:
                st.success(
                    "The model achieved a lower MAE than the "
                    "qualifying-position baseline on the 2025 test data."
                )
            elif model_mae > baseline_mae:
                st.warning(
                    "The qualifying-position baseline achieved a "
                    "lower MAE than the model on the 2025 test data."
                )
else: 
    pass
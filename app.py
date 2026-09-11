import streamlit as st
from main import calendar, predict_race

st.set_page_config(
    page_title="F1 Race Predictor 2026",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ F1 Race Predictor 2026")
st.write("Machine Learning based Formula 1 Race Prediction")

st.divider()

st.subheader("🏁 Race Prediction")

try:
    # Load F1 calendar
    schedule = calendar()

    race_options = {
        f"Round {int(row['RoundNumber'])} - {row['EventName']}":
        int(row["RoundNumber"])
        for _, row in schedule.iterrows()
    }

    selected_race = st.selectbox(
        "Select Grand Prix",
        list(race_options.keys())
    )

    if st.button("🔮 Predict Race"):

        round_no = race_options[selected_race]

        race_name = schedule.loc[
            schedule["RoundNumber"] == round_no,
            "EventName"
        ].iloc[0]

        with st.spinner(
            "Fetching F1 data and running the ML model..."
        ):
            result = predict_race(
                round_no,
                race_name,
                schedule,
                compare=False
            )

        st.success("✅ Prediction completed!")

        st.divider()
        st.subheader("🏆 Prediction Result")

        # Display whatever predict_race() returns
        if result is not None:
            if isinstance(result, dict):
                for key, value in result.items():
                    st.write(f"**{key}:** {value}")
            else:
                st.write(result)
        else:
            st.warning(
                "The prediction completed, but predict_race() "
                "did not return the prediction result."
            )

except Exception as e:
    st.error(f"Error: {e}")

st.divider()

st.subheader("🤖 Model Information")

col1, col2 = st.columns(2)

with col1:
    st.write("**Machine Learning Model:** Gradient Boosting")

with col2:
    st.write("**Evaluation Metric:** MAE")
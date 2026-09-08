import streamlit as st

st.set_page_config(
    page_title="F1 Race Predictor 2026",
    page_icon="",
    layout="wide"
)

st.title(" F1 Race Predictor 2026")
st.write("Machine Learning based Formula 1 Race Prediction")

st.divider()

st.subheader(" Race Prediction")

race = st.selectbox(
    "Select Grand Prix",
    [
        "Spanish Grand Prix",
        "Italian Grand Prix",
        "Other"
    ]
)

if st.button(" Predict Race"):
    st.success(f"Prediction generated for {race}!")

st.divider()

st.subheader(" Model Information")

col1, col2 = st.columns(2)

with col1:
    st.write("**Machine Learning Model:** Gradient Boosting")

with col2:
    st.write("**Evaluation Metric:** MAE")
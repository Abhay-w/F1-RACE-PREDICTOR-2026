# 🏎️ F1 Race Predictor 2026

A machine-learning project that predicts Formula 1 race finishing positions, with a Streamlit dashboard for the 2026 season. It uses historical race and qualifying data (2018 onward) from the [Jolpica-F1 API](https://api.jolpi.ca/ergast/f1) and a Gradient Boosting Regressor.

## Features

- **Race Predictor** – enter or load qualifying results and get predicted finishing positions and a podium.
- **Next Race Forecast** – predict a race *before* qualifying using only driver and team form. It can also back-test a completed race by showing the model only the races that came before it.
- **Race Calendar** – the 2026 schedule.
- **Model Analytics** – evaluation metrics and comparison against baselines.
- **About Project** – project background inside the app.

## How it works

Two models are trained to predict `FinishPosition`:

| Model | When it is used | Features |
|---|---|---|
| **Grid + form** (`position_model.py`) | After qualifying | `QualifyingPosition`, `DriverFormFinish`, `TeamFormFinish`, `CircuitFormFinish`, `DriverDnfRate` |
| **Form only** (`form_model.py`) | Before qualifying | `DriverFormFinish`, `TeamFormFinish`, `CircuitFormFinish`, `DriverDnfRate` |

Form features are computed with only past races (everything is shifted by one race, so there is no leakage from the race being predicted):

- **DriverFormFinish** – driver's average finish over the last 5 races
- **TeamFormFinish** – team's average finish over its last 5 races
- **CircuitFormFinish** – driver's average finish at the same circuit
- **DriverDnfRate** – share of DNFs over the last 10 races

Both models are `GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2)`. They are trained on 2018–2024 and tested on the 2025 season; the final saved models are refit on all available data (2018–2026).

## Results (2025 test season, 24 races)

Metrics are computed per race and averaged. Baselines are "predict finish = grid order" and "predict finish = recent driver form".

| Method | Winner accuracy | Top-3 hit rate | Spearman | Rank MAE |
|---|---|---|---|---|
| Grid + form model | 66.7% | 68.1% | 0.650 | 3.38 |
| Grid order baseline | 66.7% | 75.0% | 0.652 | 3.33 |
| Form-only model | 16.7% | 58.3% | 0.544 | 4.17 |
| Driver form baseline | 25.0% | 58.3% | 0.511 | 4.26 |

**Honest takeaway:** once qualifying is known, the model is statistically indistinguishable from simply using the starting grid (paired 95% confidence intervals include zero). Before qualifying, the form-only model beats the driver-form baseline but is clearly worse than the grid. Full numbers, including paired confidence intervals, are in `model_metrics.json` and `form_metrics.json`.

## Getting started

```bash
git clone https://github.com/Abhay-w/F1-RACE-PREDICTOR-2026.git
cd F1-RACE-PREDICTOR-2026

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install requests            # used by the data and app code
```

### Run the app

```bash
streamlit run app.py
```

### Retrain the models

```bash
python position_model.py        # grid + form model -> position_model.joblib, model_metrics.json
python form_model.py            # form-only model   -> form_model.joblib, form_metrics.json
```

Add `--refresh` to either command to re-download data instead of using the cache.

## Project structure

```
├── app.py                 # Streamlit app (main entry point)
├── forecast_page.py       # "Next Race Forecast" page
├── modeling.py            # Training + evaluation pipeline for both models
├── form_features.py       # Rolling form features and feature building for upcoming races
├── evaluation.py          # Per-race metrics and paired confidence intervals
├── jolpica_data.py        # Jolpica API client with retries and caching
├── position_model.py      # Train the grid + form model
├── form_model.py          # Train the form-only model
├── *.joblib, *_metrics.json   # Saved models and their metrics
├── data_cache/            # Cached season results, qualifying and schedule CSVs
├── .jolpica_cache/        # Cached raw API responses
├── test_system.py         # Offline test suite (API calls are mocked)
├── test_form.py           # Tests for form features
└── legacy/                # Earlier FastF1-based version of the project
```

## Data

Race results, qualifying results and the schedule come from the Jolpica-F1 API (a community continuation of the Ergast API). Responses are cached locally; the current season's cache refreshes every 6 hours.

## Testing

The tests mock all API calls, so they run offline.

```bash
pip install pytest
python -m pytest -v
```

## Limitations

- Qualifying position is by far the strongest predictor; the extra form features add little on top of it.
- Only past results are used. Weather, tyre strategy, safety cars, penalties and regulation changes are not modeled.
- Predictions are point estimates of finishing position, not probabilities.

## Tech stack

Python · pandas · NumPy · scikit-learn · Streamlit · requests · joblib

## Disclaimer

This is an educational project and is not affiliated with Formula 1, the FIA or any team. Do not use it for betting.
# F1 Race Predictor 2026

A machine learning model that predicts F1 race pace using qualifying
data and historical race results, built with the FastF1 API.

## Credit

This project is adapted from the structure and approach of
[`mar-antaya/2025_f1_predictions`](https://github.com/mar-antaya/2025_f1_predictions)
(MIT License). The original used a single hardcoded qualifying-time
feature per race. This version rebuilds it for the 2026 season with
live data fetching and additional feature engineering (see below).

## What's different from the original

| Original                        # F1 Race Predictor 2026

A machine learning model that predicts F1 race pace using qualifying
data and historical race results, built with the FastF1 API.

## Credit

This project is adapted from the structure and approach of
[`mar-antaya/2025_f1_predictions`](https://github.com/mar-antaya/2025_f1_predictions)
(MIT License). The original used a single hardcoded qualifying-time
feature per race. This version rebuilds it for the 2026 season with
live data fetching and additional feature engineering (see below).

## What's different from the original

| Original | This version |
|---|---|
| One `.py` file per race, hardcoded qualifying times | Modular files, qualifying data pulled live via FastF1 |
| Manually typed 2025 driver list | Updated 2026 grid (Audi, Cadillac, all driver moves) |
| Single feature: qualifying time | Two features: qualifying time + team average pace |
| No team information | Driver-to-team mapping used for feature engineering |
| Trained on "same race, previous year" | Trained on **this season's average pace across all rounds run so far** |

### Why the training approach changed

The original always trained on the exact same Grand Prix from the
previous season (e.g. 2024 Chinese GP → predict 2025 Chinese GP).
That breaks for a race with no history — **Madrid is brand new to the
2026 calendar**, so there is no "2025 Madrid GP" to learn from.

Instead, this version trains on each driver's **average race pace
across every 2026 round completed so far**. This has two benefits:
it works for new circuits, and it reflects each driver's *current*
form rather than a single race from over a year ago.

## Project structure

```
f1_predictor_2026/
├── drivers_2026.py   # 2026 driver codes and team mapping
├── fetch_data.py      # FastF1 API calls (race laps, qualifying)
├── features.py         # Feature engineering (team pace features)
├── train_model.py      # Model training, prediction, evaluation
├── main.py              # Runs the full pipeline
└── README.md
```

## How it works

1. **Training data**: pulls average race lap time per driver from the
   equivalent Grand Prix in the previous season.
2. **Prediction input**: pulls each driver's best qualifying lap for
   the target 2026 race.
3. **Feature engineering**: adds a `TeamAvgTime` feature — the average
   qualifying pace of both teammates — so the model captures car
   performance, not just one driver's single lap.
4. **Model**: Gradient Boosting Regressor (`scikit-learn`), same
   family as the original project.
5. **Evaluation**: Mean Absolute Error (MAE) on a held-out test split.

## Usage

```bash
pip install fastf1 pandas numpy scikit-learn
python main.py
```

To predict a different Grand Prix, edit the config block at the top
of `main.py`:

```python
RACE_NAME = "Spanish GP (Madrid)"
TRAIN_YEAR = 2026
TRAIN_ROUNDS = list(range(1, 16))   # all rounds run so far
PREDICT_YEAR, PREDICT_ROUND = 2026, 16
```

`ROUND` is the numeric position of the race on that season's calendar
(1 = first race, 2 = second, etc.) — check the FastF1 or F1 calendar
for the exact round number of the race you want. `TRAIN_ROUNDS`
should cover every round already completed at the time you run the
script.

## Notes / limitations

- Requires the target qualifying session to have already happened
  (FastF1 can't fetch a session that hasn't run yet).
- Driver lineup in `drivers_2026.py` reflects the season-opening grid;
  update it if a mid-season driver change occurs.
- MAE will vary race to race — a chaotic race (safety cars, rain,
  incidents) is harder to predict than a clean one.

## Future improvements

- Add weather data as a feature
- Incorporate pit stop strategy
- Track-specific historical performance per driver
- Try LightGBM/XGBoost and compare against Gradient Boosting

## License

MIT — same as the original project this is adapted from.| This version |
|---|---|
| One `.py` file per race, hardcoded qualifying times | Modular files, qualifying data pulled live via FastF1 |
| Manually typed 2025 driver list | Updated 2026 grid (Audi, Cadillac, all driver moves) |
| Single feature: qualifying time | Two features: qualifying time + team average pace |
| No team information | Driver-to-team mapping used for feature engineering |
| Trained on "same race, previous year" | Trained on **this season's average pace across all rounds run so far** |

### Why the training approach changed

The original always trained on the exact same Grand Prix from the
previous season (e.g. 2024 Chinese GP → predict 2025 Chinese GP).
That breaks for a race with no history — **Madrid is brand new to the
2026 calendar**, so there is no "2025 Madrid GP" to learn from.

Instead, this version trains on each driver's **average race pace
across every 2026 round completed so far**. This has two benefits:
it works for new circuits, and it reflects each driver's *current*
form rather than a single race from over a year ago.

## Project structure

```
f1_predictor_2026/
├── drivers_2026.py   # 2026 driver codes and team mapping
├── fetch_data.py      # FastF1 API calls (race laps, qualifying)
├── features.py         # Feature engineering (team pace features)
├── train_model.py      # Model training, prediction, evaluation
├── main.py              # Runs the full pipeline
└── README.md
```

## How it works

1. **Training data**: pulls average race lap time per driver from the
   equivalent Grand Prix in the previous season.
2. **Prediction input**: pulls each driver's best qualifying lap for
   the target 2026 race.
3. **Feature engineering**: adds a `TeamAvgTime` feature — the average
   qualifying pace of both teammates — so the model captures car
   performance, not just one driver's single lap.
4. **Model**: Gradient Boosting Regressor (`scikit-learn`), same
   family as the original project.
5. **Evaluation**: Mean Absolute Error (MAE) on a held-out test split.

## Usage

```bash
pip install fastf1 pandas numpy scikit-learn
python main.py
```

To predict a different Grand Prix, edit the config block at the top
of `main.py`:

```python
RACE_NAME = "Spanish GP (Madrid)"
TRAIN_YEAR = 2026
TRAIN_ROUNDS = list(range(1, 16))   # all rounds run so far
PREDICT_YEAR, PREDICT_ROUND = 2026, 16
```

`ROUND` is the numeric position of the race on that season's calendar
(1 = first race, 2 = second, etc.) — check the FastF1 or F1 calendar
for the exact round number of the race you want. `TRAIN_ROUNDS`
should cover every round already completed at the time you run the
script.

## Notes / limitations

- Requires the target qualifying session to have already happened
  (FastF1 can't fetch a session that hasn't run yet).
- Driver lineup in `drivers_2026.py` reflects the season-opening grid;
  update it if a mid-season driver change occurs.
- MAE will vary race to race — a chaotic race (safety cars, rain,
  incidents) is harder to predict than a clean one.

## Future improvements

- Add weather data as a feature
- Incorporate pit stop strategy
- Track-specific historical performance per driver
- Try LightGBM/XGBoost and compare against Gradient Boosting

## License

MIT — same as the original project this is adapted from.
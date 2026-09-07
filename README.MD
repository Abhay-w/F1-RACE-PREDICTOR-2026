 F1 Race Predictor 2026

A machine learning project that predicts Formula 1 race pace using qualifying performance, team pace, and historical 2026 race data.

The project uses the FastF1 API to automatically retrieve Formula 1 session data and a Gradient Boosting Regressor from Scikit-learn to generate race predictions.

---

 Features

-  Automatically detects the next upcoming 2026 race
-  Predicts race performance using qualifying data
-  Analyzes completed races
-  Uses a Gradient Boosting Regression model
-  Uses driver and team information for feature engineering
-  Calculates Mean Absolute Error (MAE)
-  Fetches Formula 1 data automatically using FastF1
-  Uses previous completed 2026 races as training data
-  Works with new circuits that do not have previous-season data

---

 How It Works

The project follows this pipeline:

1. Find the Race

The program reads the 2026 Formula 1 calendar through FastF1.

It provides two modes:

1.  Predict next upcoming race
2.  Analyze completed race

2. Collect Training Data

For the selected race, the model uses previously completed 2026 races as training data.

Average race lap pace is calculated for each driver.

3. Collect Qualifying Data

For the target race, the driver's best qualifying lap is retrieved using FastF1.

Qualifying performance is used as an important prediction feature.

If qualifying has not happened yet, the program correctly waits instead of generating a prediction.

4. Feature Engineering

The model uses:

- Driver qualifying time
- Team average qualifying pace

Team information helps the model account for the performance of the car as well as the individual driver.

5. Machine Learning

The project uses:

Gradient Boosting Regressor

from "scikit-learn".

The model learns the relationship between qualifying/team pace and race pace.

6. Prediction

The model predicts race pace for each driver and ranks them from fastest to slowest.

The program displays:

- Predicted position
- Driver
- Team
- Predicted race time
- Predicted winner
- Model MAE

7. Completed Race Analysis

For completed races, the program can compare predicted performance with the available race data.

This allows the prediction system to be tested against races that have already happened.

---

 Project Structure

f1_predictor_2026/
│
├── drivers_2026.py
│   └── 2026 driver codes and team mapping
│
├── fetch_data.py
│   └── FastF1 data collection
│
├── features.py
│   └── Feature engineering
│
├── train_model.py
│   └── Model training, prediction and MAE evaluation
│
├── main.py
│   └── Main program and race selection
│
└── README.md

---

 Technologies Used

- Python
- FastF1
- Pandas
- NumPy
- Scikit-learn
- Gradient Boosting Regression
- Machine Learning
- Feature Engineering

---

 Installation

Install the required Python libraries:

pip install fastf1 pandas numpy scikit-learn

Then run:

python main.py

---

 Usage

After running the program:

 F1 RACE PREDICTOR 2026

1.  Predict next upcoming race
2.  Analyze completed race

Enter 1 or 2:

Option 1 — Upcoming Race

The program automatically finds the next race on the 2026 calendar.

If qualifying has already taken place, the model generates a prediction.

If qualifying has not happened yet, the program informs the user and waits for the qualifying session.

Option 2 — Completed Race

The program displays completed 2026 races.

The user selects a round number, and the model generates a prediction and comparison using the completed race data.

---

 Example Output

 PREDICTED RACE RESULT

 Predicted Position   Driver              Team
 1                    Driver A            Team A
 2                    Driver B            Team B
 3                    Driver C            Team C
 ...

 Predicted Winner: Driver A
Team: Team A
Model MAE: 1.25 seconds

For completed races, the program can also display predicted versus actual race performance.

---

 Why This Project?

Traditional approaches may rely on manually entered data or previous-season results for the same Grand Prix.

That approach becomes difficult when predicting a new circuit.

This project instead uses the driver's current-season performance from completed races.

This allows the model to work even when a circuit does not have comparable previous-season data.

---

 Limitations

The current model predicts race pace, not the exact official finishing order.

Actual Formula 1 races are affected by many factors that are not currently included, such as:

- Weather
- Safety Cars
- Red Flags
- Pit-stop strategy
- Tire strategy
- Mechanical failures
- Crashes
- Track-specific performance
- Race incidents

Therefore, the prediction should be considered a machine learning estimate, not a guaranteed race result.

---

 Future Improvements

Possible improvements include:

-  Weather data
-  Tire compound and degradation
-  Pit-stop strategy
-  Track-specific driver performance
-  More historical features
-  XGBoost / LightGBM comparison
-  More advanced ensemble models
-  Driver form and recent-performance features

---

 Credit

This project is adapted from the structure and approach of:

"mar-antaya/2025_f1_predictions"

The original project is licensed under the MIT License.

This version was rebuilt and extended for the 2026 Formula 1 season, including live FastF1 data fetching, modular architecture, team-based feature engineering, automatic race detection, and completed-race analysis.

---

 License

MIT License — based on the licensing of the original project.
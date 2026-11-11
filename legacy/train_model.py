
"""
train_model.py
Trains the Gradient Boosting model on merged qualifying + race data,
and reports evaluation metrics...
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

FEATURE_COLUMNS = ["QualifyingTime (s)", "TeamAvgTime (s)"]

def train(merged_df: pd.DataFrame, random_state: int = 39):
    """
    merged_df must contain FEATURE_COLUMNS plus a 'LapTime (s)' target
    column (the actual race pace to predict)
    """
    X = merged_df[FEATURE_COLUMNS]
    y = merged_df["LapTime (s)"]

    if X.shape[0] == 0:
        raise ValueError("Dataset is empty after preprocessing. check data source!")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return model, mae

def predict(model, feature_df: pd.DataFrame) -> pd.Series:
    return model.predict(feature_df[FEATURE_COLUMNS])

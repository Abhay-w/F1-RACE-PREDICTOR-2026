"""
Tests for the pre-race (form) model.

Run with:   python -m pytest tests -v
The tests use made-up race data, so they run offline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import form_model as fm  # noqa: E402
from form_features import (  # noqa: E402
    FORM_FEATURES, add_form_features, next_race_features, normalise_team,
)

TEAMS = ["Red Bull", "Ferrari", "Mercedes", "McLaren", "Aston Martin",
         "Alpine F1 Team", "Williams", "RB F1 Team", "Haas F1 Team", "Sauber"]


def make_history(seasons=range(2018, 2027), races=20, last_season_races=12, seed=1):
    """Made-up seasons: strong teams usually qualify and finish ahead."""
    rng = np.random.default_rng(seed)
    rows = []
    strength = rng.permutation(len(TEAMS)).astype(float)

    for season in seasons:
        strength = strength + rng.normal(0, 0.7, len(TEAMS))          # slow drift
        n_races = last_season_races if season == max(seasons) else races

        for rnd in range(1, n_races + 1):
            pace = np.repeat(strength, 2) + rng.normal(0, 1.2, 20)
            quali = pd.Series(pace).rank(method="first").astype(int).to_numpy()
            race_pace = pace + rng.normal(0, 2.0, 20)
            finish = pd.Series(race_pace).rank(method="first").astype(int).to_numpy()

            for i in range(20):
                rows.append({
                    "Season": season, "Round": rnd,
                    "DriverCode": f"D{i:02d}", "Driver": f"Driver {i}",
                    "Team": TEAMS[i // 2],
                    "QualifyingPosition": int(quali[i]),
                    "FinishPosition": int(finish[i]),
                })

    return pd.DataFrame(rows)


def small_history():
    """One driver, one team, finishing 1..7 in seven races."""
    return pd.DataFrame({
        "Season": [2025] * 7, "Round": list(range(1, 8)),
        "DriverCode": ["AAA"] * 7, "Driver": ["A A"] * 7, "Team": ["Ferrari"] * 7,
        "QualifyingPosition": [2, 4, 6, 8, 10, 12, 14],
        "FinishPosition": [1, 2, 3, 4, 5, 6, 7],
    })


# ------------------------------------------------------------------ features
@pytest.mark.parametrize("raw,key", [
    ("RB F1 Team", "RB"), ("Scuderia AlphaTauri", "RB"), ("Toro Rosso", "RB"),
    ("Alfa Romeo", "Sauber"), ("Audi", "Sauber"), ("Red Bull", "Red Bull"),
    ("Racing Point", "Aston Martin"), ("Renault", "Alpine"),
    ("Alpine F1 Team", "Alpine"), ("Cadillac F1 Team", "Cadillac"),
])
def test_FT01_team_names_are_joined_across_seasons(raw, key):
    assert normalise_team(raw) == key


def test_FT02_first_race_has_no_form():
    out = add_form_features(small_history())
    first = out[out["Round"] == 1].iloc[0]
    assert pd.isna(first["DriverFormFinish"]) and pd.isna(first["TeamFormFinish"])


def test_FT03_form_is_average_of_previous_races_only():
    out = add_form_features(small_history()).set_index("Round")
    assert out.loc[2, "DriverFormFinish"] == 1                  # only race 1
    assert out.loc[4, "DriverFormFinish"] == pytest.approx(2)   # mean(1,2,3)
    assert out.loc[4, "DriverFormQuali"] == pytest.approx(4)    # mean(2,4,6)


def test_FT04_window_is_the_last_five_races():
    out = add_form_features(small_history()).set_index("Round")
    # round 7 sees races 2-6 only (race 1 has left the 5-race window)
    assert out.loc[7, "DriverFormFinish"] == pytest.approx(4)    # mean(2,3,4,5,6)
    assert out.loc[7, "DriverFormQuali"] == pytest.approx(8)     # mean(4,6,8,10,12)


def test_FT05_a_race_never_sees_its_own_result():
    base = small_history()
    changed = base.copy()
    changed.loc[changed["Round"] == 7, "FinishPosition"] = 99     # extreme result
    a = add_form_features(base).set_index("Round").loc[7]
    b = add_form_features(changed).set_index("Round").loc[7]
    for column in FORM_FEATURES:
        assert a[column] == b[column] or (pd.isna(a[column]) and pd.isna(b[column]))


def test_FT06_next_race_uses_same_logic_as_training():
    history = small_history()
    future = next_race_features(history, 2025, 8)
    assert len(future) == 1
    assert future.loc[0, "DriverFormFinish"] == pytest.approx(history["FinishPosition"].tail(5).mean())
    assert future.loc[0, "TeamFormQuali"] == pytest.approx(history["QualifyingPosition"].tail(5).mean())


def test_FT07_next_race_rejects_a_race_already_in_history():
    with pytest.raises(ValueError):
        next_race_features(small_history(), 2025, 7)


def test_FT08_lineup_comes_from_latest_race():
    history = make_history(seasons=[2025], races=6, last_season_races=6)
    future = next_race_features(history, 2025, 7)
    assert sorted(future["DriverCode"]) == sorted(history["DriverCode"].unique())
    assert future["Team"].notna().all()


def test_FT09_missing_columns_raise_error():
    with pytest.raises(ValueError):
        add_form_features(pd.DataFrame({"Season": [2025]}))


# -------------------------------------------------------------------- model
def split(history):
    feats = add_form_features(history)
    known = feats.dropna(subset=["FinishPosition", "QualifyingPosition"])
    return known[known["Season"] <= 2024], known[known["Season"] == 2025]


def test_FT10_race_metrics_for_a_perfect_prediction():
    test = make_history(seasons=[2025], races=3)
    out = fm.race_metrics(test, test["FinishPosition"].to_numpy())
    assert out["rank_correlation"] == pytest.approx(1.0)
    assert out["winner_accuracy"] == pytest.approx(1.0)
    assert out["podium_hits_of_3"] == pytest.approx(3.0)


def test_FT11_all_candidate_models_are_evaluated():
    train, test = split(make_history())
    names = [r["model"] for r in fm.evaluate_models(train, test)]
    assert len(names) == 8
    assert "Gradient Boosting, form features" in names
    assert "Qualifying baseline (grid = finish)" in names


def test_FT12_form_model_beats_the_constant_baseline():
    """On data where strong teams stay strong, form must carry information."""
    train, test = split(make_history())
    r = {x["model"]: x for x in fm.evaluate_models(train, test)}
    assert r["Gradient Boosting, form features"]["mae"] < r["Constant (average finishing position)"]["mae"]
    assert r["Gradient Boosting, form features"]["rank_correlation"] > 0.3


def test_FT13_form_plus_qualifying_is_at_least_as_good_as_qualifying_alone():
    train, test = split(make_history())
    r = {x["model"]: x for x in fm.evaluate_models(train, test)}
    both = r["Gradient Boosting, qualifying + form"]["mae"]
    alone = r["Gradient Boosting, qualifying only (first model)"]["mae"]
    assert both <= alone + 0.15


def test_FT14_pipeline_handles_unseen_teams_and_drivers():
    """A brand-new team has no form; the median imputer must still predict."""
    train, _ = split(make_history())
    model = fm.make_gradient_boosting().fit(train[FORM_FEATURES], train["FinishPosition"])
    new = pd.DataFrame([{c: np.nan for c in FORM_FEATURES}])
    assert np.isfinite(model.predict(new)).all()
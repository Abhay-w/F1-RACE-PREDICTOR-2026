"""
Test suite for F1 Race Predictor 2026.

Run from the project folder with:   python -m pytest tests -v

All API calls are replaced by fake responses, so the tests run offline and
give the same result every time.
"""

import ast
import json
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# If Streamlit is not installed, use a tiny stand-in so the data module imports.
try:
    import streamlit  # noqa: F401
except ImportError:
    sys.modules["streamlit"] = types.SimpleNamespace(
        cache_data=lambda **kw: (lambda f: f)
    )

import jolpica_data as jd  # noqa: E402
import position_model as pm  # noqa: E402


# ---------------------------------------------------------------- helpers
class FakeResponse:
    def __init__(self, status=200, data=None, headers=None):
        self.status_code = status
        self._data = data
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


def make_session(responses):
    """Return an object whose get() hands out the given responses in order."""
    queue = list(responses)
    calls = []

    def get(url, timeout=None):
        calls.append(url)
        return queue.pop(0)

    return types.SimpleNamespace(get=get, calls=calls)


def api_payload(races):
    return {"MRData": {"RaceTable": {"Races": races}}}


def clear_cache(func):
    if hasattr(func, "clear"):
        func.clear()


@pytest.fixture(autouse=True)
def fast_and_isolated(monkeypatch, tmp_path):
    """No waiting, and a private cache folder for every test."""
    monkeypatch.setattr(jd.time, "sleep", lambda s: None)
    monkeypatch.setattr(jd, "REQUEST_DELAY", 0)
    monkeypatch.setattr(jd, "CACHE_DIR", tmp_path / "cache")
    clear_cache(jd.fetch_qualifying)
    clear_cache(jd.fetch_results)


def qualifying_race(year, rnd, rows):
    return api_payload([{
        "season": str(year), "round": str(rnd),
        "QualifyingResults": [{
            "position": str(pos),
            "Driver": {"code": code, "givenName": code, "familyName": "X"},
            "Constructor": {"name": team},
        } for pos, code, team in rows],
    }])


def results_race(year, rnd, rows):
    return api_payload([{
        "season": str(year), "round": str(rnd),
        "Results": [{
            "position": str(pos),
            "status": status,
            "Driver": {"code": code, "givenName": code, "familyName": "X"},
            "Constructor": {"name": team},
        } for pos, code, team, status in rows],
    }])


# ------------------------------------------------- data layer (jolpica_data)
def test_TC01_retry_after_rate_limit(monkeypatch):
    """HTTP 429 then HTTP 503 are retried; the third response succeeds."""
    session = make_session([
        FakeResponse(429, headers={"Retry-After": "0"}),
        FakeResponse(503),
        FakeResponse(200, data=api_payload([1, 2, 3])),
    ])
    monkeypatch.setattr(jd, "_session", session)
    assert jd._season_round_count(2018) == 3
    assert len(session.calls) == 3


def test_TC02_gives_up_after_max_attempts(monkeypatch):
    """Permanent failure raises an error instead of returning silently."""
    session = make_session([FakeResponse(429) for _ in range(jd.MAX_ATTEMPTS)])
    monkeypatch.setattr(jd, "_session", session)
    with pytest.raises(requests.RequestException):
        jd._get_json("2018/1/results.json", cacheable=False)


def test_TC03_completed_season_is_cached(monkeypatch):
    """The second request for the same race is read from disk, not the API."""
    session = make_session([FakeResponse(200, data=api_payload([1]))])
    monkeypatch.setattr(jd, "_session", session)
    jd._get_json("2019.json", cacheable=True)
    jd._get_json("2019.json", cacheable=True)  # would crash if it hit the API
    assert len(session.calls) == 1


def test_TC04_qualifying_parsed(monkeypatch):
    payload = qualifying_race(2018, 1, [(1, "HAM", "Mercedes"), (2, "VET", "Ferrari")])
    monkeypatch.setattr(jd, "_session", make_session([FakeResponse(200, data=payload)]))
    df = jd.fetch_qualifying(2018, 1)
    assert list(df["DriverCode"]) == ["HAM", "VET"]
    assert list(df["QualifyingPosition"]) == [1, 2]
    assert set(df["Season"]) == {2018}


def test_TC05_results_non_numeric_position_becomes_missing(monkeypatch):
    payload = results_race(2018, 2, [(1, "HAM", "Mercedes", "Finished"),
                                     ("R", "VET", "Ferrari", "Retired")])
    monkeypatch.setattr(jd, "_session", make_session([FakeResponse(200, data=payload)]))
    df = jd.fetch_results(2018, 2)
    assert df.loc[df.DriverCode == "HAM", "FinishPosition"].iloc[0] == 1
    assert pd.isna(df.loc[df.DriverCode == "VET", "FinishPosition"].iloc[0])


def test_TC06_future_race_returns_empty_table(monkeypatch):
    monkeypatch.setattr(jd, "_session",
                        make_session([FakeResponse(200, data=api_payload([]))]))
    assert jd.fetch_qualifying(2018, 3).empty


def test_TC07_merge_joins_on_season_round_driver(monkeypatch):
    q = qualifying_race(2018, 4, [(1, "HAM", "Mercedes"), (2, "VET", "Ferrari")])
    r = results_race(2018, 4, [(2, "HAM", "Mercedes", "Finished"),
                               (1, "VET", "Ferrari", "Finished")])
    monkeypatch.setattr(jd, "_session",
                        make_session([FakeResponse(200, data=q), FakeResponse(200, data=r)]))
    merged = jd._load_round(2018, 4)
    row = merged.set_index("DriverCode")
    assert row.loc["HAM", "QualifyingPosition"] == 1 and row.loc["HAM", "FinishPosition"] == 2
    assert row.loc["VET", "QualifyingPosition"] == 2 and row.loc["VET", "FinishPosition"] == 1


def test_TC08_driver_without_finish_position_is_dropped(monkeypatch):
    q = qualifying_race(2018, 5, [(1, "HAM", "Mercedes"), (2, "VET", "Ferrari")])
    r = results_race(2018, 5, [(1, "HAM", "Mercedes", "Finished"),
                               ("R", "VET", "Ferrari", "Retired")])
    monkeypatch.setattr(jd, "_session",
                        make_session([FakeResponse(200, data=q), FakeResponse(200, data=r)]))
    merged = jd._load_round(2018, 5)
    assert list(merged["DriverCode"]) == ["HAM"]


def test_TC09_failed_race_is_reported_not_hidden(monkeypatch, capsys):
    """A season where one race keeps failing prints a WARNING line."""
    def fake_count(year):
        return 2

    def fake_load(year, rnd):
        if rnd == 2:
            raise requests.RequestException("boom")
        return pd.DataFrame({"Season": [year], "Round": [rnd], "DriverCode": ["HAM"],
                             "QualifyingPosition": [1], "FinishPosition": [1]})

    monkeypatch.setattr(jd, "_season_round_count", fake_count)
    monkeypatch.setattr(jd, "_load_round", fake_load)
    data = jd.fetch_race_training_data(2018, 2018)
    out = capsys.readouterr().out
    assert len(data) == 1
    assert "1/2 races loaded" in out and "WARNING" in out and "[2]" in out


# ---------------------------------------------- training module (position_model)
def test_TC10_model_settings():
    model = pm.create_model()
    assert model.n_estimators == 100
    assert model.learning_rate == 0.05
    assert model.max_depth == 2
    assert model.random_state == 42


def test_TC11_clean_data_rejects_missing_columns():
    with pytest.raises(ValueError):
        pm.clean_data(pd.DataFrame({"QualifyingPosition": [1, 2]}))


def test_TC12_clean_data_converts_text_numbers():
    raw = pd.DataFrame({"QualifyingPosition": ["1", "2", "x"],
                        "FinishPosition": ["2", "1", "3"],
                        "Season": [2018, 2018, 2018]})
    clean = pm.clean_data(raw)
    assert len(clean) == 2
    assert clean["QualifyingPosition"].dtype.kind in "if"


def test_TC13_model_learns_from_qualifying_position():
    """On synthetic data where drivers finish near where they qualify,
    the model predicts better positions for better grid slots."""
    rng = np.random.default_rng(0)
    q = np.tile(np.arange(1, 21), 40)
    f = np.clip(q + rng.integers(-3, 4, size=q.size), 1, 20)
    model = pm.create_model().fit(pd.DataFrame({"QualifyingPosition": q}), f)
    pole, last = model.predict(pd.DataFrame({"QualifyingPosition": [1, 20]}))
    assert pole < last


# ------------------------------------------------ saved model and predict_race
def load_predict_race():
    """Take predict_race() out of app.py without starting the Streamlit page."""
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    func = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "predict_race")
    module = ast.Module(body=[func], type_ignores=[])
    namespace = {"pd": pd}
    exec(compile(module, "app.py", "exec"), namespace)
    return namespace["predict_race"]


def make_grid(n=20):
    return pd.DataFrame({
        "Driver": [f"Driver {i}" for i in range(1, n + 1)],
        "Team": [f"Team {(i + 1) // 2}" for i in range(1, n + 1)],
        "QualifyingPosition": list(range(1, n + 1)),
    }).sample(frac=1, random_state=1)  # shuffled input order


def test_TC14_saved_model_loads_and_predicts():
    model = joblib.load(ROOT / "position_model.joblib")
    out = model.predict(pd.DataFrame({"QualifyingPosition": [1, 10, 20]}))
    assert len(out) == 3 and np.isfinite(out).all()
    assert out[0] < out[2]  # pole is predicted to finish ahead of last place


def test_TC15_predicted_positions_are_one_to_n():
    predict_race = load_predict_race()
    model = joblib.load(ROOT / "position_model.joblib")
    result = predict_race(make_grid(20), model)
    assert sorted(result["PredictedPosition"]) == list(range(1, 21))


def test_TC16_ranking_follows_model_score():
    predict_race = load_predict_race()
    model = joblib.load(ROOT / "position_model.joblib")
    result = predict_race(make_grid(20), model)
    assert result["ModelScore"].is_monotonic_increasing


def test_TC17_ties_are_broken_by_qualifying_position():
    predict_race = load_predict_race()

    class Flat:                       # gives every driver the same score
        def predict(self, X):
            return np.full(len(X), 5.0)

    result = predict_race(make_grid(6), Flat())
    assert list(result["QualifyingPosition"]) == [1, 2, 3, 4, 5, 6]


def test_TC18_empty_qualifying_gives_empty_result():
    predict_race = load_predict_race()
    model = joblib.load(ROOT / "position_model.joblib")
    assert predict_race(pd.DataFrame(), model).empty


def test_TC19_missing_columns_raise_error():
    predict_race = load_predict_race()
    model = joblib.load(ROOT / "position_model.joblib")
    with pytest.raises(ValueError):
        predict_race(pd.DataFrame({"Driver": ["A"]}), model)


def test_TC20_metrics_file_is_consistent():
    m = json.loads((ROOT / "model_metrics.json").read_text())
    expected = (m["baseline_mae"] - m["model_mae"]) / m["baseline_mae"] * 100
    assert abs(expected - m["relative_improvement_percent"]) < 0.01
    assert m["features"] == ["QualifyingPosition"] and m["target"] == "FinishPosition"
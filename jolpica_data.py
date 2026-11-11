import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://api.jolpi.ca/ergast/f1"
CACHE_DIR = Path(__file__).resolve().parent / "data_cache"

SLEEP_SECONDS = 0.5
PAGE_LIMIT = 100
CURRENT_SEASON_TTL = 6 * 3600

RESULT_COLS = ["Season", "Round", "RaceName", "CircuitId", "DriverCode",
               "Driver", "Team", "Grid", "FinishPosition", "Status"]
QUALI_COLS = ["Season", "Round", "DriverCode", "Driver", "Team",
              "QualifyingPosition"]
SCHEDULE_COLS = ["Round", "Race", "Date", "CircuitId", "Circuit", "Location"]

_session = None


def _get_session():
    global _session
    if _session is None:
        retry = Retry(
            total=6, connect=6, read=6, backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",), respect_retry_after_header=True,
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.headers["User-Agent"] = "f1-race-predictor/1.0"
        _session = session
    return _session


def get_json(path, params=None):
    time.sleep(SLEEP_SECONDS)
    response = _get_session().get(f"{BASE_URL}/{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()["MRData"]


def _paginate(path):
    races, offset = [], 0
    while True:
        data = get_json(path, {"limit": PAGE_LIMIT, "offset": offset})
        races.extend(data["RaceTable"]["Races"])
        offset += PAGE_LIMIT
        if offset >= int(data["total"]):
            return races


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _driver_fields(item):
    driver, team = item["Driver"], item["Constructor"]
    code = driver.get("code") or driver.get("driverId", "")[:3].upper()
    name = f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip()
    return code, name, team.get("name", "")


def _parse_results(season, races):
    rows = []
    for race in races:
        for item in race.get("Results", []):
            code, name, team = _driver_fields(item)
            rows.append({
                "Season": season, "Round": int(race["round"]),
                "RaceName": race.get("raceName", ""),
                "CircuitId": race.get("Circuit", {}).get("circuitId", ""),
                "DriverCode": code, "Driver": name, "Team": team,
                "Grid": _int(item.get("grid")),
                "FinishPosition": _int(item.get("position")),
                "Status": item.get("status", ""),
            })
    return pd.DataFrame(rows, columns=RESULT_COLS)


def _parse_quali(season, races):
    rows = []
    for race in races:
        for item in race.get("QualifyingResults", []):
            code, name, team = _driver_fields(item)
            rows.append({
                "Season": season, "Round": int(race["round"]),
                "DriverCode": code, "Driver": name, "Team": team,
                "QualifyingPosition": _int(item.get("position")),
            })
    return pd.DataFrame(rows, columns=QUALI_COLS)


def _is_fresh(path, season):
    if not path.exists():
        return False
    modified = path.stat().st_mtime
    if season < dt.date.today().year:
        return dt.date.fromtimestamp(modified).year > season
    return time.time() - modified < CURRENT_SEASON_TTL


def _cached(name, season, fetch, columns, refresh=False):
    path = CACHE_DIR / f"{name}_{season}.csv"
    if not refresh and _is_fresh(path, season):
        return pd.read_csv(path)
    df = fetch()
    if df.empty:
        return pd.DataFrame(columns=columns)
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df


def season_results(season, refresh=False):
    return _cached("results", season,
                   lambda: _parse_results(season, _paginate(f"{season}/results.json")),
                   RESULT_COLS, refresh)


def season_qualifying(season, refresh=False):
    return _cached("qualifying", season,
                   lambda: _parse_quali(season, _paginate(f"{season}/qualifying.json")),
                   QUALI_COLS, refresh)


def fetch_schedule(year, refresh=False):
    def fetch():
        rows = []
        for race in get_json(f"{year}.json", {"limit": 100})["RaceTable"]["Races"]:
            circuit = race.get("Circuit", {})
            place = circuit.get("Location", {})
            rows.append({
                "Round": int(race["round"]), "Race": race.get("raceName", ""),
                "Date": race.get("date", ""),
                "CircuitId": circuit.get("circuitId", ""),
                "Circuit": circuit.get("circuitName", ""),
                "Location": f"{place.get('locality', '')}, {place.get('country', '')}".strip(", "),
            })
        return pd.DataFrame(rows, columns=SCHEDULE_COLS)
    return _cached("schedule", year, fetch, SCHEDULE_COLS, refresh).sort_values("Round")


def fetch_qualifying(year, round_no):
    q = season_qualifying(year)
    return q[q["Round"] == round_no].reset_index(drop=True)


def fetch_results(year, round_no):
    r = season_results(year)
    return r[r["Round"] == round_no].reset_index(drop=True)


def fetch_race_training_data(start_year=2018, end_year=2025, refresh=False):
    frames = []
    for year in range(start_year, end_year + 1):
        results, quali = season_results(year, refresh), season_qualifying(year, refresh)
        if results.empty or quali.empty:
            print(f"[jolpica_data] WARNING: no data for {year}")
            continue
        frames.append(results.merge(
            quali[["Season", "Round", "DriverCode", "QualifyingPosition"]],
            on=["Season", "Round", "DriverCode"], how="inner"))
    if not frames:
        raise ValueError("No historical data retrieved. Check your connection "
                         "or the Jolpica API.")
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["FinishPosition", "QualifyingPosition"])
    return data.sort_values(["Season", "Round", "FinishPosition"]).reset_index(drop=True)
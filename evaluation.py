import numpy as np
import pandas as pd


def _spearman(a, b):
    a, b = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def per_race_metrics(df, score_col):
    rows = []
    for (season, rnd), race in df.groupby(["Season", "Round"]):
        if len(race) < 3:
            continue
        race = race.sort_values([score_col, "DriverCode"], kind="stable")
        pred_rank = np.arange(1, len(race) + 1)
        actual = race["FinishPosition"].to_numpy(float)
        rows.append({
            "Season": season, "Round": rnd,
            "winner_hit": float(actual[0] == 1),
            "top3_hit": float((actual[:3] <= 3).sum() / 3),
            "spearman": _spearman(pred_rank, actual),
            "rank_mae": float(np.abs(pred_rank - actual).mean()),
        })
    return pd.DataFrame(rows)


def summarize(per_race):
    return {
        "races": int(len(per_race)),
        "winner_accuracy": float(per_race["winner_hit"].mean()),
        "top3_hit_rate": float(per_race["top3_hit"].mean()),
        "spearman": float(per_race["spearman"].mean()),
        "rank_mae": float(per_race["rank_mae"].mean()),
    }


def paired_difference(model_races, base_races, metric):
    merged = model_races.merge(base_races, on=["Season", "Round"], suffixes=("_m", "_b"))
    diff = (merged[f"{metric}_m"] - merged[f"{metric}_b"]).dropna()
    n = len(diff)
    if n < 2:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": n}
    mean, se = diff.mean(), diff.std(ddof=1) / np.sqrt(n)
    return {"mean": float(mean), "ci95_low": float(mean - 1.96 * se),
            "ci95_high": float(mean + 1.96 * se), "n": int(n)}
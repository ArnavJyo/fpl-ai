"""Build past-only rolling averages: python -m src.features.build_rolling."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from src.common import load_config, read_table, write_table

KEY = ["element_id", "gameweek"]


def build_rolling_features(
    history: pd.DataFrame,
    windows: Sequence[int],
    stats: Sequence[str],
    target: str = "total_points",
) -> pd.DataFrame:
    """Return target-week rows with averages from strictly earlier gameweeks.

    Windows span calendar gameweeks, including weeks with missing records.
    Missing records are unknown, not zero. Counts describe the available
    records in each window; players without history have missing averages.
    The target is a label, never an input to the model for that same week.
    """
    if not windows or any(type(window) is not int or window < 1 for window in windows):
        raise ValueError("Rolling windows must be positive integers.")
    if len(set(windows)) != len(windows):
        raise ValueError("Rolling windows must be unique.")
    if not stats or len(set(stats)) != len(stats) or set(stats) & set(KEY):
        raise ValueError("Rolling stats must be unique and must not include the row keys.")
    if target in KEY:
        raise ValueError("The target must not be a row key.")
    required = list(dict.fromkeys([*KEY, target, *stats]))
    missing = sorted(set(required) - set(history.columns))
    if missing:
        raise ValueError(f"History is missing required columns: {missing}")
    if history[required].isna().any().any():
        raise ValueError("Required history values must not be missing.")
    for key in KEY:
        if not is_integer_dtype(history[key]) or (history[key] < 1).any():
            raise ValueError(f"{key} must contain positive integers.")
    for stat in dict.fromkeys([target, *stats]):
        if not is_numeric_dtype(history[stat]) or history[stat].isin([float("inf"), -float("inf")]).any():
            raise ValueError(f"{stat} must contain finite numeric values.")
    if history.duplicated(KEY).any():
        raise ValueError("History must have one row per player per gameweek; resolve duplicates upstream.")

    ordered = history.sort_values(KEY).reset_index(drop=True)
    features = ordered[[*KEY, target]].copy()
    for window in windows:
        features[f"history_count_last_{window}"] = pd.Series(0, index=features.index, dtype="int64")
        for stat in stats:
            features[f"{stat}_mean_last_{window}"] = float("nan")

    for _, player in ordered.groupby("element_id", sort=False):
        weeks = range(int(player.gameweek.min()), int(player.gameweek.max()) + 1)
        past = player.set_index("gameweek")[list(stats)].reindex(weeks).shift(1)
        for window in windows:
            rolling = past.rolling(window=window, min_periods=1)
            means = rolling.mean().loc[player.gameweek]
            columns = [f"{stat}_mean_last_{window}" for stat in stats]
            features.loc[player.index, columns] = means.to_numpy()
            counts = rolling.count().loc[player.gameweek, stats[0]].fillna(0)
            features.loc[player.index, f"history_count_last_{window}"] = counts.to_numpy(dtype="int64")

    return features


def main() -> None:
    """Save the rolling-only feature table using the existing configuration."""
    settings = load_config()["features"]
    features = build_rolling_features(
        read_table("history_file"),
        windows=settings["rolling_windows"],
        stats=settings["rolling_stats"],
        target=settings["target"],
    )
    write_table(features, "rolling_features_file")
    print("Rolling-only diagnostic table; use src.features.pipeline for the complete feature dataset.")


if __name__ == "__main__":
    main()

"""Convert event/live JSON into player performance for one gameweek.

Decimal strings become floats; gameweek selection and I/O live in stages.py."""

from __future__ import annotations

# Keep direct-file commands working; package imports use the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Any

import pandas as pd

INT_STATS = [
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "total_points",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "bps",
    "defensive_contribution",
    "recoveries",
    "tackles",
    "clearances_blocks_interceptions",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
]

FLOAT_STATS = [
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "ict_index",
    "influence",
    "creativity",
    "threat",
]

BOOL_STATS = ["played"]

STAT_COLUMNS = INT_STATS + FLOAT_STATS + BOOL_STATS


def unwrap(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "elements" in payload:
        return payload["elements"]
    raise TypeError(f"unexpected live payload: {type(payload).__name__}")


def parse_event(payload: Any, gameweek: int) -> pd.DataFrame:
    rows = []
    for entry in unwrap(payload):
        stats = entry["stats"]
        row = {"element_id": entry["id"], "gameweek": gameweek}
        for column in INT_STATS:
            row[column] = int(stats[column])
        for column in FLOAT_STATS:
            row[column] = float(stats[column])
        for column in BOOL_STATS:
            row[column] = bool(stats[column])
        rows.append(row)

    frame = pd.DataFrame(rows, columns=["element_id", "gameweek", *STAT_COLUMNS])
    frame["gameweek"] = frame["gameweek"].astype("Int64")
    frame["played"] = frame["played"].astype(bool)
    return frame.sort_values(["element_id", "gameweek"]).reset_index(drop=True)


def main() -> None:
    from src.data.pipeline import stage_main

    stage_main("gameweek-stats")


if __name__ == "__main__":
    main()

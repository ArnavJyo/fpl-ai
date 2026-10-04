"""Convert fixtures JSON into one typed row per match.

Summary history.fixture joins to fixture_id, not code."""

from __future__ import annotations

# Keep direct-file commands working; package imports use the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Any

import pandas as pd

COLUMNS = [
    "fixture_id",
    "code",
    "gameweek",
    "kickoff_time",
    "finished",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
    "home_difficulty",
    "away_difficulty",
]


def unwrap(raw: Any) -> list[dict[str, Any]]:
    """The endpoint has returned a bare list, but guard against a dict wrapper."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "fixtures" in raw:
        return raw["fixtures"]
    raise TypeError(f"unexpected fixtures payload: {type(raw).__name__}")


def parse_fixtures(raw: Any) -> pd.DataFrame:
    rows = []
    for fixture in unwrap(raw):
        rows.append(
            {
                "fixture_id": fixture["id"],
                "code": fixture["code"],
                "gameweek": fixture["event"],
                "kickoff_time": fixture["kickoff_time"],
                "finished": fixture["finished"],
                "home_team_id": fixture["team_h"],
                "away_team_id": fixture["team_a"],
                "home_goals": fixture["team_h_score"],
                "away_goals": fixture["team_a_score"],
                "home_difficulty": fixture["team_h_difficulty"],
                "away_difficulty": fixture["team_a_difficulty"],
            }
        )
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True)
    frame["gameweek"] = frame["gameweek"].astype("Int64")
    frame["home_goals"] = frame["home_goals"].astype("Int64")
    frame["away_goals"] = frame["away_goals"].astype("Int64")
    return frame.sort_values(["gameweek", "kickoff_time"]).reset_index(drop=True)


def main() -> None:
    from src.data.pipeline import stage_main

    stage_main("fixtures")


if __name__ == "__main__":
    main()

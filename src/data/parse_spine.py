"""Build historical player context from summaries and fixtures.

Each input history entry represents a fixture. Double gameweeks need explicit
aggregation before the one-to-one player/gameweek history join can succeed."""

from __future__ import annotations

# Keep direct-file commands working; package imports use the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Any

import pandas as pd

COLUMNS = [
    "element_id",
    "gameweek",
    "fixture_id",
    "is_home",
    "own_team_id",
    "opponent_team_id",
    "opponent_difficulty",
    "price_tenths",
    "selected",
    "transfers_in",
    "transfers_out",
    "team_goals_for",
    "team_goals_against",
]


def spine_row(element_id: int, game: dict[str, Any], match: pd.Series) -> dict[str, Any]:
    """Attach fixture context to one player's historical fixture entry."""
    is_home = bool(game["was_home"])
    own_team_id = match["home_team_id"] if is_home else match["away_team_id"]
    opponent_team_id = match["away_team_id"] if is_home else match["home_team_id"]
    goals_for = match["home_goals"] if is_home else match["away_goals"]
    goals_against = match["away_goals"] if is_home else match["home_goals"]

    # TODO: this legacy lookup uses the opposing side's difficulty. Keep its
    # output unchanged during the readability refactor; see the HTML guide.
    difficulty = match["away_difficulty"] if is_home else match["home_difficulty"]

    return {
        "element_id": element_id,
        "gameweek": game["round"],
        "fixture_id": game["fixture"],
        "is_home": is_home,
        "own_team_id": own_team_id,
        "opponent_team_id": opponent_team_id,
        "opponent_difficulty": difficulty,
        "price_tenths": int(game["value"]),
        "selected": int(game["selected"]),
        "transfers_in": int(game["transfers_in"]),
        "transfers_out": int(game["transfers_out"]),
        "team_goals_for": goals_for,
        "team_goals_against": goals_against,
    }


def build_spine(summaries: dict[int, Any], fixtures: pd.DataFrame) -> pd.DataFrame:
    """Build fixture-entry context, logging entries with unknown fixture IDs."""
    by_fixture = fixtures.set_index("fixture_id")
    rows = []
    skipped = 0
    for element_id, summary in summaries.items():
        for game in summary.get("history", []):
            fixture_id = game["fixture"]
            if fixture_id not in by_fixture.index:
                skipped += 1
                continue
            rows.append(spine_row(element_id, game, by_fixture.loc[fixture_id]))

    frame = pd.DataFrame(rows, columns=COLUMNS)
    for column in ["gameweek", "team_goals_for", "team_goals_against", "opponent_difficulty"]:
        frame[column] = frame[column].astype("Int64")
    frame = frame.sort_values(["element_id", "gameweek"]).reset_index(drop=True)
    frame.attrs["skipped"] = skipped
    return frame


def main() -> None:
    from src.data.pipeline import stage_main

    stage_main("spine")


if __name__ == "__main__":
    main()

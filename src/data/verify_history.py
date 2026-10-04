"""Check history against cached player summaries, totals, and match scores.

These checks establish specific invariants; they do not validate every field."""

from __future__ import annotations

# Keep direct-file commands working; package imports use the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from pathlib import Path

import pandas as pd

FIELDS = ["total_points", "minutes", "goals_scored", "assists", "bonus", "bps", "clean_sheets", "saves"]


class Report:
    def __init__(self) -> None:
        self.results: list[tuple[bool, str, str]] = []

    def check(self, passed: bool, name: str, detail: str = "") -> None:
        self.results.append((passed, name, detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    def summary(self) -> bool:
        failed = [r for r in self.results if not r[0]]
        print(f"\n{len(self.results) - len(failed)}/{len(self.results)} checks passed")
        if failed:
            print("\nFAILED:")
            for _, name, detail in failed:
                print(f"  - {name} {detail}")
        return not failed


def check_cross_source(history: pd.DataFrame, raw_dir: Path, report: Report) -> None:
    """A: compare event/live-derived history against element-summary-derived rows."""
    source = {}
    for path in raw_dir.glob("element_summary_*.json"):
        element_id = int(path.stem.split("_")[-1])
        for game in json.loads(path.read_text()).get("history", []):
            source[(element_id, game["round"])] = game

    compared = mismatches = 0
    examples = []
    for row in history.itertuples(index=False):
        game = source.get((row.element_id, row.gameweek))
        if game is None:
            continue
        compared += 1
        for field in FIELDS:
            left, right = getattr(row, field), game[field]
            if abs(float(left) - float(right)) > 1e-9:
                mismatches += 1
                if len(examples) < 5:
                    examples.append(f"  {row.element_id} gw{row.gameweek} {field}: history={left} summary={right}")

    report.check(
        mismatches == 0,
        "A. every player/gameweek matches element-summary on 8 fields",
        f"compared {compared} player-gameweeks, {mismatches} field mismatches",
    )
    for line in examples:
        print(line)


def check_season_totals(history: pd.DataFrame, players: pd.DataFrame, report: Report) -> None:
    summed = history.groupby("element_id")["total_points"].sum()
    joined = players.set_index("element_id")[["web_name", "total_points_season"]].join(
        summed.rename("summed"), how="inner"
    )
    mismatch = joined[joined.total_points_season != joined.summed]
    report.check(
        len(mismatch) == 0,
        "B. gameweek points sum to the season total on bootstrap",
        f"{len(joined) - len(mismatch)}/{len(joined)} players match",
    )
    if len(mismatch):
        print(mismatch.head().to_string())


def check_goal_reconciliation(history: pd.DataFrame, fixtures: pd.DataFrame, report: Report) -> None:
    """C: goals scored by a team's players + opponent own goals == fixture goals.

    This is independent of both endpoints' totals: it checks that player-level
    goals actually add up to the match scoreline.
    """
    by_fixture = fixtures.set_index("fixture_id")

    rows = []
    for fixture_id, group in history.groupby("fixture_id"):
        if fixture_id not in by_fixture.index:
            continue
        match = by_fixture.loc[fixture_id]
        home = group[group.own_team_id == match.home_team_id]
        away = group[group.own_team_id == match.away_team_id]
        rows.append(
            {
                "fixture_id": fixture_id,
                "calc_home": int(home.goals_scored.sum()) + int(away.own_goals.sum()),
                "calc_away": int(away.goals_scored.sum()) + int(home.own_goals.sum()),
                "actual_home": int(match.home_goals),
                "actual_away": int(match.away_goals),
            }
        )
    frame = pd.DataFrame(rows)
    bad = frame[(frame.calc_home != frame.actual_home) | (frame.calc_away != frame.actual_away)]
    report.check(
        len(bad) == 0,
        "C. player goals reconcile to every match scoreline (own goals counted)",
        f"{len(frame) - len(bad)}/{len(frame)} fixtures reconcile",
    )
    if len(bad):
        print(bad.head().to_string(index=False))


def check_shape(history: pd.DataFrame, report: Report) -> None:
    dupes = int(history.duplicated(["element_id", "gameweek"]).sum())
    report.check(dupes == 0, "D1. no duplicate (element_id, gameweek) keys", f"{dupes} duplicates")

    required = ["total_points", "minutes", "price_tenths", "opponent_difficulty", "is_home", "played"]
    missing = history[required].isna().sum()
    missing = missing[missing > 0]
    report.check(len(missing) == 0, "D2. no nulls in required columns", missing.to_dict() or "none null")

    bad_minutes = int((history.minutes > 90).sum())
    report.check(bad_minutes == 0, "D3. no player played more than 90 minutes", f"{bad_minutes} offenders")

    bad_diff = int((~history.opponent_difficulty.between(1, 5)).sum())
    report.check(bad_diff == 0, "D4. opponent_difficulty is always 1-5", f"{bad_diff} offenders")

    bad_points = int((history.total_points.abs() > 40).sum())
    report.check(bad_points == 0, "D5. no implausible single-gameweek point total", f"{bad_points} offenders")


def verify_history(
    history: pd.DataFrame,
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    raw_dir: Path,
) -> Report:
    """Run the existing cross-source and structural checks without rewriting data."""
    report = Report()
    check_cross_source(history, raw_dir, report)
    check_season_totals(history, players, report)
    check_goal_reconciliation(history, fixtures, report)
    check_shape(history, report)
    return report


def main() -> None:
    from src.data.pipeline import stage_main

    stage_main("verify")


if __name__ == "__main__":
    main()

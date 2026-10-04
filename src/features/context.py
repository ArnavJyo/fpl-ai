"""Gameweek-grain history and fixture strength from earlier matches."""

from __future__ import annotations

import pandas as pd

KEY = ["element_id", "gameweek"]
MARKET = ["price_tenths", "selected", "transfers_in", "transfers_out"]


def build_feature_history(stats: pd.DataFrame, spine: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Preserve event/live outcomes, including blanks; collapse fixture context.

    Double-week performance is already aggregated by event/live. Prices and
    ownership/transfer counts are snapshots, so take the last fixture's values
    rather than adding them. Unknown context is not inferred from today's club.
    """
    if stats.duplicated(KEY).any() or stats[KEY].isna().any().any():
        raise ValueError("Performance must have unique, nonmissing player/gameweek keys.")
    if spine.duplicated(["element_id", "fixture_id"]).any():
        raise ValueError("Context contains duplicate player/fixture records.")
    if (spine.groupby(KEY).own_team_id.nunique() > 1).any():
        raise ValueError("A player has multiple clubs in one gameweek; resolve that context first.")
    ordered = spine.merge(
        fixtures[["fixture_id", "kickoff_time", "gameweek", "home_team_id", "away_team_id"]].rename(columns={"gameweek": "fixture_gameweek"}),
        on="fixture_id", how="left", validate="many_to_one",
    )
    if ordered.kickoff_time.isna().any():
        raise ValueError("Historical fixture context needs known kickoff times.")
    if (ordered.gameweek != ordered.fixture_gameweek).any() or not (
        (ordered.own_team_id == ordered.home_team_id) | (ordered.own_team_id == ordered.away_team_id)
    ).all():
        raise ValueError("Player context does not match the fixture's gameweek or clubs.")
    context = ordered.sort_values([*KEY, "kickoff_time", "fixture_id"]).groupby(KEY, as_index=False).agg(
        {"own_team_id": "last", **{column: "last" for column in MARKET}}
    )
    result = stats.merge(context, on=KEY, how="left", validate="one_to_one").sort_values(KEY)
    result["context_observed"] = result.own_team_id.notna().astype(int)
    result["own_team_id"] = result.groupby("element_id").own_team_id.ffill()
    return result.reset_index(drop=True)


def build_team_results(
    fixtures: pd.DataFrame, spine: pd.DataFrame, fixture_stats: pd.DataFrame,
) -> pd.DataFrame:
    """One result per club per match; xGC is the opposing players' summed xG.

    Summing player xGC would count the same team defence multiple times.
    Fixture-level summaries keep xG correctly allocated in double gameweeks.
    """
    if fixtures.fixture_id.duplicated().any():
        raise ValueError("Fixture IDs must be unique.")
    if fixture_stats.duplicated(["element_id", "fixture_id"]).any():
        raise ValueError("Fixture stats must have unique player/fixture keys.")
    if spine.duplicated(["element_id", "fixture_id"]).any():
        raise ValueError("Context must have unique player/fixture keys.")
    joined = spine[["element_id", "fixture_id", "own_team_id"]].merge(
        fixture_stats, on=["element_id", "fixture_id"], how="left", validate="one_to_one",
    )
    grouped = joined.groupby(["fixture_id", "own_team_id"]).expected_goals
    xg = grouped.sum(min_count=1)
    xg[grouped.count() != grouped.size()] = float("nan")
    rows = []
    for match in fixtures.itertuples(index=False):
        if not match.finished or pd.isna(match.gameweek):
            continue
        if pd.isna(match.home_goals) or pd.isna(match.away_goals):
            raise ValueError("Finished fixtures must have scorelines.")
        for team, opponent, goals_for, goals_against in (
            (match.home_team_id, match.away_team_id, match.home_goals, match.away_goals),
            (match.away_team_id, match.home_team_id, match.away_goals, match.home_goals),
        ):
            rows.append({
                "fixture_id": match.fixture_id, "team_id": team, "gameweek": int(match.gameweek),
                "goals_for": goals_for, "goals_against": goals_against,
                "xg_for": xg.get((match.fixture_id, team), float("nan")),
                "xg_against": xg.get((match.fixture_id, opponent), float("nan")),
            })
    return pd.DataFrame(rows, columns=[
        "fixture_id", "team_id", "gameweek", "goals_for", "goals_against", "xg_for", "xg_against",
    ])


def build_fixture_features(
    rows: pd.DataFrame, fixtures: pd.DataFrame, results: pd.DataFrame, windows: list[int],
) -> pd.DataFrame:
    """Describe target fixtures using only opponent results in earlier weeks."""
    schedule = {}
    for match in fixtures.itertuples(index=False):
        if pd.isna(match.gameweek):
            continue
        schedule.setdefault((match.home_team_id, int(match.gameweek)), []).append((match.away_team_id, 1))
        schedule.setdefault((match.away_team_id, int(match.gameweek)), []).append((match.home_team_id, 0))
    records = []
    cache = {}
    metrics = ["goals_for", "goals_against", "xg_for", "xg_against"]
    for row in rows.itertuples(index=False):
        known_team = pd.notna(row.own_team_id)
        matches = schedule.get((int(row.own_team_id), int(row.gameweek)), []) if known_team else []
        record = {
            "fixture_context_known": int(known_team),
            "fixture_count": len(matches) if known_team else float("nan"),
            "home_fixture_count": sum(home for _, home in matches) if known_team else float("nan"),
            "away_fixture_count": sum(1 - home for _, home in matches) if known_team else float("nan"),
            "is_blank": int(not matches) if known_team else float("nan"),
            "is_double": int(len(matches) > 1) if known_team else float("nan"),
        }
        for window in windows:
            opponents = []
            for opponent, _ in matches:
                cache_key = (opponent, int(row.gameweek), window)
                if cache_key not in cache:
                    previous = results[
                        (results.team_id == opponent)
                        & (results.gameweek >= row.gameweek - window)
                        & (results.gameweek < row.gameweek)
                    ]
                    cache[cache_key] = {
                        **previous[metrics].mean().to_dict(), "matches": len(previous),
                    }
                opponents.append(cache[cache_key])
            for metric in metrics:
                values = [opponent[metric] for opponent in opponents]
                record[f"opponent_{metric}_mean_last_{window}"] = (
                    sum(values) / len(values) if values else float("nan")
                )
            record[f"opponent_history_matches_last_{window}"] = (
                sum(opponent["matches"] for opponent in opponents) if known_team else float("nan")
            )
        records.append(record)
    return pd.DataFrame(records, index=rows.index)

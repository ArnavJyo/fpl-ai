"""Run individual data stages: fetch inputs, transform them, and save outputs.

The parse_* modules contain transformations. This module owns file/API I/O.
The pipeline module decides which stages to run and in what order.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.common import load_config, path_for, read_table, write_table
from src.data.build_history import build_history, content_hash, unmatched_keys
from src.data.fpl_client import FplClient
from src.data.parse_bootstrap import parse_players, parse_teams
from src.data.parse_event_live import parse_event
from src.data.parse_fixtures import parse_fixtures
from src.data.parse_spine import build_spine
from src.data.season_state import season_state
from src.data.verify_history import verify_history

Bootstrap = dict[str, Any]


def make_client(refresh: bool = False) -> FplClient:
    """Create a client using the project's configured API and cache paths."""
    api = load_config()["fpl_api"]
    return FplClient(api["base_url"], path_for("raw_data_dir"), refresh=refresh)


def fetch_bootstrap(client: FplClient) -> Bootstrap:
    """Fetch the shared player, team, and event snapshot once."""
    return client.get(load_config()["fpl_api"]["bootstrap_static"], "bootstrap_static")


def run_bootstrap(bootstrap: Bootstrap) -> None:
    """Save current player and team snapshots."""
    write_table(parse_players(bootstrap), "players_file")
    write_table(parse_teams(bootstrap), "teams_file")


def run_fixtures(client: FplClient) -> None:
    """Save one row per match, including schedule, score, and difficulty."""
    raw = client.get(load_config()["fpl_api"]["fixtures"], "fixtures")
    write_table(parse_fixtures(raw), "fixtures_file")


def run_gameweek_stats(
    client: FplClient,
    bootstrap: Bootstrap,
    gameweeks: list[int] | None = None,
) -> None:
    """Rebuild player performance for the selected or finished gameweeks."""
    state = season_state(bootstrap["events"])
    selected_weeks = gameweeks or list(state.finished_gameweeks)
    if not selected_weeks:
        raise ValueError("No finished gameweeks to build. Try again after GW1 finishes.")

    print(state.describe())
    frames = []
    endpoint = load_config()["fpl_api"]["event_live"]
    for gameweek in selected_weeks:
        payload = client.get(endpoint.format(event_id=gameweek), f"event_{gameweek}_live")
        frames.append(parse_event(payload, gameweek))

    stats = pd.concat(frames, ignore_index=True)
    write_table(stats, "gw_stats_file")


def run_spine(client: FplClient, bootstrap: Bootstrap, workers: int = 12) -> None:
    """Save historical fixture, price, and ownership context for every player."""
    element_ids = [element["id"] for element in bootstrap["elements"]]
    endpoint = load_config()["fpl_api"]["element_summary"]
    requests = [
        (endpoint.format(element_id=element_id), f"element_summary_{element_id}")
        for element_id in element_ids
    ]
    payloads = client.get_many(requests, max_workers=workers)
    summaries = dict(zip(element_ids, payloads))
    spine = build_spine(summaries, read_table("fixtures_file"))

    write_table(spine, "spine_file")
    print(f"Summary entries with missing fixtures: {spine.attrs.get('skipped', 0)}")


def run_history() -> None:
    """Combine saved performance and context into the full history table."""
    stats = read_table("gw_stats_file")
    spine = read_table("spine_file")
    only_stats, only_spine = unmatched_keys(stats, spine)

    history = build_history(stats, spine)
    write_table(history, "history_file")
    print(f"Unmatched keys: {len(only_stats)} performance, {len(only_spine)} context")
    if only_stats or only_spine:
        print("WARNING: unmatched rows were dropped by the inner join.")
    print(f"Content hash: {content_hash(history)}")


def run_verification() -> bool:
    """Check saved tables and return whether all existing checks passed."""
    report = verify_history(
        history=read_table("history_file"),
        players=read_table("players_file"),
        fixtures=read_table("fixtures_file"),
        raw_dir=path_for("raw_data_dir"),
    )
    return report.summary()

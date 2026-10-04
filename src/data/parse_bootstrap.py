"""Convert bootstrap JSON into current player and club snapshots.

Historical price and performance belong in the spine and history tables."""

from __future__ import annotations

# Keep direct-file commands working; package imports use the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Any

import pandas as pd

from src.common import position_name

STATUS_LABELS = {
    "a": "available", "d": "doubtful", "i": "injured",
    "s": "suspended", "u": "unknown",
}


def to_float(value: Any) -> float | None:
    """The FPL API returns most numbers as strings. None stays None."""
    if value is None or value == "":
        return None
    return float(value)


def to_availability(value: Any) -> float:
    """chance_of_playing_next_round is null for fully-fit players (=100%)."""
    return 100.0 if value is None else float(value)


def set_piece_rank(order: Any) -> int | None:
    """1 = primary taker. Several players can share rank 1 on different teams."""
    return None if order in (None, "") else int(order)


def player_row(element: dict[str, Any]) -> dict[str, Any]:
    """Map one bootstrap player into the snapshot table's columns."""
    return {
        "element_id": element["id"],
        "web_name": element["web_name"],
        "first_name": element["first_name"],
        "position": position_name(element["element_type"]),
        "element_type": element["element_type"],
        "team_id": element["team"],
        "price_tenths": int(element["now_cost"]),
        "total_points_season": element["total_points"],
        "minutes_season": element["minutes"],
        "goals_season": element["goals_scored"],
        "assists_season": element["assists"],
        "bonus_season": element["bonus"],
        "ict_season": to_float(element["ict_index"]),
        "form": to_float(element["form"]),
        "selected_by_percent": to_float(element["selected_by_percent"]),
        "transfers_in_event": element["transfers_in_event"],
        "transfers_out_event": element["transfers_out_event"],
        "ep_next": to_float(element["ep_next"]),
        "availability_pct": to_availability(element["chance_of_playing_next_round"]),
        "status": element["status"],
        "status_label": STATUS_LABELS.get(element["status"], "unknown"),
        "news": element["news"],
        "penalty_order": set_piece_rank(element.get("penalties_order")),
        "corner_order": set_piece_rank(element.get("corners_and_indirect_freekicks_order")),
        "freekick_order": set_piece_rank(element.get("direct_freekicks_order")),
    }


def parse_players(raw: dict[str, Any]) -> pd.DataFrame:
    """Build the current player snapshot; prices and totals are not historical."""
    frame = pd.DataFrame([player_row(element) for element in raw["elements"]])
    for column in ["penalty_order", "corner_order", "freekick_order"]:
        frame[column] = frame[column].astype("Int64")
    frame["is_penalty_taker"] = frame["penalty_order"] == 1
    frame["is_corner_taker"] = frame["corner_order"] == 1
    frame["is_freekick_taker"] = frame["freekick_order"] == 1
    return frame


def parse_teams(raw: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "team_id": team["id"],
            "name": team["name"],
            "short_name": team["short_name"],
            "strength_overall_home": team["strength_overall_home"],
            "strength_overall_away": team["strength_overall_away"],
            "strength_attack_home": team["strength_attack_home"],
            "strength_attack_away": team["strength_attack_away"],
            "strength_defence_home": team["strength_defence_home"],
            "strength_defence_away": team["strength_defence_away"],
        }
        for team in raw["teams"]
    ]
    return pd.DataFrame(rows)


def main() -> None:
    from src.data.pipeline import stage_main

    stage_main("bootstrap")


if __name__ == "__main__":
    main()

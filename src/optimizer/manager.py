"""Public team context and an exact, user-supplied transfer-state contract.

Public picks are a past-deadline snapshot. Never invent current selling prices,
bank, or remaining free transfers from that snapshot. See docs/OPTIMIZER.md.
"""

from __future__ import annotations

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.common import PROJECT_ROOT, load_config
from src.data.fpl_client import FplClient


def integer(value: object, name: str, minimum: int = 0) -> int:
    """Reject booleans, decimals, and UUIDs instead of silently coercing them."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def validate_manager_state(state: dict, season_id: str, gameweek: int) -> dict:
    """Require exact current money and remaining transfers for this forecast."""
    integer(state.get("gameweek"), "gameweek", 1)
    if not isinstance(state.get("season_id"), str) or not state["season_id"] or not 1 <= state["gameweek"] <= 38:
        raise ValueError("Manager state requires a season_id and gameweek in 1..38.")
    if state.get("season_id") != season_id or state.get("gameweek") != gameweek:
        raise ValueError("Manager state season/gameweek differs from the prediction.")
    integer(state.get("team_id"), "team_id", 1)
    integer(state.get("bank_tenths"), "bank_tenths")
    integer(state.get("existing_transfer_hit_points", 0), "existing_transfer_hit_points")
    free = integer(state.get("free_transfers"), "free_transfers")
    if free > 5:
        raise ValueError("free_transfers must be the remaining allowance, between 0 and 5.")
    if state.get("active_chip") not in {None, "wildcard", "freehit", "3xc", "bboost"}:
        raise ValueError("Unsupported active_chip; use null, wildcard, freehit, 3xc, or bboost.")
    picks = state.get("picks", [])
    if not isinstance(picks, list) or len(picks) != 15:
        raise ValueError("Manager state must contain all 15 current players.")
    ids = []
    for pick in picks:
        ids.append(integer(pick.get("element_id"), "element_id", 1))
        integer(pick.get("selling_price_tenths"), "selling_price_tenths", 1)
    if len(set(ids)) != 15:
        raise ValueError("Manager picks must contain unique player IDs.")
    observed = datetime.fromisoformat(state.get("observed_at", "").replace("Z", "+00:00"))
    if observed.tzinfo is None or observed > datetime.now(timezone.utc):
        raise ValueError("Manager observed_at must be a timezone-aware timestamp no later than now.")
    return state


def import_my_team(payload: dict, team_id: int, season_id: str, gameweek: int,
                   active_chip: str | None = None) -> dict:
    """Convert a manually supplied authenticated API response without logging in.

    limit is this week's free allowance; made includes transfers already made.
    A chip is explicit because chip availability does not mean it is activated.
    """
    transfers = payload["transfers"]
    made = integer(transfers.get("made"), "transfers.made")
    limit = transfers.get("limit")
    if limit is None and active_chip in {"wildcard", "freehit"}:
        remaining = 0  # This chip grants unlimited free transfers in the solver.
    else:
        remaining = max(0, integer(limit, "transfers.limit") - made)
    state = {"team_id": team_id, "season_id": season_id, "gameweek": gameweek,
             "observed_at": datetime.now(timezone.utc).isoformat(),
             "bank_tenths": integer(transfers.get("bank"), "transfers.bank"),
             "free_transfers": remaining, "active_chip": active_chip,
             # The live API's cost is points PER paid transfer, not the total.
             "existing_transfer_hit_points": (max(0, made - limit) * integer(transfers.get("cost", 4), "transfers.cost"))
                 if isinstance(limit, int) and active_chip not in {"wildcard", "freehit"} else 0,
             "picks": [{"element_id": pick["element"], "selling_price_tenths": pick["selling_price"]}
                       for pick in payload["picks"]]}
    return validate_manager_state(state, season_id, gameweek)


def load_manager_state(settings: dict, season_id: str, gameweek: int, path: Path | None = None) -> dict | None:
    source = PROJECT_ROOT / (path if path is not None else settings["paths"]["manager_state_file"])
    if not source.exists():
        if path is not None:
            raise ValueError(f"Manager-state file does not exist: {source}")
        return None
    state = validate_manager_state(json.loads(source.read_text()), season_id, gameweek)
    configured_id = settings["manager"]["team_id"]
    if configured_id is not None and configured_id != state["team_id"]:
        raise ValueError("Configured team ID differs from the manager-state file.")
    return state


def public_manager_context(settings: dict, prediction_gameweek: int, finished_gameweeks: list[int], refresh: bool = False, offline: bool = False) -> dict:
    """Read public last-deadline picks for display, never as exact transfer state."""
    team_id = settings["manager"]["team_id"]
    if team_id is None:
        return {"status": "awaiting_numeric_team_id", "team_id": None,
                "note": "An FPL account UUID cannot identify a public team entry."}
    integer(team_id, "manager.team_id", 1)
    api = settings["fpl_api"]
    weeks = [week for week in finished_gameweeks if week < prediction_gameweek]
    cache_names = [f"entry_{team_id}", *([f"entry_{team_id}_gw{max(weeks)}_picks"] if weeks else [])]
    raw_dir = PROJECT_ROOT / settings["paths"]["raw_data_dir"]
    if offline and any(not (raw_dir / f"{name}.json").exists() for name in cache_names):
        return {"status": "public_snapshot_not_cached", "team_id": team_id,
                "note": "Run online once to fetch the public team context."}
    client = FplClient(api["base_url"], raw_dir, refresh=refresh)
    try:
        profile = client.get(api["entry"].format(manager_id=team_id), f"entry_{team_id}")
        if not weeks:
            return {"status": "no_previous_picks", "team_id": team_id, "team_name": profile.get("name")}
        week = max(weeks)
        picks = client.get(api["entry_picks"].format(manager_id=team_id, event_id=week), f"entry_{team_id}_gw{week}_picks")
        return {"status": "public_last_deadline_snapshot", "team_id": team_id, "team_name": profile.get("name"),
                "picks_gameweek": week, "player_ids": [int(pick["element"]) for pick in picks["picks"]],
                "last_deadline_bank_tenths": profile.get("last_deadline_bank"),
                "note": f"Public squad from the GW{week} deadline; later transfers may not be reflected. Exact transfer planning needs your current squad, selling prices, bank, and remaining free transfers."}
    finally:
        client.session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import your saved my-team JSON; no credentials are used.")
    parser.add_argument("--import-my-team", type=Path, required=True)
    parser.add_argument("--chip", choices=["wildcard", "freehit", "3xc", "bboost"])
    args = parser.parse_args()
    settings = load_config()
    manifest = json.loads((PROJECT_ROOT / settings["paths"]["feature_manifest_file"]).read_text())
    state = import_my_team(json.loads(args.import_my_team.read_text()), settings["manager"]["team_id"],
                           manifest["season_id"], manifest["prediction_gameweek"], args.chip)
    from src.models.train import save_json

    path = PROJECT_ROOT / settings["paths"]["manager_state_file"]
    save_json(path, state)
    print(f"Imported exact team state for entry {state['team_id']}, GW{state['gameweek']}: {path}")


if __name__ == "__main__":
    main()

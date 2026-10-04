"""Plan from LIVE team state; compare one chip at a time against no chip.

Scarce chips have configurable minimum marginal gains. A one-week model cannot
learn the best season-long chip timing, so these are explicit policy heuristics.
The free-transfer cap is a hard constraint, not just a four-point penalty.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from src.common import position_name
from src.data.season_state import season_state
from src.optimizer.manager import import_my_team, integer
from src.optimizer.solve import OptimizationError, candidate_table, solve_squad
from src.automation.client import FplActionError

CHIPS = {"wildcard", "freehit", "bboost", "3xc"}
TRANSFER_CHIPS = {"wildcard", "freehit"}


def timestamp(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError
        return result
    except (AttributeError, TypeError, ValueError):
        raise FplActionError("A required action timestamp is missing or has no timezone.") from None


def validate_forecast(forecast: dict, bootstrap: dict, policy: dict, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    upcoming = season_state(bootstrap["events"])
    first_deadline = min(bootstrap["events"], key=lambda e: e["id"])["deadline_time"]
    if forecast["season_id"] != first_deadline or forecast["gameweek"] != upcoming.next_gameweek:
        raise FplActionError("Forecast is for a different season/gameweek than the live FPL deadline.")
    buffer = integer(policy["deadline_buffer_seconds"], "deadline_buffer_seconds")
    if (timestamp(upcoming.next_deadline) - now).total_seconds() <= buffer:
        raise FplActionError("The live FPL deadline is too close or has passed; no changes submitted.")
    age = (now - timestamp(forecast.get("generated_at"))).total_seconds()
    max_age = float(policy["max_forecast_age_hours"]) * 3600
    if max_age <= 0 or not 0 <= age <= max_age:
        raise FplActionError("Forecast is stale or dated in the future; refresh the full pipeline first.")
    source_age = (now - timestamp(forecast.get("source_snapshot_observed_at"))).total_seconds()
    if not 0 <= source_age <= max_age:
        raise FplActionError("Forecast inputs are stale; refresh the full pipeline before live execution.")
    if forecast.get("training_status") == "experimental" and not policy["allow_experimental"]:
        raise FplActionError("Live execution of an experimental model is disabled in configuration.")


def chip_state(payload: dict, gameweek: int) -> tuple[str | None, set[str]]:
    chips = payload.get("chips")
    if not isinstance(chips, list):
        raise FplActionError("Live My Team response has no chip availability data.")
    active, available = set(), set()
    for chip in chips:
        name, status = chip.get("name"), chip.get("status_for_entry")
        if status in {"active", "proposed"}:
            if name not in CHIPS:
                raise FplActionError("An unsupported chip is active; automatic execution stopped.")
            active.add(name)
        if name in CHIPS and status == "available":
            start, end = chip.get("start_event"), chip.get("stop_event", chip.get("end_event"))
            if (start is None or gameweek >= start) and (end is None or gameweek <= end):
                available.add(name)
    if len(active) > 1:
        raise FplActionError("My Team reports multiple active chips; automatic execution stopped.")
    return next(iter(active), None), available


def team_fingerprint(payload: dict) -> str:
    """Hash money, current players, lineup and chips, excluding volatile metadata."""
    normalized = {"picks": sorted(payload["picks"], key=lambda p: p["element"]),
                  "transfers": payload["transfers"], "chips": sorted(payload["chips"], key=lambda c: (c.get("name", ""), c.get("id", 0)))}
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def live_candidates(bootstrap: dict, fixtures: list[dict], forecast: dict) -> pd.DataFrame:
    """Use live integer buy prices, positions, availability and fixture counts."""
    players = pd.DataFrame([{"element_id": e["id"], "web_name": e["web_name"], "position": position_name(e["element_type"]),
                             "team_id": e["team"], "price_tenths": e["now_cost"],
                             "availability_pct": 100 if e.get("chance_of_playing_next_round") is None else e["chance_of_playing_next_round"],
                             "status": e["status"]} for e in bootstrap["elements"]])
    counts = {t["id"]: 0 for t in bootstrap["teams"]}
    for fixture in fixtures:
        if fixture.get("event") == forecast["gameweek"]:
            for club in [fixture["team_h"], fixture["team_a"]]:
                counts[club] += 1
    context = pd.DataFrame([{"element_id": e["id"], "gameweek": forecast["gameweek"], "fixture_context_known": 1,
                             "fixture_count": counts[e["team"]]} for e in bootstrap["elements"]])
    teams = pd.DataFrame([{"team_id": t["id"], "short_name": t["short_name"]} for t in bootstrap["teams"]])
    return candidate_table(players, forecast, context, teams)


def choose_chip(candidates: pd.DataFrame, state: dict, available: set[str], rules: dict, policy: dict) -> tuple[dict, list[dict]]:
    hit_budget = integer(policy["max_additional_hit_points"], "max_additional_hit_points")
    if hit_budget != 0:
        raise FplActionError("This executor is authorized for free transfers only; additional hits must stay zero.")
    def solve(chip):
        constrained = {**rules, "max_transfers": min(rules["max_transfers"], state["free_transfers"])}
        return solve_squad(candidates, constrained, {**state, "active_chip": chip})

    current = state.get("active_chip")
    if current:
        result = solve(current)
        return result, [{"chip": current, "reason": "Preserve the chip already active on your team."}]
    evaluated, options = [], []
    try:
        baseline = solve(None)
        options.append((baseline["net_expected_points"], baseline))
        evaluated.append({"chip": None, "net_expected_points": baseline["net_expected_points"], "eligible": True})
    except OptimizationError:
        baseline = None
    for chip in sorted(available.intersection(policy["allowed_chips"])):
        minimum = float(policy["chip_min_gain"][chip])
        if not 0 <= minimum < float("inf"):
            raise FplActionError("Chip gain thresholds must be finite and nonnegative.")
        try:
            result = solve(chip)
        except OptimizationError:
            evaluated.append({"chip": chip, "eligible": False, "reason": "No feasible legal team."})
            continue
        gain = result["net_expected_points"] - baseline["net_expected_points"] if baseline else None
        eligible = baseline is None or gain > minimum
        evaluated.append({"chip": chip, "net_expected_points": result["net_expected_points"],
                          "gain_over_no_chip": gain, "minimum_gain": minimum, "eligible": eligible})
        if eligible:
            # Subtract the configured reserve value to preserve chips for better
            # weeks. Exact ties prefer the no-chip option inserted first.
            options.append((result["net_expected_points"] - minimum, result))
    if not options:
        raise FplActionError("No legal free-transfer plan exists with the currently available chips.")
    return max(options, key=lambda option: option[0])[1], evaluated


def build_plan(forecast: dict, bootstrap: dict, fixtures: list[dict], payload: dict, settings: dict,
               now: datetime | None = None) -> dict:
    policy = settings["automation"]
    validate_forecast(forecast, bootstrap, policy, now)
    active, available = chip_state(payload, forecast["gameweek"])
    state = import_my_team(payload, settings["manager"]["team_id"], forecast["season_id"], forecast["gameweek"], active)
    candidates = live_candidates(bootstrap, fixtures, forecast)
    result, evaluation = choose_chip(candidates, state, available, settings["squad_rules"], policy)
    if result["additional_transfer_hit_points"] != 0:
        raise FplActionError("The proposed team would spend transfer-hit points; stopped.")
    frame = candidates.set_index("element_id")
    selling = {p["element_id"]: p["selling_price_tenths"] for p in state["picks"]}
    transfers = []
    for position in ["GKP", "DEF", "MID", "FWD"]:
        incoming = sorted(i for i in result["transfers"]["in"] if frame.at[i, "position"] == position)
        outgoing = sorted(i for i in result["transfers"]["out"] if frame.at[i, "position"] == position)
        if len(incoming) != len(outgoing):
            raise FplActionError("Transfer pair positions do not match.")
        transfers.extend({"element_in": new, "element_out": old, "purchase_price": int(frame.at[new, "price_tenths"]),
                          "selling_price": selling[old]} for new, old in zip(incoming, outgoing))
    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    starters = sorted(result["starting_xi_ids"], key=lambda i: (order[frame.at[i, "position"]], i))
    ids = [*starters, result["bench_goalkeeper_id"], *result["bench_outfield_ids"]]
    picks = [{"element": player, "position": slot, "is_captain": player == result["captain_id"],
              "is_vice_captain": player == result["vice_captain_id"]} for slot, player in enumerate(ids, start=1)]
    chip = result["active_chip"]
    return {"format_version": 1, "team_id": state["team_id"], "season_id": state["season_id"], "gameweek": state["gameweek"],
            "created_at": (now or datetime.now(timezone.utc)).isoformat(), "forecast_generated_at": forecast["generated_at"],
            "training_status": forecast.get("training_status"),
            "source_snapshot_observed_at": forecast["source_snapshot_observed_at"],
            "deadline": season_state(bootstrap["events"]).next_deadline,
            "source_fingerprint": team_fingerprint(payload), "source_bank_tenths": state["bank_tenths"],
            "source_transfer_made": payload["transfers"]["made"], "free_transfers_before": state["free_transfers"],
            "source_active_chip": next((c["name"] for c in payload["chips"] if c.get("status_for_entry") == "active"), None),
            "chip": chip, "bank_after_tenths": result["bank_after_tenths"],
            "transfer_request": {"entry": state["team_id"], "event": state["gameweek"],
                                 "chip": chip if chip in TRANSFER_CHIPS else None, "transfers": transfers},
            # FPL team POST includes team chips only; transfer chips are already
            # active on transfers/ and must not be resent/cancelled here.
            "lineup_request": {"chip": chip if chip in {"bboost", "3xc"} else None, "picks": picks},
            "net_expected_points": result["net_expected_points"], "additional_hit_points": 0,
            "chip_evaluation": evaluation,
            "players": [{"element_id": int(i), "name": str(frame.at[i, "web_name"]), "position": str(frame.at[i, "position"])} for i in ids]}

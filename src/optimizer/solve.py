"""Select a legal squad, XI, captain, and transfers with PuLP/CBC.

Run: python -m src.optimizer.solve [--manager-state data/manager_state.json].
The pure solve_squad function has no API/file side effects. See docs/OPTIMIZER.md
for every variable, constraint, budget equation, and output field.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

from src.common import PROJECT_ROOT, load_config
from src.models.train import save_json
from src.optimizer.manager import integer, load_manager_state, public_manager_context, validate_manager_state


class OptimizationError(ValueError):
    """Input contracts or solver feasibility prevented a usable recommendation."""


def candidate_table(players: pd.DataFrame, forecasts: dict, features: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Join one-week forecasts to display/price context; blank weeks score zero."""
    rows = pd.DataFrame(forecasts["predictions"])
    needed = {"element_id", "gameweek", "predicted_points"}
    if rows.empty or not needed.issubset(rows) or rows.element_id.duplicated().any():
        raise OptimizationError("Predictions must contain one row per player.")
    if not rows.gameweek.eq(forecasts["gameweek"]).all():
        raise OptimizationError("Prediction records and metadata disagree on the gameweek.")
    if players.element_id.duplicated().any() or teams.team_id.duplicated().any():
        raise OptimizationError("Player and team snapshots need unique IDs.")
    context = features.loc[features.gameweek == forecasts["gameweek"], ["element_id", "fixture_context_known", "fixture_count"]]
    if context.element_id.duplicated().any():
        raise OptimizationError("Fixture context needs one row per player.")
    rows = rows.merge(players[["element_id", "web_name", "position", "team_id", "price_tenths", "availability_pct", "status"]],
                      on="element_id", how="left", validate="one_to_one")
    rows = rows.merge(context, on="element_id", how="left", validate="one_to_one")
    rows = rows.merge(teams[["team_id", "short_name"]], on="team_id", how="left", validate="many_to_one")
    if rows[["position", "team_id", "price_tenths", "fixture_context_known", "fixture_count", "short_name"]].isna().any().any():
        raise OptimizationError("Every forecast needs player, team, and fixture context.")
    if not rows.fixture_context_known.eq(1).all():
        raise OptimizationError("Upcoming fixture context is unknown; refresh sources before optimizing.")
    rows["raw_predicted_points"] = rows.predicted_points
    rows.loc[rows.fixture_count.eq(0), "predicted_points"] = 0.0
    # Retain unavailable existing players if transfer limits require it; they
    # cannot start, and the default rules prevent purchasing new unavailable ones.
    rows["can_start"] = rows.availability_pct.gt(0) & ~rows.status.isin(["u", "s"])
    rows.loc[~rows.can_start, "predicted_points"] = 0.0
    return rows.sort_values("element_id").reset_index(drop=True)


def solve_squad(candidates: pd.DataFrame, rules: dict, manager_state: dict | None = None) -> dict:
    """Maximize XI + captain points minus hits under exact integer money rules.

    A second solve keeps the primary optimum and improves bench/vice quality.
    Both phases must prove optimality; a timeout incumbent is not labeled optimal.
    """
    needed = {"element_id", "position", "team_id", "price_tenths", "predicted_points", "can_start"}
    if candidates.empty or not needed.issubset(candidates) or candidates.element_id.duplicated().any():
        raise OptimizationError("Candidates must be nonempty and uniquely identified with complete optimizer columns.")
    for column in ["element_id", "team_id", "price_tenths"]:
        for value in candidates[column].tolist():
            integer(value, column, 1)
    if not np.isfinite(candidates.predicted_points.to_numpy(dtype=float)).all():
        raise OptimizationError("Predicted points must be finite.")
    if not candidates.can_start.isin([True, False]).all() or candidates.can_start.isna().any():
        raise OptimizationError("can_start must contain booleans.")
    if not candidates.position.isin(["GKP", "DEF", "MID", "FWD"]).all():
        raise OptimizationError("Unsupported player position.")
    if rules["squad_size"] != 15 or rules["starting_xi_size"] != 11 or rules["positions"] != {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}:
        raise OptimizationError("The optimizer requires standard FPL squad sizes and positions.")
    for name in ["max_players_per_club", "max_transfers", "solver_time_limit_seconds"]:
        integer(rules[name], name, 0 if name == "max_transfers" else 1)
    if not np.isfinite(rules["transfer_hit_cost"]) or rules["transfer_hit_cost"] < 0:
        raise OptimizationError("Transfer-hit cost must be finite and nonnegative.")
    frame = candidates.set_index("element_id", drop=False)
    ids = [int(player) for player in frame.index]
    prices = frame.price_tenths.to_dict()
    points = {player: float(frame.at[player, "predicted_points"]) if bool(frame.at[player, "can_start"]) else 0.0 for player in ids}
    current, selling, bank, free, chip = set(), {}, 0, 0, None
    if manager_state is not None:
        # Validate the full contract even when called without the file-I/O wrapper.
        validate_manager_state(manager_state, manager_state.get("season_id"), manager_state.get("gameweek"))
        current = {pick["element_id"] for pick in manager_state["picks"]}
        if not current.issubset(ids):
            raise OptimizationError("Current squad contains players without predictions; rebuild the full forecast universe.")
        if frame.loc[list(current)].position.value_counts().to_dict() != rules["positions"]:
            raise OptimizationError("Current squad does not match the required 2/5/5/3 position counts.")
        selling = {pick["element_id"]: pick["selling_price_tenths"] for pick in manager_state["picks"]}
        if any(selling[player] > prices[player] for player in current):
            raise OptimizationError("Selling prices exceed current market prices; refresh or correct the manager snapshot.")
        bank, free, chip = manager_state["bank_tenths"], manager_state["free_transfers"], manager_state.get("active_chip")
    budget = integer(rules["budget_tenths"], "budget_tenths", 1)
    problem = pulp.LpProblem("FPL_one_gameweek", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    start = pulp.LpVariable.dicts("start", ids, cat="Binary")
    captain = pulp.LpVariable.dicts("captain", ids, cat="Binary")
    vice = pulp.LpVariable.dicts("vice", ids, cat="Binary")
    for player in ids:
        problem += start[player] <= squad[player]
        problem += captain[player] <= start[player]
        problem += vice[player] <= start[player]
        problem += captain[player] + vice[player] <= 1
        if not bool(frame.at[player, "can_start"]):
            problem += start[player] == 0
            if player not in current and rules.get("exclude_unavailable_new_signings", True):
                problem += squad[player] == 0
    problem += pulp.lpSum(squad.values()) == 15
    problem += pulp.lpSum(start.values()) == 11
    problem += pulp.lpSum(captain.values()) == 1
    problem += pulp.lpSum(vice.values()) == 1
    for position, count in rules["positions"].items():
        position_ids = [player for player in ids if frame.at[player, "position"] == position]
        problem += pulp.lpSum(squad[player] for player in position_ids) == count
        if position == "GKP":
            problem += pulp.lpSum(start[player] for player in position_ids) == 1
        else:
            label = {"DEF": "defenders", "MID": "midfielders", "FWD": "forwards"}[position]
            problem += pulp.lpSum(start[player] for player in position_ids) >= rules[f"min_{label}"]
            problem += pulp.lpSum(start[player] for player in position_ids) <= rules[f"max_{label}"]
    for club in frame.team_id.unique():
        problem += pulp.lpSum(squad[player] for player in ids if frame.at[player, "team_id"] == club) <= rules["max_players_per_club"]
    transfers = pulp.lpSum(squad[player] for player in ids if player not in current) if current else 0
    paid = pulp.LpVariable("paid_transfers", lowBound=0, upBound=15, cat="Integer")
    transfer_chip = chip in {"wildcard", "freehit"}
    if current:
        problem += pulp.lpSum(prices[player] * squad[player] for player in ids if player not in current) <= bank + pulp.lpSum(selling[player] * (1 - squad[player]) for player in current)
        problem += transfers <= (15 if transfer_chip else rules["max_transfers"])
        if transfer_chip:
            problem += paid == 0
        else:
            problem += paid >= transfers - free
    else:
        problem += pulp.lpSum(prices[player] * squad[player] for player in ids) <= budget
        problem += paid == 0
    scoring = squad if chip == "bboost" else start
    captain_extra = 2 if chip == "3xc" else 1
    gross = pulp.lpSum(points[player] * (scoring[player] + captain_extra * captain[player]) for player in ids)
    existing_hit = float(manager_state.get("existing_transfer_hit_points", 0)) if manager_state else 0.0
    primary = gross - rules["transfer_hit_cost"] * paid - existing_hit
    problem.setObjective(primary)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=rules["solver_time_limit_seconds"], threads=1, gapRel=0, options=["randomSeed 42"])
    if not solver.available():
        raise OptimizationError("CBC is unavailable; install the pinned PuLP package with its bundled solver.")

    def optimize(phase: str) -> None:
        problem.solve(solver)
        if problem.status != pulp.LpStatusOptimal or problem.sol_status != pulp.LpSolutionOptimal:
            raise OptimizationError(f"{phase} solve did not prove optimality: {pulp.LpStatus[problem.status]}. No recommendation saved.")

    optimize("Primary")
    best = float(pulp.value(primary))
    problem += primary >= best - 1e-6
    # Secondary rewards do not trade away points for bench value. Price and
    # transfer terms break near ties; all remain subject to the primary bound.
    secondary = pulp.lpSum(points[player] * (squad[player] + vice[player]) - 0.001 * prices[player] * squad[player] for player in ids)
    if current:
        secondary -= 0.01 * transfers
    problem.setObjective(secondary)
    optimize("Secondary")
    chosen = [player for player in ids if pulp.value(squad[player]) > 0.5]
    starters = [player for player in chosen if pulp.value(start[player]) > 0.5]
    captain_id = next(player for player in starters if pulp.value(captain[player]) > 0.5)
    vice_id = next(player for player in starters if pulp.value(vice[player]) > 0.5)
    bench = [player for player in chosen if player not in starters]
    # The reserve keeper is separate from the ordered three outfield substitutes.
    bench.sort(key=lambda player: (frame.at[player, "position"] == "GKP", -points[player], player))
    incoming, outgoing = sorted(set(chosen) - current) if current else [], sorted(current - set(chosen))
    paid_count = max(0, len(incoming) - free) if current and not transfer_chip else 0
    hit = float(paid_count * rules["transfer_hit_cost"])
    starting_points = float(sum(points[player] for player in starters))
    gross_points = float(sum(points[player] for player in (chosen if chip == "bboost" else starters)) + captain_extra * points[captain_id])
    result = {
        "mode": "transfers" if current else "initial_squad", "solver": "CBC", "solver_status": "Optimal",
        "squad_ids": chosen, "starting_xi_ids": starters, "captain_id": captain_id, "vice_captain_id": vice_id,
        "bench_ids": bench, "bench_outfield_ids": [player for player in bench if frame.at[player, "position"] != "GKP"],
        "bench_goalkeeper_id": next(player for player in bench if frame.at[player, "position"] == "GKP"),
        "formation": {position: int(sum(frame.at[player, "position"] == position for player in starters)) for position in ["DEF", "MID", "FWD"]},
        "market_value_tenths": int(sum(prices[player] for player in chosen)),
        "budget_tenths": bank + sum(selling.values()) if current else budget,
        "bank_after_tenths": int(bank + sum(selling[player] for player in outgoing) - sum(prices[player] for player in incoming)) if current else int(budget - sum(prices[player] for player in chosen)),
        "active_chip": chip, "starting_xi_points": starting_points, "gross_expected_points": gross_points,
        "additional_transfer_hit_points": hit, "existing_transfer_hit_points": existing_hit,
        "transfer_hit_points": hit + existing_hit, "net_expected_points": gross_points - hit - existing_hit, "primary_optimum": best,
        "transfers": {"in": incoming, "out": outgoing, "count": len(incoming), "paid_count": paid_count,
                      "free_transfers_before": free if current else None},
    }
    if len(chosen) != 15 or len(starters) != 11 or result["bank_after_tenths"] < 0 or result["net_expected_points"] < best - 1e-5:
        raise OptimizationError("Solver result failed the final squad/budget/objective checks.")
    return result


def run_optimization(settings: dict | None = None, manager_path: Path | None = None,
                     fetch_manager: bool = False, refresh: bool = False) -> dict:
    """Load forecasts, solve, and save a labeled result only after success."""
    settings = load_config() if settings is None else settings
    paths = settings["paths"]
    forecasts = json.loads((PROJECT_ROOT / paths["predictions_file"]).read_text())
    manifest = json.loads((PROJECT_ROOT / paths["feature_manifest_file"]).read_text())
    if forecasts["season_id"] != manifest["season_id"] or forecasts["gameweek"] != manifest["prediction_gameweek"]:
        raise OptimizationError("Predictions and feature manifest disagree; regenerate predictions.")
    candidates = candidate_table(pd.read_parquet(PROJECT_ROOT / paths["players_file"]), forecasts,
                                 pd.read_parquet(PROJECT_ROOT / paths["prediction_features_file"]),
                                 pd.read_parquet(PROJECT_ROOT / paths["teams_file"]))
    state = load_manager_state(settings, forecasts["season_id"], forecasts["gameweek"], manager_path)
    result = solve_squad(candidates, settings["squad_rules"], state)
    context = public_manager_context(settings, forecasts["gameweek"], manifest["finished_gameweeks"],
                                     refresh=refresh, offline=not fetch_manager)
    public_lineup = None
    if context.get("player_ids"):
        fixed = candidates.loc[candidates.element_id.isin(context["player_ids"])].copy()
        fixed_rules = {**settings["squad_rules"], "budget_tenths": max(1, int(fixed.price_tenths.sum())),
                       "exclude_unavailable_new_signings": False}
        try:
            public_lineup = solve_squad(fixed, fixed_rules)
        except OptimizationError as exc:
            context["lineup_unavailable_reason"] = str(exc)
    baseline = None
    if state:
        no_transfer_rules = {**settings["squad_rules"], "max_transfers": 0}
        # Remove transfer chips for a fixed-squad counterfactual while preserving
        # scoring chips. An infeasible injured XI is reported, never fabricated.
        comparison_state = {**state, "active_chip": None if state.get("active_chip") in {"wildcard", "freehit"} else state.get("active_chip")}
        try:
            baseline = solve_squad(candidates, no_transfer_rules, comparison_state)
        except OptimizationError:
            baseline = None
    result.update({"season_id": forecasts["season_id"], "gameweek": forecasts["gameweek"],
                   "generated_at": datetime.now(timezone.utc).isoformat(), "training_status": forecasts["training_status"],
                   "model_checkpoint": forecasts["model_checkpoint"], "validation_metrics": forecasts["validation_metrics"],
                   "source_snapshot_observed_at": manifest.get("bootstrap_cache_observed_at"),
                   "manager_context": context, "manager_state_observed_at": state.get("observed_at") if state else None,
                   "public_snapshot_lineup": public_lineup,
                   "current_squad_net_expected_points": baseline["net_expected_points"] if baseline else None,
                   "expected_gain_after_hits": result["net_expected_points"] - baseline["net_expected_points"] if baseline else None,
                   "scope": "One-gameweek expected points. Auto-substitution probabilities and future transfer value are not modeled."})
    result["squad"] = []
    for player in result["squad_ids"]:
        row = candidates.loc[candidates.element_id == player].iloc[0]
        result["squad"].append({"element_id": int(player), "name": str(row.web_name), "position": str(row.position),
                                "team": str(row.short_name), "team_id": int(row.team_id), "price_tenths": int(row.price_tenths),
                                "predicted_points": float(row.predicted_points), "raw_predicted_points": float(row.raw_predicted_points),
                                "availability_pct": float(row.availability_pct), "fixture_count": int(row.fixture_count),
                                "starter": player in result["starting_xi_ids"], "captain": player == result["captain_id"],
                                "vice_captain": player == result["vice_captain_id"]})
    by_id = candidates.set_index("element_id")
    selling_prices = {pick["element_id"]: pick["selling_price_tenths"] for pick in state["picks"]} if state else {}
    result["transfer_details"] = {}
    for direction in ["in", "out"]:
        result["transfer_details"][direction] = [
            {"element_id": int(player), "name": str(by_id.at[player, "web_name"]),
             "price_tenths": int(selling_prices[player] if direction == "out" else by_id.at[player, "price_tenths"])}
            for player in result["transfers"][direction]]
    save_json(PROJECT_ROOT / paths["manager_context_file"], context)
    save_json(PROJECT_ROOT / paths["squad_file"], result)
    print(f"GW{result['gameweek']} {result['mode']}: {result['net_expected_points']:.2f} expected points; bank £{result['bank_after_tenths']/10:.1f}m")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manager-state", type=Path)
    parser.add_argument("--refresh-manager", action="store_true")
    args = parser.parse_args()
    run_optimization(manager_path=args.manager_state, fetch_manager=args.refresh_manager, refresh=args.refresh_manager)


if __name__ == "__main__":
    main()

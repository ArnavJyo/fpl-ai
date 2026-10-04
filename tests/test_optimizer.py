"""Known optima, legal formations, exact transfer money, hits, and team state."""

import copy
import itertools
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pulp

from src.common import load_config
from src.optimizer.manager import import_my_team, validate_manager_state
from src.optimizer.solve import OptimizationError, candidate_table, solve_squad


def roster(extra_midfield_points=None):
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    scores = [4, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 9, 5, 3]
    records = [{"element_id": i, "position": position, "team_id": i, "price_tenths": 50,
                "predicted_points": float(score), "can_start": True}
               for i, (position, score) in enumerate(zip(positions, scores), start=1)]
    if extra_midfield_points is not None:
        records.append({"element_id": 16, "position": "MID", "team_id": 16, "price_tenths": 45,
                        "predicted_points": float(extra_midfield_points), "can_start": True})
    return pd.DataFrame(records)


def exact_state(selling_price=50, bank=0, free=1, chip=None):
    return {"team_id": 10123827, "season_id": "example", "gameweek": 6,
            "observed_at": datetime.now(timezone.utc).isoformat(), "bank_tenths": bank,
            "free_transfers": free, "active_chip": chip,
            "picks": [{"element_id": i, "selling_price_tenths": selling_price} for i in range(1, 16)]}


class SquadSolverTests(unittest.TestCase):
    def setUp(self):
        self.rules = copy.deepcopy(load_config()["squad_rules"])

    def test_matches_independent_enumeration_of_legal_starting_elevens(self):
        candidates = roster()
        result = solve_squad(candidates, self.rules)
        optimal = -np.inf
        for selected in itertools.combinations(range(15), 11):
            rows = candidates.iloc[list(selected)]
            counts = rows.position.value_counts().to_dict()
            if counts.get("GKP") == 1 and 3 <= counts.get("DEF", 0) <= 5 and 2 <= counts.get("MID", 0) <= 5 and 1 <= counts.get("FWD", 0) <= 3:
                optimal = max(optimal, rows.predicted_points.sum() + rows.predicted_points.max())
        self.assertEqual(result["net_expected_points"], optimal)
        self.assertEqual(optimal, 75)
        self.assertEqual(len(result["squad_ids"]), 15)
        self.assertEqual(len(result["starting_xi_ids"]), 11)
        self.assertIn(result["captain_id"], result["starting_xi_ids"])
        self.assertIn(result["vice_captain_id"], result["starting_xi_ids"])
        self.assertNotEqual(result["captain_id"], result["vice_captain_id"])
        self.assertEqual(len(result["bench_outfield_ids"]), 3)
        self.assertEqual(result["market_value_tenths"], 750)

    def test_free_transfer_is_worthwhile_but_the_same_move_with_a_hit_is_not(self):
        candidates = roster(6.5)
        no_free = solve_squad(candidates, self.rules, exact_state(free=0))
        one_free = solve_squad(candidates, self.rules, exact_state(free=1))
        self.assertEqual(no_free["transfers"]["count"], 0)
        self.assertEqual(one_free["transfers"]["in"], [16])
        self.assertEqual(one_free["transfers"]["out"], [12])
        self.assertAlmostEqual(one_free["net_expected_points"], 77.5)
        strong = solve_squad(roster(10), self.rules, exact_state(free=0))
        self.assertEqual(strong["transfers"]["paid_count"], 1)
        self.assertEqual(strong["transfer_hit_points"], 4)
        self.assertGreater(strong["net_expected_points"], no_free["net_expected_points"])
        already_paid = solve_squad(candidates, self.rules, {**exact_state(free=0), "existing_transfer_hit_points": 8})
        self.assertEqual(already_paid["transfer_hit_points"], 8)
        self.assertEqual(already_paid["additional_transfer_hit_points"], 0)
        self.assertEqual(already_paid["net_expected_points"], no_free["net_expected_points"] - 8)

    def test_selling_prices_not_market_value_fund_the_purchase(self):
        rules = {**self.rules, "max_transfers": 1}
        blocked = solve_squad(roster(10), rules, exact_state(selling_price=40, bank=0))
        funded = solve_squad(roster(10), rules, exact_state(selling_price=40, bank=5))
        self.assertEqual(blocked["transfers"]["count"], 0)
        self.assertEqual(funded["transfers"]["in"], [16])
        self.assertEqual(funded["bank_after_tenths"], 0)
        self.assertGreater(funded["market_value_tenths"], funded["budget_tenths"])
        no_changes = solve_squad(roster(100), {**rules, "max_transfers": 0}, exact_state(bank=100))
        self.assertEqual(no_changes["squad_ids"], list(range(1, 16)))

    def test_chips_change_scoring_or_transfer_hits_explicitly(self):
        triple = solve_squad(roster(), self.rules, exact_state(chip="3xc"))
        self.assertEqual(triple["net_expected_points"], 84)
        boost = solve_squad(roster(), self.rules, exact_state(chip="bboost"))
        self.assertEqual(boost["gross_expected_points"], roster().predicted_points.sum() + 9)
        wildcard = solve_squad(roster(6.5), {**self.rules, "max_transfers": 0}, exact_state(free=0, chip="wildcard"))
        self.assertEqual(wildcard["transfers"]["in"], [16])
        self.assertEqual(wildcard["transfer_hit_points"], 0)

    def test_budget_club_and_availability_can_make_a_squad_infeasible(self):
        for candidates, rules in [(roster(), {**self.rules, "budget_tenths": 749}),
                                  (roster().assign(team_id=1), self.rules),
                                  (roster().assign(can_start=False), self.rules)]:
            with self.subTest(rules=rules), self.assertRaises(OptimizationError):
                solve_squad(candidates, rules)
        with self.assertRaises(OptimizationError):
            solve_squad(roster(), self.rules, exact_state(selling_price=51))

    def test_post_execution_display_keeps_saved_squad_even_on_a_wildcard(self):
        frozen = {**self.rules, "max_transfers": 0, "freeze_current_squad": True}
        result = solve_squad(roster(100), frozen, exact_state(chip="wildcard"))
        self.assertEqual(result["squad_ids"], list(range(1, 16)))
        self.assertEqual(result["transfers"]["count"], 0)

    def test_unavailable_new_signing_cannot_be_selected_or_captained(self):
        candidates = roster(100)
        candidates.loc[candidates.element_id == 16, "can_start"] = False
        result = solve_squad(candidates, self.rules)
        self.assertNotIn(16, result["squad_ids"])


class TeamStateTests(unittest.TestCase):
    def test_uuid_stale_state_duplicate_players_and_unknown_allowance_are_rejected(self):
        for field, value in [("team_id", "78f93b29-b203-4ac1-b368-ff9a7666df22"), ("free_transfers", None),
                             ("bank_tenths", -1), ("free_transfers", 6), ("gameweek", 5), ("season_id", "other")]:
            state = exact_state()
            state[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_manager_state(state, "example", 6)
        state = exact_state()
        state["picks"][0] = state["picks"][1]
        with self.assertRaises(ValueError):
            validate_manager_state(state, "example", 6)

    def test_import_calculates_remaining_free_transfers_from_limit_and_made(self):
        payload = {"picks": [{"element": i, "selling_price": 50} for i in range(1, 16)],
                   "transfers": {"limit": 3, "made": 2, "bank": 36}}
        state = import_my_team(payload, 10123827, "example", 6)
        self.assertEqual(state["free_transfers"], 1)
        self.assertEqual(state["bank_tenths"], 36)
        self.assertEqual(len(state["picks"]), 15)
        payload["transfers"]["limit"] = None
        with self.assertRaises(ValueError):
            import_my_team(payload, 10123827, "example", 6)
        self.assertEqual(import_my_team(payload, 10123827, "example", 6, "wildcard")["free_transfers"], 0)

    def test_known_blank_gameweek_is_zero_and_unknown_fixtures_fail(self):
        predictions = {"gameweek": 6, "predictions": [{"element_id": 1, "gameweek": 6, "predicted_points": 20.0}]}
        players = pd.DataFrame({"element_id": [1], "web_name": ["Player"], "position": ["FWD"], "team_id": [1],
                                "price_tenths": [50], "availability_pct": [100], "status": ["a"]})
        features = pd.DataFrame({"element_id": [1], "gameweek": [6], "fixture_context_known": [1], "fixture_count": [0]})
        teams = pd.DataFrame({"team_id": [1], "short_name": ["AAA"]})
        rows = candidate_table(players, predictions, features, teams)
        self.assertEqual(rows.predicted_points.iloc[0], 0)
        self.assertEqual(rows.raw_predicted_points.iloc[0], 20)
        features["fixture_context_known"] = 0
        with self.assertRaises(OptimizationError):
            candidate_table(players, predictions, features, teams)


if __name__ == "__main__":
    unittest.main()

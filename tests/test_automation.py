"""Free-only policy, chip selection, live writes, recovery, and stale-state gates."""

import copy
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from src.common import load_config
from src.automation.apply import execute_plan, journal_path, scheduled_window
from src.automation.client import AuthenticatedFplClient, FplActionError
from src.automation.plan import build_plan, chip_state, choose_chip, validate_forecast
from src.optimizer.manager import import_my_team
from test_optimizer import roster, exact_state


def live_fixture():
    now = datetime.now(timezone.utc)
    season = (now - timedelta(days=35)).isoformat()
    candidates = roster(25)
    bootstrap = {"events": [{"id": 1, "deadline_time": season, "finished": True},
                             {"id": 6, "deadline_time": (now + timedelta(days=1)).isoformat(), "is_next": True}],
                 "teams": [{"id": i, "short_name": f"T{i}"} for i in range(1, 17)],
                 "elements": [{"id": int(r.element_id), "web_name": f"Player {r.element_id}",
                               "element_type": {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}[r.position],
                               "team": int(r.team_id), "now_cost": int(r.price_tenths), "status": "a",
                               "chance_of_playing_next_round": None} for r in candidates.itertuples()]}
    fixtures = [{"event": 6, "team_h": i, "team_a": i+1} for i in range(1, 17, 2)]
    forecast = {"season_id": season, "gameweek": 6, "generated_at": now.isoformat(),
                "source_snapshot_observed_at": now.isoformat(), "training_status": "experimental",
                "predictions": [{"element_id": int(r.element_id), "gameweek": 6, "predicted_points": r.predicted_points}
                                for r in candidates.itertuples()]}
    ids = [1, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 2, 11, 12, 15]
    payload = {"picks": [{"element": i, "position": slot, "selling_price": 50, "purchase_price": 50,
                          "is_captain": i == 13, "is_vice_captain": i == 8} for slot, i in enumerate(ids, 1)],
               "transfers": {"bank": 0, "limit": 1, "made": 0, "cost": 4},
               "chips": [{"name": chip, "status_for_entry": "available", "start_event": 1, "stop_event": 19}
                         for chip in ["wildcard", "freehit", "bboost", "3xc"]]}
    settings = copy.deepcopy(load_config())
    settings["automation"].update(chip_min_gain={c: 999 for c in ["wildcard", "freehit", "bboost", "3xc"]}, receipt_repository=None)
    return bootstrap, fixtures, forecast, payload, settings


class FakeFpl:
    def __init__(self, bootstrap, payload):
        self.bootstrap = copy.deepcopy(bootstrap)
        self.payload = copy.deepcopy(payload)
        self.posts = []
        self.fail_transfer = None
        self.fail_lineup = None
        self.fail_second_bootstrap = False
        self.bootstrap_reads = 0

    def verify_identity(self):
        pass

    def my_team(self):
        return copy.deepcopy(self.payload)

    def request(self, method, route, body=None):
        if method == "GET":
            self.bootstrap_reads += 1
            if self.fail_second_bootstrap and self.bootstrap_reads == 2:
                raise FplActionError("Deadline read unavailable after transfers.")
            return copy.deepcopy(self.bootstrap)
        self.posts.append((route, copy.deepcopy(body)))
        if route == "/transfers/":
            if self.fail_transfer == "before":
                raise FplActionError("No response.")
            for transfer in body["transfers"]:
                pick = next(p for p in self.payload["picks"] if p["element"] == transfer["element_out"])
                pick["element"] = transfer["element_in"]
                pick["selling_price"] = transfer["purchase_price"]
                self.payload["transfers"]["bank"] += transfer["selling_price"] - transfer["purchase_price"]
                self.payload["transfers"]["made"] += 1
            chip = body["chip"]
            if chip:
                self.payload["transfers"]["limit"] = None
            if self.fail_transfer == "after":
                self.set_chip(chip)
                raise FplActionError("Accepted but response lost.")
        else:
            if self.fail_lineup == "before":
                raise FplActionError("No response.")
            old = {p["element"]: p for p in self.payload["picks"]}
            self.payload["picks"] = [{**old[p["element"]], **p} for p in body["picks"]]
            chip = body["chip"]
            if self.fail_lineup == "after":
                self.set_chip(chip)
                raise FplActionError("Accepted but response lost.")
        self.set_chip(chip)
        return {}

    def set_chip(self, chip):
        if chip:
            for item in self.payload["chips"]:
                item["status_for_entry"] = "active" if item["name"] == chip else "unavailable"


class PlanningTests(unittest.TestCase):
    def test_free_only_is_a_hard_cap_even_when_a_paid_transfer_has_high_gain(self):
        _, _, _, _, settings = live_fixture()
        result, _ = choose_chip(roster(100), exact_state(free=0), set(), settings["squad_rules"], settings["automation"])
        self.assertEqual(result["transfers"]["count"], 0)
        self.assertEqual(result["additional_transfer_hit_points"], 0)

    def test_chip_reserve_and_availability_control_selection_and_one_chip_only(self):
        bootstrap, fixtures, forecast, payload, settings = live_fixture()
        saved = copy.deepcopy(payload)
        plan = build_plan(forecast, bootstrap, fixtures, payload, settings)
        self.assertIsNone(plan["chip"])
        settings["automation"]["chip_min_gain"]["3xc"] = 0
        triple = build_plan(forecast, bootstrap, fixtures, payload, settings)
        self.assertEqual(triple["chip"], "3xc")
        self.assertIsNone(triple["transfer_request"]["chip"])
        self.assertEqual(triple["lineup_request"]["chip"], "3xc")
        payload["chips"][3]["status_for_entry"] = "played"
        self.assertIsNone(build_plan(forecast, bootstrap, fixtures, payload, settings)["chip"])
        self.assertEqual(saved["picks"], payload["picks"])

    def test_wildcard_freehit_and_boost_use_the_correct_write_route(self):
        for chip in ["wildcard", "freehit", "bboost"]:
            bootstrap, fixtures, forecast, payload, settings = live_fixture()
            payload["transfers"]["limit"] = 0
            settings["automation"]["allowed_chips"] = [chip]
            settings["automation"]["chip_min_gain"][chip] = 0
            plan = build_plan(forecast, bootstrap, fixtures, payload, settings)
            self.assertEqual(plan["chip"], chip)
            self.assertEqual(plan["transfer_request"]["chip"], chip if chip != "bboost" else None)
            self.assertEqual(plan["lineup_request"]["chip"], chip if chip == "bboost" else None)

    def test_lineup_slots_transfer_prices_and_captain_are_legal(self):
        bootstrap, fixtures, forecast, payload, settings = live_fixture()
        plan = build_plan(forecast, bootstrap, fixtures, payload, settings)
        picks = plan["lineup_request"]["picks"]
        types = {e["id"]: e["element_type"] for e in bootstrap["elements"]}
        self.assertEqual([p["position"] for p in picks], list(range(1, 16)))
        self.assertEqual(types[picks[0]["element"]], 1)
        self.assertEqual(types[picks[11]["element"]], 1)
        self.assertEqual(sum(p["is_captain"] for p in picks[:11]), 1)
        self.assertEqual(sum(p["is_vice_captain"] for p in picks[:11]), 1)
        self.assertEqual(plan["transfer_request"]["transfers"][0]["purchase_price"], 45)
        self.assertEqual(plan["bank_after_tenths"], 5)
        self.assertEqual(plan["additional_hit_points"], 0)

    def test_deadline_wrong_week_old_forecast_old_inputs_and_unknown_chips_stop(self):
        bootstrap, _, forecast, payload, settings = live_fixture()
        policy = settings["automation"]
        for updates in [{"gameweek": 5}, {"season_id": "old"}, {"generated_at": None},
                        {"generated_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
                        {"source_snapshot_observed_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()}]:
            with self.subTest(updates=updates), self.assertRaises(FplActionError):
                validate_forecast({**forecast, **updates}, bootstrap, policy)
        future = datetime.fromisoformat(bootstrap["events"][1]["deadline_time"]) - timedelta(seconds=100)
        with self.assertRaises(FplActionError):
            validate_forecast(forecast, bootstrap, policy, future)
        payload["chips"].append({"name": "unknown", "status_for_entry": "active"})
        with self.assertRaises(FplActionError):
            chip_state(payload, 6)
        self.assertFalse(scheduled_window(settings, bootstrap))
        self.assertTrue(scheduled_window(settings, bootstrap, future - timedelta(minutes=30)))

    def test_transfer_cost_is_per_paid_transfer_and_wildcard_refunds_prior_hits(self):
        _, _, forecast, payload, settings = live_fixture()
        payload["transfers"].update(made=3, limit=1, cost=4)
        state = import_my_team(payload, 10123827, forecast["season_id"], 6)
        self.assertEqual(state["existing_transfer_hit_points"], 8)
        self.assertEqual(state["free_transfers"], 0)
        self.assertEqual(import_my_team(payload, 10123827, forecast["season_id"], 6, "wildcard")["existing_transfer_hit_points"], 0)
        result, _ = choose_chip(roster(100), {**state, "active_chip": "wildcard"}, set(), settings["squad_rules"], settings["automation"])
        self.assertEqual(result["transfer_hit_points"], 0)

    def test_proposed_wildcard_is_committed_and_active_chip_is_preserved(self):
        bootstrap, fixtures, forecast, payload, settings = live_fixture()
        payload["chips"][0]["status_for_entry"] = "proposed"
        proposed = build_plan(forecast, bootstrap, fixtures, payload, settings)
        self.assertEqual(proposed["chip"], "wildcard")
        self.assertIsNone(proposed["source_active_chip"])
        payload["chips"][0]["status_for_entry"] = "available"
        payload["chips"][3]["status_for_entry"] = "active"
        active = build_plan(forecast, bootstrap, fixtures, payload, settings)
        self.assertEqual(active["chip"], "3xc")
        self.assertEqual(active["source_active_chip"], "3xc")

    def test_live_pipeline_requires_refresh_and_refuses_offline_before_authentication(self):
        from src.pipeline import run_end_to_end
        _, _, _, _, settings = live_fixture()
        with patch('src.automation.client.AuthenticatedFplClient.from_settings') as client:
            for options in [{}, {"offline": True}, {"refresh": True, "manager_path": Path('stale.json')}]:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    run_end_to_end(settings, apply_team=True, **options)
            client.assert_not_called()


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        bootstrap, fixtures, forecast, payload, self.settings = live_fixture()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings["automation"]["actions_dir"] = self.temp.name
        self.plan = build_plan(forecast, bootstrap, fixtures, payload, self.settings)
        self.client = FakeFpl(bootstrap, payload)

    def test_live_save_is_verified_and_rerun_never_posts_again(self):
        receipt = execute_plan(self.client, self.plan, self.settings)
        self.assertEqual(receipt["status"], "applied")
        self.assertEqual(len(self.client.posts), 2)
        execute_plan(self.client, self.plan, self.settings)
        self.assertEqual(len(self.client.posts), 2)
        path = journal_path(self.settings, self.plan["season_id"], 6)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_lost_responses_are_recovered_by_get_not_post_retries(self):
        self.client.fail_transfer = "after"
        self.client.fail_lineup = "after"
        self.assertEqual(execute_plan(self.client, self.plan, self.settings)["status"], "applied")
        self.assertEqual(len(self.client.posts), 2)

    def test_unconfirmed_transfer_is_durable_and_never_blindly_retried(self):
        self.client.fail_transfer = "before"
        with self.assertRaises(FplActionError):
            execute_plan(self.client, self.plan, self.settings)
        self.client.fail_transfer = None
        with self.assertRaises(FplActionError):
            execute_plan(self.client, self.plan, self.settings)
        self.assertEqual(len(self.client.posts), 1)

    def test_confirmed_transfers_can_resume_lineup_only(self):
        self.client.fail_second_bootstrap = True
        with self.assertRaises(FplActionError):
            execute_plan(self.client, self.plan, self.settings)
        self.client.fail_second_bootstrap = False
        result = execute_plan(self.client, self.plan, self.settings)
        self.assertEqual(result["status"], "applied")
        self.assertEqual(sum(route == "/transfers/" for route, _ in self.client.posts), 1)

    def test_unconfirmed_lineup_save_does_not_repeat_chip_activation(self):
        self.client.fail_lineup = "before"
        with self.assertRaises(FplActionError):
            execute_plan(self.client, self.plan, self.settings)
        self.client.fail_lineup = None
        with self.assertRaises(FplActionError):
            execute_plan(self.client, self.plan, self.settings)
        self.assertEqual(len(self.client.posts), 2)

    def test_changed_team_and_prices_stop_before_any_post(self):
        self.client.payload["transfers"]["bank"] += 1
        with self.assertRaises(FplActionError):
            execute_plan(self.client, self.plan, self.settings)
        self.client.payload["transfers"]["bank"] -= 1
        self.client.bootstrap["elements"][-1]["now_cost"] += 1
        with self.assertRaises(FplActionError):
            execute_plan(self.client, self.plan, self.settings)
        self.assertEqual(self.client.posts, [])

    def test_all_supported_chips_are_read_back_after_application(self):
        for chip in ["wildcard", "freehit", "bboost", "3xc"]:
            with self.subTest(chip=chip), tempfile.TemporaryDirectory() as directory:
                bootstrap, fixtures, forecast, payload, settings = live_fixture()
                payload["transfers"]["limit"] = 0
                settings["automation"].update(actions_dir=directory, allowed_chips=[chip])
                settings["automation"]["chip_min_gain"][chip] = 0
                plan = build_plan(forecast, bootstrap, fixtures, payload, settings)
                client = FakeFpl(bootstrap, payload)
                self.assertEqual(execute_plan(client, plan, settings)["status"], "applied")
                self.assertEqual([c["name"] for c in client.payload["chips"] if c["status_for_entry"] == "active"], [chip])

    def test_failed_remote_receipt_prevents_fpl_post_and_remote_receipt_wins_over_cache(self):
        self.settings["automation"]["receipt_repository"] = "ArnavJyo/fpl-ai"
        with patch('src.automation.receipts.ReceiptStore') as Store:
            store = Store.return_value
            store.load.return_value = None
            store.save.side_effect = FplActionError("Cannot persist.")
            with self.assertRaises(FplActionError):
                execute_plan(self.client, self.plan, self.settings)
            self.assertEqual(self.client.posts, [])
            store.load.return_value = {"status": "applied", "plan": self.plan, "events": []}
            self.assertEqual(execute_plan(self.client, self.plan, self.settings)["status"], "applied")
            self.assertEqual(self.client.posts, [])


class TransportTests(unittest.TestCase):
    def test_cookie_scope_csrf_and_no_write_retries(self):
        auth = {"team_id": 10123827, "cookies": [{"name": "csrftoken", "value": "test-only", "domain": "fantasy.premierleague.com"}]}
        client = AuthenticatedFplClient(auth, 10123827)
        self.addCleanup(client.close)
        response = Mock(ok=False, is_redirect=False, status_code=503)
        with patch.object(client.session, "request", return_value=response) as request:
            with self.assertRaises(FplActionError):
                client.request("POST", "/transfers/", {})
            request.assert_called_once()
            self.assertEqual(request.call_args.kwargs["headers"]["X-CSRFToken"], "test-only")
        with patch.object(client.session, "request", side_effect=requests.Timeout):
            with self.assertRaises(FplActionError):
                client.request("POST", "/transfers/", {})
        auth["cookies"][0]["domain"] = "google.com"
        with self.assertRaises(FplActionError):
            AuthenticatedFplClient(auth, 10123827)


if __name__ == "__main__":
    unittest.main()

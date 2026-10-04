"""Free-only policy, chip selection, live writes, recovery, and stale-state gates."""

import copy
import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

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
                if not body["chip"]:
                    self.payload["transfers"]["made"] += 1
            chip = body["chip"]
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
    def test_rotating_token_is_persisted_before_an_fpl_request(self):
        auth = {"team_id": 10123827, "access_token": "old-test-token", "refresh_token": "old-test-refresh",
                "client_id": "bfcbaf69-aade-4c1b-8f00-c1cb8a193030", "expires_at": 0, "cookies": []}
        client = AuthenticatedFplClient(auth, 10123827)
        self.addCleanup(client.close)
        vault = Mock()
        client.vault = vault
        response = Mock(ok=True)
        response.json.return_value = {"access_token": "new-test-token", "refresh_token": "new-test-refresh", "expires_in": 1800}
        actual = Mock(ok=True, is_redirect=False, status_code=200, content=b'{}')
        actual.json.return_value = {}
        calls = []
        vault.begin_refresh.side_effect = lambda a: calls.append('reserve')
        vault.save.side_effect = lambda a: calls.append('persist')
        with patch('src.automation.client.requests.post', side_effect=lambda *a, **k: (calls.append('renew'),response)[1]), patch.object(client.session,'request',side_effect=lambda *a,**k: (calls.append('fpl'),actual)[1]):
            client.request('GET','/me/')
        self.assertEqual(calls,['reserve','renew','persist','fpl'])
        self.assertEqual(client.auth['refresh_token'],'new-test-refresh')
        self.assertGreater(client.auth['expires_at'],time.time())
        self.assertEqual(client.session.headers['X-API-Authorization'],'Bearer new-test-token')

    def test_failed_renewal_reservation_prevents_token_and_account_requests(self):
        client = AuthenticatedFplClient({"team_id":10123827,"access_token":"test-token","refresh_token":"test-refresh",
                    "client_id":"bfcbaf69-aade-4c1b-8f00-c1cb8a193030","expires_at":0,"cookies":[]},10123827)
        self.addCleanup(client.close)
        client.vault = Mock()
        client.vault.begin_refresh.side_effect = FplActionError('Concurrent renewal blocked.')
        with patch('src.automation.client.requests.post') as renew, patch.object(client.session,'request') as request:
            with self.assertRaises(FplActionError):client.request('POST','/transfers/',{})
            renew.assert_not_called()
            request.assert_not_called()

    def test_modern_fpl_token_authentication_does_not_require_legacy_csrf_cookie(self):
        client = AuthenticatedFplClient({"team_id": 10123827, "access_token": "unit-test-only", "cookies": []}, 10123827)
        self.addCleanup(client.close)
        self.assertEqual(client.session.headers["X-API-Authorization"], "Bearer unit-test-only")
        self.assertEqual(client.session.headers["X-API-Language"], "en")
        response = Mock(ok=True, is_redirect=False, status_code=200, content=b'{}')
        response.json.return_value = {}
        with patch.object(client.session, "request", return_value=response) as request:
            self.assertEqual(client.request("POST", "/transfers/", {}), {})
            request.assert_called_once()
            self.assertNotIn("X-CSRFToken", request.call_args.kwargs["headers"])
        with self.assertRaises(FplActionError):
            AuthenticatedFplClient({"team_id": 10123827, "access_token": "bad\r\ntoken"}, 10123827)

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


class SessionVaultTests(unittest.TestCase):
    def test_rotating_credentials_are_encrypted_and_authenticated_to_the_entry(self):
        from src.automation.session import SessionVault
        store = Mock()
        vault = SessionVault('ArnavJyo/fpl-ai',10123827,'ab'*32,store=store)
        auth={'team_id':10123827,'access_token':'test-access-secret','refresh_token':'test-refresh-secret','cookies':[]}
        vault.save(auth)
        envelope=store.save_value.call_args.args[0]
        self.assertNotIn('test-access-secret',json.dumps(envelope))
        self.assertNotIn('test-refresh-secret',json.dumps(envelope))
        store.load.return_value=envelope
        self.assertEqual(vault.load(),auth)
        other=SessionVault('ArnavJyo/fpl-ai',999,'ab'*32,store=store)
        with self.assertRaises(FplActionError):other.load()
        envelope['status']='renewing'
        with self.assertRaises(FplActionError):vault.load()


class ExistingProfileTests(unittest.TestCase):
    def test_selected_profile_reads_only_current_fpl_record_and_rejects_deletion(self):
        from src.automation.login import existing_chrome_session
        record=SimpleNamespace(leveldb_seq_number=1,is_live=True,value=json.dumps({
            'access_token':'profile-test-access','refresh_token':'profile-test-refresh','expires_at':time.time()+1800,
            'profile':{'email':'must-not-export@example.test'}}))
        database=Mock()
        database.iter_records_for_script_key.return_value=[record]
        factory=Mock()
        factory.return_value.__enter__=Mock(return_value=database)
        factory.return_value.__exit__=Mock(return_value=False)
        module=SimpleNamespace(ccl_chromium_localstorage=SimpleNamespace(LocalStoreDb=factory))
        with patch.dict('sys.modules',{'ccl_chromium_reader':module}):
            auth=existing_chrome_session('Profile 1',10123827)
            self.assertEqual(auth['access_token'],'profile-test-access')
            self.assertNotIn('profile',auth)
            self.assertNotIn('must-not-export@example.test',json.dumps(auth))
            self.assertEqual(database.iter_records_for_script_key.call_args.args[0],'https://fantasy.premierleague.com')
            database.iter_records_for_script_key.return_value=[record,SimpleNamespace(leveldb_seq_number=2,is_live=False,value=None)]
            with self.assertRaises(FplActionError):existing_chrome_session('Profile 1',10123827)
            with self.assertRaises(FplActionError):existing_chrome_session('../another-profile',10123827)

    def test_reconnecting_an_older_browser_record_preserves_newer_rotated_pair(self):
        from src.automation.login import save_connection
        settings=copy.deepcopy(load_config())
        older={'team_id':10123827,'access_token':'old-access','refresh_token':'old-refresh','expires_at':100}
        newer={**older,'access_token':'new-access','refresh_token':'new-refresh','expires_at':200}
        vault=Mock()
        vault.load.return_value=newer
        with patch('src.automation.session.configured_vault',return_value=vault), patch('src.automation.login.AuthenticatedFplClient') as Client, patch('src.automation.login.private_json') as save:
            Client.return_value.auth=newer
            save_connection(settings,older,False,'ArnavJyo/fpl-ai')
            Client.assert_called_once_with(newer,10123827)
            self.assertEqual(save.call_args.args[1]['refresh_token'],'new-refresh')


if __name__ == "__main__":
    unittest.main()

"""Stage ordering, failure isolation, offline behavior, and forecast timing."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

from src.common import load_config
from src.dashboard.build_dashboard import safe_script_json
from src.dashboard.tracker import archive_directory, archive_forecast, performance_tracker
from src.pipeline import run_end_to_end, runtime_settings


class ForecastTrackerTests(unittest.TestCase):
    def test_deadline_blocks_retrospective_writes_and_unfinished_weeks_are_not_scored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = copy.deepcopy(load_config())
            settings["paths"].update(raw_data_dir=str(root / "raw"), outputs_dir=str(root / "outputs"), features_file=str(root / "features.parquet"))
            (root / "raw").mkdir()
            now = datetime.now(timezone.utc)
            first = (now - timedelta(days=30)).isoformat()
            deadline = now + timedelta(days=1)
            (root / "raw/bootstrap_static.json").write_text(json.dumps({"events": [
                {"id": 1, "deadline_time": first}, {"id": 6, "deadline_time": deadline.isoformat()}]}))
            forecasts = {"season_id": first, "gameweek": 6, "training_status": "experimental",
                         "predictions": [{"element_id": 1, "gameweek": 6, "predicted_points": 2.0},
                                         {"element_id": 2, "gameweek": 6, "predicted_points": 4.0},
                                         {"element_id": 3, "gameweek": 6, "predicted_points": 0.0}]}
            path = archive_forecast(settings, forecasts, now)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                archive_forecast(settings, forecasts, deadline)
            self.assertEqual(path.read_bytes(), before)
            pd.DataFrame({"element_id": [1, 2], "gameweek": [6, 6], "total_points": [1, 6]}).to_parquet(root / "features.parquet")
            self.assertEqual(performance_tracker(settings, first, [1, 2, 3, 4, 5]), [])
            result = performance_tracker(settings, first, [1, 2, 3, 4, 5, 6])
            self.assertEqual(result[0]["scored_rows"], 2)
            self.assertEqual(result[0]["unmatched_rows"], 1)
            self.assertEqual(result[0]["metrics"]["mae"], 1.5)
            self.assertAlmostEqual(result[0]["metrics"]["rmse"], 2.5**0.5)
            self.assertEqual(performance_tracker(settings, "other-season", [6]), [])
            corrupted = json.loads(path.read_text())
            corrupted["generated_at"] = deadline.isoformat()
            path.write_text(json.dumps(corrupted))
            with self.assertRaises(ValueError):
                performance_tracker(settings, first, [6])

    def test_player_text_cannot_escape_embedded_json(self):
        value = {"name": '</script><script>alert("x")</script> & \u2028'}
        encoded = safe_script_json(value)
        self.assertNotIn("</script>", encoded)
        self.assertNotIn("&", encoded)
        self.assertEqual(json.loads(encoded), value)


class EndToEndTests(unittest.TestCase):
    def test_complete_real_code_flow_on_fixture_data_is_offline_and_writes_matching_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = copy.deepcopy(load_config())
            for name, value in settings["paths"].items():
                settings["paths"][name] = str(root / value)
            settings["model"].update(hidden_layers=[8], dropout=0, epochs=3, batch_size=16)
            settings["manager"]["team_id"] = None
            now = datetime.now(timezone.utc)
            season = (now - timedelta(days=35)).isoformat()
            raw = Path(settings["paths"]["raw_data_dir"])
            raw.mkdir(parents=True)
            (raw / "bootstrap_static.json").write_text(json.dumps({"events": [
                {"id": 1, "deadline_time": season}, {"id": 6, "deadline_time": (now + timedelta(days=1)).isoformat()}]}))
            positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
            players = pd.DataFrame({"element_id": range(1, 16), "web_name": ['</script><script>bad()</script>'] + [f"Player {i}" for i in range(2, 16)],
                                    "position": positions, "team_id": range(1, 16), "price_tenths": [50] * 15,
                                    "availability_pct": [100] * 15, "status": ["a"] * 15})
            teams = pd.DataFrame({"team_id": range(1, 16), "short_name": [f"T{i}" for i in range(1, 16)]})
            history = pd.DataFrame([{"element_id": i, "gameweek": week, "total_points": float(i % 4 + week),
                                     "recent_minutes": float(week), "total_points_mean_last_5": float(i % 4)}
                                    for week in range(1, 6) for i in range(1, 16)])
            future = history.loc[history.gameweek == 5].drop(columns="total_points").copy()
            future["gameweek"], future["fixture_context_known"], future["fixture_count"] = 6, 1, 1
            for frame, key in [(players, "players_file"), (teams, "teams_file"), (history, "features_file"), (future, "prediction_features_file")]:
                path = Path(settings["paths"][key]);path.parent.mkdir(parents=True, exist_ok=True);frame.to_parquet(path, index=False)
            manifest = {"season_id": season, "target": "total_points", "finished_gameweeks": [1, 2, 3, 4, 5],
                        "prediction_gameweek": 6, "history_rows": 75, "prediction_rows": 15,
                        "feature_columns": ["recent_minutes", "total_points_mean_last_5"],
                        "baseline_feature_column": "total_points_mean_last_5", "bootstrap_cache_observed_at": now.isoformat()}
            Path(settings["paths"]["feature_manifest_file"]).write_text(json.dumps(manifest))
            with patch.object(requests.Session, "get", side_effect=AssertionError("network used")), redirect_stdout(io.StringIO()):
                report = run_end_to_end(settings, offline=True, allow_experimental=True)
            self.assertEqual(report["status"], "complete")
            squad = json.loads(Path(settings["paths"]["squad_file"]).read_text())
            forecasts = json.loads(Path(settings["paths"]["predictions_file"]).read_text())
            self.assertEqual(len(squad["squad"]), 15)
            self.assertEqual(squad["gameweek"], forecasts["gameweek"])
            self.assertEqual(len(forecasts["predictions"]), 15)
            dashboard = Path(report["dashboard"])
            page = dashboard.read_text()
            self.assertNotIn('</script><script>bad()', page)
            self.assertNotIn("{{DASHBOARD_DATA}}", page)
            data = json.loads((dashboard.parent / "data.json").read_text())
            self.assertEqual(data["squad"], squad)
            self.assertEqual(data["tracker"], [])
            self.assertEqual(len(list(archive_directory(settings, season).glob("*.json"))), 1)

    def test_offline_flow_makes_no_ingestion_calls_and_records_failure_before_downstream(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = copy.deepcopy(load_config())
            settings["paths"]["pipeline_report_file"] = str(Path(temporary) / "run.json")
            forecasts = {"gameweek": 6, "training_status": "experimental"}
            calls = []
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(io.StringIO()))
                ingestion = stack.enter_context(patch("src.pipeline.make_client", side_effect=AssertionError("offline API call")))
                for name in ["run_baseline", "run_training", "archive_forecast", "run_optimization"]:
                    stack.enter_context(patch(f"src.pipeline.{name}", side_effect=lambda *args, _name=name, **kwargs: calls.append(_name)))
                stack.enter_context(patch("src.pipeline.run_prediction", side_effect=lambda *args: calls.append("run_prediction") or forecasts))
                dashboard = stack.enter_context(patch("src.pipeline.run_dashboard", return_value=Path("dashboard/index.html")))
                report = run_end_to_end(settings, offline=True, allow_experimental=True)
                self.assertEqual(report["status"], "complete")
                self.assertEqual(calls, ["run_baseline", "run_training", "run_prediction", "archive_forecast", "run_optimization"])
                ingestion.assert_not_called()
                dashboard.reset_mock()
                with patch("src.pipeline.run_optimization", side_effect=ValueError("infeasible budget")):
                    with self.assertRaisesRegex(ValueError, "infeasible budget"):
                        run_end_to_end(settings, offline=True, allow_experimental=True)
                dashboard.assert_not_called()
            saved = json.loads(Path(settings["paths"]["pipeline_report_file"]).read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["stages"][-1]["name"], "Optimize squad and lineup")
            self.assertEqual(saved["stages"][-1]["status"], "failed")

    def test_skip_training_requires_explicit_experimental_permission_before_forecast_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = copy.deepcopy(load_config())
            settings["paths"].update(latest_model_file=str(root / "latest.json"), pipeline_report_file=str(root / "run.json"))
            (root / "latest.json").write_text(json.dumps({"training_status": "experimental"}))
            with patch("src.pipeline.run_baseline"), patch("src.pipeline.run_prediction") as predict, redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "experimental"):
                    run_end_to_end(settings, offline=True, skip_training=True)
                predict.assert_not_called()
            with self.assertRaises(ValueError):
                run_end_to_end(settings, offline=True, refresh=True)

    def test_numeric_ci_team_id_overrides_settings_without_mutating_config(self):
        original = copy.deepcopy(load_config())
        with patch.dict("os.environ", {"FPL_TEAM_ID": "123456"}):
            self.assertEqual(runtime_settings()["manager"]["team_id"], 123456)
        self.assertEqual(load_config(), original)
        with patch.dict("os.environ", {"FPL_TEAM_ID": "78f93b29-b203-4ac1-b368-ff9a7666df22"}):
            with self.assertRaises(ValueError):
                runtime_settings()


if __name__ == "__main__":
    unittest.main()

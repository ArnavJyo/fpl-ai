"""Regression checks for the data contracts and pipeline orchestration.

Run with: .venv/bin/python -m unittest discover -s tests -v
The cached-data integration test is skipped if local raw/table data is absent.
"""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests
from pandas.testing import assert_frame_equal

from src.common import PROJECT_ROOT, load_config
from src.data.build_history import build_history, unmatched_keys
from src.data.fpl_client import FplClient
from src.data.pipeline import main, run_pipeline


class HistoryContractTests(unittest.TestCase):
    def test_join_matches_player_and_week_and_reports_unmatched_keys(self) -> None:
        stats = pd.DataFrame({
            "element_id": [12, 12, 20], "gameweek": [2, 1, 1], "total_points": [3, 9, 0],
        })
        spine = pd.DataFrame({
            "element_id": [12, 12, 21], "gameweek": [1, 2, 1], "price_tenths": [95, 96, 50],
        })
        history = build_history(stats, spine)
        self.assertEqual(history[["gameweek", "total_points", "price_tenths"]].values.tolist(),
                         [[1, 9, 95], [2, 3, 96]])
        self.assertEqual(unmatched_keys(stats, spine), ({(20, 1)}, {(21, 1)}))

    def test_duplicate_context_is_rejected_before_it_can_multiply_rows(self) -> None:
        stats = pd.DataFrame({"element_id": [12], "gameweek": [1], "total_points": [9]})
        spine = pd.DataFrame({"element_id": [12, 12], "gameweek": [1, 1]})
        with self.assertRaises(pd.errors.MergeError):
            build_history(stats, spine)


class ClientTests(unittest.TestCase):
    def test_existing_cache_never_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            (raw_dir / "sample.json").write_text('{"points": 9}')
            client = FplClient("https://example.invalid/api", raw_dir)
            self.addCleanup(client.session.close)
            with patch.object(client, "_download", side_effect=AssertionError("network used")):
                self.assertEqual(client.get("/sample/", "sample"), {"points": 9})
            self.assertEqual((client.hits, client.misses), (1, 0))

    def test_refresh_replaces_cached_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            (raw_dir / "sample.json").write_text('{"points": 9}')
            client = FplClient("https://example.invalid/api", raw_dir, refresh=True)
            self.addCleanup(client.session.close)
            with patch.object(client, "_download", return_value='{"points": 10}') as download:
                self.assertEqual(client.get("/sample/", "sample"), {"points": 10})
            download.assert_called_once_with("/sample/")
            self.assertEqual(json.loads((raw_dir / "sample.json").read_text()), {"points": 10})

    def test_transient_failure_retries_but_404_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = FplClient("https://example.invalid/api", Path(directory))
            self.addCleanup(client.session.close)
            response = Mock(text='{"ok": true}')
            response.raise_for_status.return_value = None
            with patch.object(client.session, "get", side_effect=[requests.Timeout(), response]) as get:
                with patch("src.data.fpl_client.time.sleep") as sleep, redirect_stdout(io.StringIO()):
                    self.assertEqual(client._download("/sample/"), '{"ok": true}')
                self.assertEqual(get.call_count, 2)
                sleep.assert_called_once_with(1)

            missing = requests.Response()
            missing.status_code = 404
            error = requests.HTTPError(response=missing)
            with patch.object(client.session, "get", side_effect=error) as get:
                with self.assertRaises(requests.HTTPError):
                    client._download("/missing/")
                self.assertEqual(get.call_count, 1)


class PipelineTests(unittest.TestCase):
    def test_failed_stage_stops_downstream_work_and_closes_session(self) -> None:
        client = Mock()
        with patch("src.data.pipeline.make_client", return_value=client), \
             patch("src.data.pipeline.fetch_bootstrap", return_value={}), \
             patch("src.data.pipeline.run_bootstrap"), \
             patch("src.data.pipeline.run_fixtures", side_effect=RuntimeError("bad fixtures")), \
             patch("src.data.pipeline.run_gameweek_stats") as stats, \
             patch("src.data.pipeline.run_spine") as spine, \
             patch("src.data.pipeline.run_history") as history:
            with self.assertRaisesRegex(RuntimeError, "bad fixtures"):
                run_pipeline()
        stats.assert_not_called()
        spine.assert_not_called()
        history.assert_not_called()
        client.session.close.assert_called_once()

    def test_verification_failure_returns_nonzero_cli_status(self) -> None:
        with patch("src.data.pipeline.run_verification", return_value=False):
            with self.assertRaises(SystemExit) as result:
                main(["--stage", "verify"])
        self.assertEqual(result.exception.code, 1)

    def test_partial_gameweeks_require_the_performance_stage(self) -> None:
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                main(["--gameweek", "1"])
        self.assertEqual(result.exception.code, 2)

    @unittest.skipUnless(
        (PROJECT_ROOT / "data/raw/bootstrap_static.json").exists()
        and (PROJECT_ROOT / "data/history.parquet").exists(),
        "requires the local saved API snapshot",
    )
    def test_full_cached_pipeline_preserves_all_six_saved_tables(self) -> None:
        table_names = ["players", "teams", "fixtures", "gw_stats", "spine", "history"]
        expected = {name: pd.read_parquet(PROJECT_ROOT / f"data/{name}.parquet") for name in table_names}
        settings = copy.deepcopy(load_config())
        with tempfile.TemporaryDirectory() as directory:
            for name in table_names:
                settings["paths"][f"{name}_file"] = str(Path(directory) / f"{name}.parquet")
            with patch("src.common.load_config", return_value=settings), \
                 patch.object(requests.Session, "get", side_effect=AssertionError("network used")), \
                 redirect_stdout(io.StringIO()):
                self.assertTrue(run_pipeline())
            for name in table_names:
                with self.subTest(table=name):
                    rebuilt = pd.read_parquet(Path(directory) / f"{name}.parquet")
                    assert_frame_equal(rebuilt, expected[name])


if __name__ == "__main__":
    unittest.main()

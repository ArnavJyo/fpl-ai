"""Feature-context contracts, leakage checks, preprocessing and offline builds."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests
from pandas.testing import assert_frame_equal

from src.common import PROJECT_ROOT, load_config
from src.features.build_features import SNAPSHOT_COLUMNS, assemble_features, build_market_features, player_snapshot
from src.features.context import build_feature_history, build_fixture_features, build_team_results
from src.features.pipeline import load_snapshots, read_fixture_stats, run_features
from src.features.preparation import FeaturePreprocessor, walk_forward_splits


def example_fixtures() -> pd.DataFrame:
    return pd.DataFrame({
        "fixture_id": [1, 2, 3, 4], "gameweek": [1, 2, 3, 3],
        "kickoff_time": pd.to_datetime(["2026-01-01", "2026-01-08", "2026-01-15", "2026-01-18"], utc=True),
        "finished": [True, True, False, False],
        "home_team_id": [1, 2, 1, 3], "away_team_id": [2, 3, 2, 1],
        "home_goals": [1, 0, np.nan, np.nan], "away_goals": [2, 3, np.nan, np.nan],
    })


def example_spine() -> pd.DataFrame:
    return pd.DataFrame({
        "element_id": [11, 22, 22, 33], "gameweek": [1, 1, 2, 2],
        "fixture_id": [1, 1, 2, 2], "own_team_id": [1, 2, 2, 3],
        "price_tenths": [70, 80, 81, 60], "selected": [100, 50, 100, 25],
        "transfers_in": [10, 5, 20, 2], "transfers_out": [0, 1, 2, 0],
    })


def example_players() -> pd.DataFrame:
    return pd.DataFrame({
        "element_id": [11, 22, 33, 44], "team_id": [1, 2, 3, 4],
        "price_tenths": [71, 81, 60, 45], "form": [4.0, 6.0, 2.0, 0.0],
        "selected_by_percent": [10.0, 20.0, 1.0, 0.0], "availability_pct": [100, 75, 0, 100],
        "position": ["FWD", "MID", "DEF", "GKP"], "status": ["a", "d", "i", "u"],
        "is_penalty_taker": [True, False, False, False], "is_corner_taker": [False, True, False, False],
        "is_freekick_taker": [False, False, False, False],
    })


class FixtureContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixtures = example_fixtures()
        self.spine = example_spine()
        self.fixture_stats = self.spine[["element_id", "fixture_id"]].assign(expected_goals=[0.7, 1.8, 0.5, 2.5])

    def test_team_xgc_uses_opponents_xg_once_and_excludes_unfinished_matches(self) -> None:
        results = build_team_results(self.fixtures, self.spine, self.fixture_stats)
        self.assertEqual(len(results), 4)
        team = results[(results.team_id == 1) & (results.gameweek == 1)].iloc[0]
        self.assertEqual(team.xg_for, 0.7)
        self.assertEqual(team.xg_against, 1.8)
        self.assertEqual(team.goals_against, 2)
        incomplete = self.fixture_stats.iloc[1:]
        result = build_team_results(self.fixtures, self.spine, incomplete)
        self.assertTrue(pd.isna(result[(result.team_id == 2) & (result.gameweek == 1)].iloc[0].xg_against))

    def test_double_blank_unknown_club_and_opponent_strength(self) -> None:
        results = build_team_results(self.fixtures, self.spine, self.fixture_stats)
        rows = pd.DataFrame({"element_id": [11, 44, 55], "gameweek": [3, 3, 3], "own_team_id": [1, 4, np.nan]})
        context = build_fixture_features(rows, self.fixtures, results, [3])
        self.assertEqual(context.loc[0, "fixture_count"], 2)
        self.assertEqual(context.loc[0, "home_fixture_count"], 1)
        self.assertEqual(context.loc[0, "away_fixture_count"], 1)
        self.assertEqual(context.loc[0, "is_double"], 1)
        self.assertEqual(context.loc[0, "opponent_goals_against_mean_last_3"], 1)
        self.assertEqual(context.loc[0, "opponent_xg_against_mean_last_3"], 1.05)
        self.assertEqual(context.loc[1, "fixture_count"], 0)
        self.assertEqual(context.loc[1, "is_blank"], 1)
        self.assertTrue(pd.isna(context.loc[2, "fixture_count"]))
        self.assertEqual(context.loc[2, "fixture_context_known"], 0)

    def test_current_and_future_match_results_cannot_change_strength(self) -> None:
        results = build_team_results(self.fixtures, self.spine, self.fixture_stats)
        row = pd.DataFrame({"element_id": [11], "gameweek": [2], "own_team_id": [2]})
        changed = results.copy()
        changed.loc[changed.gameweek >= 2, ["goals_for", "goals_against", "xg_for", "xg_against"]] = 999
        assert_frame_equal(
            build_fixture_features(row, self.fixtures, results, [3]),
            build_fixture_features(row, self.fixtures, changed, [3]),
        )

    def test_feature_history_preserves_blanks_and_does_not_sum_double_week_snapshots(self) -> None:
        stats = pd.DataFrame({"element_id": [11, 11, 11], "gameweek": [1, 2, 3], "total_points": [5, 0, 16]})
        double = pd.DataFrame({
            "element_id": [11, 11], "gameweek": [3, 3], "fixture_id": [3, 4], "own_team_id": [1, 1],
            "price_tenths": [71, 72], "selected": [110, 120], "transfers_in": [4, 6], "transfers_out": [1, 2],
        })
        spine = pd.concat([self.spine[self.spine.element_id == 11], double], ignore_index=True)
        history = build_feature_history(stats, spine, self.fixtures)
        self.assertEqual(history.total_points.tolist(), [5, 0, 16])
        self.assertEqual(history.own_team_id.tolist(), [1, 1, 1])
        self.assertEqual(history.context_observed.tolist(), [1, 0, 1])
        self.assertEqual(history.iloc[2].price_tenths, 72)
        self.assertEqual(history.iloc[2].transfers_in, 6)
        self.assertTrue(pd.isna(history.iloc[1].selected))
        earlier_high_id = self.fixtures.copy()
        earlier_high_id.loc[earlier_high_id.fixture_id == 4, "kickoff_time"] = pd.Timestamp("2026-01-14", tz="UTC")
        self.assertEqual(build_feature_history(stats, spine, earlier_high_id).iloc[2].price_tenths, 71)


class MarketAndAssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixtures = example_fixtures()
        self.spine = example_spine()
        stats = pd.DataFrame({
            "element_id": [11, 11, 22, 22, 33], "gameweek": [1, 2, 1, 2, 2],
            "total_points": [5, 0, 8, 3, 6], "minutes": [90, 0, 90, 20, 90], "starts": [1, 0, 1, 0, 1],
        })
        self.history = build_feature_history(stats, self.spine, self.fixtures)
        fixture_stats = self.spine[["element_id", "fixture_id"]].assign(expected_goals=[0.7, 1.8, 0.5, 2.5])
        self.results = build_team_results(self.fixtures, self.spine, fixture_stats)
        self.players = example_players()

    def test_market_is_lagged_and_normalized_and_carries_age(self) -> None:
        query = pd.DataFrame({"element_id": [11, 11, 22], "gameweek": [1, 3, 2]})
        market = build_market_features(self.history, query)
        self.assertTrue(pd.isna(market.loc[0, "ownership_rank_prior"]))
        self.assertEqual(market.loc[1, "ownership_rank_prior"], 1)
        self.assertEqual(market.loc[1, "transfers_in_rate_prior"], 0.1)
        self.assertEqual(market.loc[1, "market_age_gameweeks"], 2)
        self.assertEqual(market.loc[2, "price_tenths_prior"], 80)
        changed = self.history.copy()
        changed.loc[changed.gameweek == 2, ["selected", "transfers_in", "price_tenths"]] = 999
        assert_frame_equal(market, build_market_features(changed, query))

    def test_training_and_prediction_share_features_without_borrowing_current_attributes(self) -> None:
        snapshots = player_snapshot(self.players, 3)
        train, prediction, columns = assemble_features(
            self.history, self.players, self.fixtures, self.results, snapshots, [3, 5],
            ["total_points", "minutes"], "total_points", 3,
        )
        self.assertEqual(len(train), 5)
        self.assertEqual(len(prediction), 4)
        self.assertEqual(train.drop(columns="total_points").columns.tolist(), prediction.columns.tolist())
        self.assertTrue(train[SNAPSHOT_COLUMNS].isna().all().all())
        forecast = prediction[prediction.element_id == 11].iloc[0]
        self.assertEqual(forecast.total_points_mean_last_3, 2.5)
        self.assertEqual(forecast.appearance_rate_mean_last_3, 0.5)
        self.assertEqual(forecast.position_FWD, 1)
        self.assertEqual(forecast.price_tenths_snapshot, 71)
        new_player = prediction[prediction.element_id == 44].iloc[0]
        self.assertTrue(pd.isna(new_player.total_points_mean_last_3))
        self.assertEqual(new_player.history_count_last_3, 0)
        self.assertEqual(new_player.is_blank, 1)
        self.assertNotIn("ep_next", columns)
        self.assertNotIn("element_id", columns)
        processor = FeaturePreprocessor.fit(train, columns)
        self.assertTrue(set(SNAPSHOT_COLUMNS).issubset(processor.excluded_columns))
        self.assertTrue(np.isfinite(processor.transform(prediction).to_numpy()).all())

    def test_season_end_has_no_prediction_rows(self) -> None:
        empty = pd.DataFrame(columns=["element_id", "gameweek", *SNAPSHOT_COLUMNS])
        train, prediction, _ = assemble_features(
            self.history, self.players, self.fixtures, self.results, empty, [3],
            ["total_points", "minutes"], "total_points", None,
        )
        self.assertEqual(len(train), 5)
        self.assertTrue(prediction.empty)
        with self.assertRaisesRegex(ValueError, "later than all historical"):
            assemble_features(self.history, self.players, self.fixtures, self.results, empty,
                              [3], ["total_points"], "total_points", 2)


class PreparationTests(unittest.TestCase):
    def test_walk_forward_windows_follow_config_and_never_shuffle(self) -> None:
        frame = pd.DataFrame({"gameweek": list(range(1, 15)) * 2})
        splits = walk_forward_splits(frame, 6, 4)
        self.assertEqual(splits, [
            {"train_gameweeks": list(range(1, 7)), "validation_gameweeks": [7, 8, 9, 10]},
            {"train_gameweeks": list(range(1, 11)), "validation_gameweeks": [11, 12, 13, 14]},
        ])
        self.assertEqual(walk_forward_splits(pd.DataFrame({"gameweek": [1, 2, 3, 4, 5]}), 6, 4), [])
        with self.assertRaisesRegex(ValueError, "consecutive"):
            walk_forward_splits(pd.DataFrame({"gameweek": [1, 3, 4]}), 1, 1)

    def test_preprocessing_is_training_only_serializable_and_preserves_unknowns(self) -> None:
        train = pd.DataFrame({"value": [1, 3, np.nan], "constant": [5, 5, 5], "unseen": [np.nan] * 3})
        validation = pd.DataFrame({"value": [1000, np.nan], "constant": [5, 5], "unseen": [8, 9]})
        processor = FeaturePreprocessor.fit(train, ["value", "constant", "unseen"])
        state = processor.to_dict()
        transformed = processor.transform(validation)
        self.assertEqual(processor.medians["value"], 2)
        self.assertEqual(processor.means["value"], 2)
        self.assertEqual(processor.excluded_columns, ["unseen"])
        self.assertEqual(transformed.loc[1, "value"], 0)
        self.assertEqual(transformed.value__missing.tolist(), [0, 1])
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())
        self.assertEqual(processor.to_dict(), state)
        restored = FeaturePreprocessor.from_dict(json.loads(json.dumps(state, allow_nan=False)))
        assert_frame_equal(restored.transform(validation), transformed)

    def test_historical_snapshots_must_precede_deadline(self) -> None:
        bootstrap = {"events": [{"id": 1, "deadline_time": "2026-01-01T10:00:00Z"}]}
        snapshot = player_snapshot(example_players(), 1)
        snapshot["season_id"] = bootstrap["events"][0]["deadline_time"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshots.parquet"
            with patch("src.features.pipeline.path_for", return_value=path):
                snapshot["observed_at"] = "2026-01-01T09:00:00Z"
                snapshot.to_parquet(path, index=False)
                self.assertEqual(len(load_snapshots(bootstrap)), 4)
                snapshot["observed_at"] = "2026-01-01T10:00:00Z"
                snapshot.to_parquet(path, index=False)
                with self.assertRaisesRegex(ValueError, "before its target deadline"):
                    load_snapshots(bootstrap)


class SavedDataFeatureTests(unittest.TestCase):
    @unittest.skipUnless((PROJECT_ROOT / "data/raw/bootstrap_static.json").exists(), "requires saved source data")
    def test_offline_pipeline_outputs_and_future_outcome_independence(self) -> None:
        settings = copy.deepcopy(load_config())
        output_keys = [
            "features_file", "prediction_features_file", "training_matrix_file", "prediction_matrix_file",
            "player_snapshots_file", "feature_preprocessor_file", "feature_manifest_file", "prices_file", "feature_folds_dir",
        ]
        with tempfile.TemporaryDirectory() as directory:
            for key in output_keys:
                settings["paths"][key] = str(Path(directory) / Path(settings["paths"][key]).name)
            with patch("src.features.pipeline.load_config", return_value=settings), \
                 patch("src.common.load_config", return_value=settings), \
                 patch.object(requests.Session, "get", side_effect=AssertionError("network used")), \
                 redirect_stdout(io.StringIO()):
                report = run_features()
                train = pd.read_parquet(settings["paths"]["features_file"])
                prediction = pd.read_parquet(settings["paths"]["prediction_features_file"])
                prepared = pd.read_parquet(settings["paths"]["prediction_matrix_file"])
                processor = FeaturePreprocessor.from_dict(json.loads(Path(settings["paths"]["feature_preprocessor_file"]).read_text()))
                assert_frame_equal(processor.transform(prediction).reset_index(drop=True),
                                   prepared[processor.output_columns].reset_index(drop=True))
                self.assertTrue(np.isfinite(prepared[processor.output_columns].to_numpy()).all())
                self.assertEqual(len(prediction), len(pd.read_parquet(PROJECT_ROOT / "data/players.parquet")))
                self.assertEqual(train.drop(columns=report["target"]).columns.tolist(), prediction.columns.tolist())
                self.assertNotIn(report["target"], processor.columns)
                stats = pd.read_parquet(PROJECT_ROOT / "data/gw_stats.parquet")
                spine = pd.read_parquet(PROJECT_ROOT / "data/spine.parquet")
                fixtures = pd.read_parquet(PROJECT_ROOT / "data/fixtures.parquet")
                players = pd.read_parquet(PROJECT_ROOT / "data/players.parquet")
                changed_stats = stats.copy()
                last_week = max(report["finished_gameweeks"])
                changed_stats.loc[changed_stats.gameweek == last_week, ["total_points", "minutes", "starts"]] = 999
                changed_fixtures = fixtures.copy()
                changed_fixtures.loc[changed_fixtures.gameweek == last_week, ["home_goals", "away_goals"]] = 999
                fixture_stats = read_fixture_stats(PROJECT_ROOT / "data/raw")
                results = build_team_results(changed_fixtures, spine, fixture_stats)
                history = build_feature_history(changed_stats, spine, changed_fixtures)
                snapshots = pd.read_parquet(settings["paths"]["player_snapshots_file"])
                changed, _, columns = assemble_features(
                    history, players, changed_fixtures, results, snapshots,
                    settings["features"]["rolling_windows"], settings["features"]["rolling_stats"],
                    report["target"], report["prediction_gameweek"],
                )
                assert_frame_equal(train[columns], changed[columns])
                run_features()
                assert_frame_equal(train, pd.read_parquet(settings["paths"]["features_file"]))
                settings["model"]["min_gameweeks_for_training"] = 2
                settings["model"]["validation_window_gws"] = 1
                with patch("src.features.pipeline.load_config", return_value=settings):
                    fold_report = run_features()
                self.assertEqual(len(fold_report["folds"]), len(report["finished_gameweeks"]) - 2)
                for fold in fold_report["folds"]:
                    root = Path(settings["paths"]["feature_folds_dir"]) / fold["directory"]
                    fitted = FeaturePreprocessor.from_dict(json.loads((root / "preprocessor.json").read_text()))
                    raw_train = train[train.gameweek.isin(fold["train_gameweeks"])].reset_index(drop=True)
                    raw_validation = train[train.gameweek.isin(fold["validation_gameweeks"])].reset_index(drop=True)
                    self.assertEqual(fitted.to_dict(), FeaturePreprocessor.fit(raw_train, columns).to_dict())
                    stored_validation = pd.read_parquet(root / "validation.parquet")
                    assert_frame_equal(fitted.transform(raw_validation), stored_validation[fitted.output_columns])
                    self.assertLess(max(fold["train_gameweeks"]), min(fold["validation_gameweeks"]))


if __name__ == "__main__":
    unittest.main()

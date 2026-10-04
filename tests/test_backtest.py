"""Chronological backtest isolation, ranking checks, and report artifacts."""

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
import torch

from src.common import load_config
from src.models.backtest import evaluate_backtest, ranking_metrics, run_backtest
from src.models.data import FeatureDataset, prepare_backtest
from src.models.network import predict_array
from src.models.predict import load_checkpoint
from src.models.train import training_options


def dataset_example() -> FeatureDataset:
    """Two players over five weeks, including an unseen training attribute."""
    rows = [{"element_id": player, "gameweek": week, "total_points": player * week,
             "minutes_mean_last_3": float(week - 1),
             "total_points_mean_last_5": float(player * (week - 1)) if week > 1 else np.nan,
             "availability_pct": np.nan if week <= 3 else 100.0}
            for week in range(1, 6) for player in [1, 2]]
    history = pd.DataFrame(rows)
    forecast = history.loc[history.gameweek == 5].drop(columns="total_points").copy()
    forecast["gameweek"] = 6
    return FeatureDataset(history, forecast, {
        "target": "total_points", "season_id": "example", "finished_gameweeks": [1, 2, 3, 4, 5],
        "prediction_gameweek": 6, "history_rows": 10, "prediction_rows": 2,
        "feature_columns": ["minutes_mean_last_3", "total_points_mean_last_5", "availability_pct"],
        "baseline_feature_column": "total_points_mean_last_5",
    })


def small_settings() -> dict:
    settings = copy.deepcopy(load_config())
    settings["model"].update({"hidden_layers": [8], "dropout": 0.2, "epochs": 5,
                              "early_stopping_patience": 2, "batch_size": 4})
    return settings


class BacktestDataTests(unittest.TestCase):
    def test_training_preprocessor_excludes_attributes_observed_in_later_periods(self) -> None:
        prepared = prepare_backtest(dataset_example(), [1, 2, 3], [4], [5])
        self.assertEqual(len(prepared.train), 6)
        self.assertEqual(len(prepared.validation), 2)
        self.assertEqual(len(prepared.test), 2)
        self.assertEqual(prepared.processor.excluded_columns, ["availability_pct"])
        self.assertEqual(prepared.processor.means["minutes_mean_last_3"], 1)
        self.assertEqual(prepared.x_test.shape, (2, 4))
        self.assertEqual(prepared.x_train.dtype, np.float32)
        self.assertEqual(prepared.y_test.shape, (2,))

    def test_invalid_or_missing_periods_are_rejected(self) -> None:
        for periods in [([], [4], [5]), ([1, 2, 2], [4], [5]), ([1, 3], [4], [5]),
                        ([1, 2, 3], [3], [5]), ([1, 2, 3], [5], [4]),
                        ([1, 2, 3], [4], [6]), ([True], [4], [5])]:
            with self.subTest(periods=periods), self.assertRaises(ValueError):
                prepare_backtest(dataset_example(), *periods)

    def test_later_feature_values_cannot_change_training_preprocessing(self) -> None:
        dataset = dataset_example()
        before = prepare_backtest(dataset, [1, 2, 3], [4], [5])
        dataset.history.loc[dataset.history.gameweek >= 4, dataset.feature_columns] = 999
        after = prepare_backtest(dataset, [1, 2, 3], [4], [5])
        self.assertEqual(before.processor.to_dict(), after.processor.to_dict())
        np.testing.assert_array_equal(before.x_train, after.x_train)


class BacktestRankingTests(unittest.TestCase):
    def test_top_players_and_shortfall_are_hand_calculated(self) -> None:
        rows = pd.DataFrame({"element_id": [1, 2, 3, 4], "gameweek": [5] * 4,
                             "total_points": [10, 8, 1, 0], "predicted_points": [0, 5, 6, 3]})
        result = ranking_metrics(rows, "predicted_points", "total_points", 2)
        self.assertEqual(result["selected_element_ids"], [3, 2])
        self.assertEqual(result["actual_points_total"], 9)
        self.assertEqual(result["optimal_actual_points_total"], 18)
        self.assertEqual(result["points_shortfall"], 9)
        self.assertEqual(result["precision_at_k"], 0.5)
        self.assertAlmostEqual(result["spearman_rank_correlation"], -0.4)

    def test_constant_scores_ties_and_small_rosters_are_defined(self) -> None:
        rows = pd.DataFrame({"element_id": [2, 1], "gameweek": [5, 5],
                             "total_points": [2, 2], "predicted_points": [1, 1]})
        result = ranking_metrics(rows, "predicted_points", "total_points", 10)
        self.assertEqual(result["top_k"], 2)
        self.assertEqual(result["selected_element_ids"], [1, 2])
        self.assertEqual(result["points_shortfall"], 0)
        self.assertIsNone(result["spearman_rank_correlation"])
        json.dumps(result, allow_nan=False)


class BacktestRunTests(unittest.TestCase):
    def test_assessment_inputs_and_labels_cannot_change_weights_or_epoch_selection(self) -> None:
        settings = small_settings()
        options = training_options(settings)
        dataset = dataset_example()
        with redirect_stdout(io.StringIO()):
            before = evaluate_backtest(dataset, options, settings["backtest"])
            dataset.history.loc[dataset.history.gameweek == 5, ["total_points", "minutes_mean_last_3"]] = 999
            after = evaluate_backtest(dataset, options, settings["backtest"])
        self.assertEqual(before.report["best_epoch"], after.report["best_epoch"])
        self.assertEqual(before.report["epoch_history"], after.report["epoch_history"])
        self.assertEqual(before.processor.to_dict(), after.processor.to_dict())
        for key, tensor in before.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, after.model.state_dict()[key]))
        self.assertNotEqual(before.report["test"]["nn"], after.report["test"]["nn"])

    def test_artifacts_roundtrip_and_do_not_replace_the_current_model(self) -> None:
        settings = small_settings()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings["paths"]["backtest_dir"] = str(root / "backtest")
            settings["paths"]["players_file"] = str(root / "players.parquet")
            settings["paths"]["latest_model_file"] = str(root / "latest.json")
            settings["paths"]["predictions_file"] = str(root / "production_predictions.json")
            for name in ["latest.json", "production_predictions.json"]:
                (root / name).write_text("keep the current model")
            pd.DataFrame({"element_id": [1, 2], "web_name": ['<script>alert("x")</script>', "Player B"]}).to_parquet(root / "players.parquet")
            with patch("src.models.backtest.load_feature_dataset", return_value=dataset_example()), redirect_stdout(io.StringIO()):
                report = run_backtest(settings)
            output = root / "backtest"
            expected = {"model.pt", "metrics.json", "predictions.csv", "predictions.parquet", "results.md", "report.html",
                        "predicted_vs_actual.png", "learning_curve.png"}
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            self.assertEqual(json.loads((output / "metrics.json").read_text()), report)
            for name in ["latest.json", "production_predictions.json"]:
                self.assertEqual((root / name).read_text(), "keep the current model")
            self.assertEqual(settings["model"]["min_gameweeks_for_training"], 6)
            self.assertEqual(settings["model"]["validation_window_gws"], 4)
            page = (output / "report.html").read_text()
            self.assertNotIn("{{", page)
            self.assertNotIn('<script>alert("x")</script>', page)
            self.assertIn("&lt;script&gt;", page)
            self.assertIn("data:image/png;base64,", page)
            self.assertNotIn('src="https://', page)
            for name in ["predicted_vs_actual.png", "learning_curve.png"]:
                self.assertTrue((output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            model, processor, checkpoint = load_checkpoint(output / "model.pt")
            self.assertEqual(checkpoint["trained_gameweeks"], [1, 2, 3])
            self.assertEqual(checkpoint["selected_epoch"], report["best_epoch"])
            rows = pd.read_parquet(output / "predictions.parquet")
            test = rows.loc[rows.split == "test"]
            prepared = prepare_backtest(dataset_example(), [1, 2, 3], [4], [5])
            np.testing.assert_array_equal(predict_array(model, prepared.x_test), test.predicted_points.to_numpy())


if __name__ == "__main__":
    unittest.main()

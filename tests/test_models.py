"""Model input contracts, temporal isolation, and baseline holdout scoring."""

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
import torch

from src.common import PROJECT_ROOT, load_config
from src.models.baseline import baseline_predictions, evaluate_baseline, regression_metrics, run_baseline
from src.models.data import FeatureDataset, load_feature_dataset, prepare_folds, validate_dataset
from src.models.network import PointsMLP, predict_array
from src.models.predict import load_checkpoint, run_prediction
from src.models.train import ModelNotReadyError, fit_model, run_training, training_options


def example_dataset(weeks: int = 14) -> FeatureDataset:
    history = pd.DataFrame([
        {"element_id": player, "gameweek": week, "total_points": week if player == 11 else 2 * week - 2,
         "recent_minutes": float(week), "total_points_mean_last_5": float(week - 1) if player == 11 and week > 1 else np.nan,
         "availability_pct": np.nan if week <= 10 else 100.0}
        for week in range(1, weeks + 1) for player in [11, 22]
    ])
    prediction = pd.DataFrame([
        {"element_id": player, "gameweek": weeks + 1, "recent_minutes": float(weeks),
         "total_points_mean_last_5": float(weeks), "availability_pct": 100.0}
        for player in [11, 22]
    ])
    return FeatureDataset(history, prediction, {
        "season_id": "example", "target": "total_points",
        "feature_columns": ["recent_minutes", "total_points_mean_last_5", "availability_pct"],
        "baseline_feature_column": "total_points_mean_last_5",
        "finished_gameweeks": list(range(1, weeks + 1)), "prediction_gameweek": weeks + 1,
        "history_rows": len(history), "prediction_rows": len(prediction),
    })


class ModelDataTests(unittest.TestCase):
    def test_complete_folds_keep_whole_weeks_and_use_train_only_schema(self) -> None:
        folds = prepare_folds(example_dataset(), 6, 4)
        self.assertEqual(len(folds), 2)
        self.assertEqual(folds[0].train_gameweeks, list(range(1, 7)))
        self.assertEqual(folds[0].validation_gameweeks, [7, 8, 9, 10])
        self.assertEqual(folds[1].train_gameweeks, list(range(1, 11)))
        self.assertEqual(folds[1].validation_gameweeks, [11, 12, 13, 14])
        for fold in folds:
            self.assertEqual(fold.processor.excluded_columns, ["availability_pct"])
            self.assertEqual(fold.x_train.shape, (len(fold.train), 4))
            self.assertEqual(fold.x_validation.shape, (len(fold.validation), 4))
            self.assertEqual(fold.y_train.shape, (len(fold.train),))
            self.assertEqual(fold.x_train.dtype, np.float32)
            self.assertTrue(np.isfinite(fold.x_validation).all())
            self.assertLess(max(fold.train.gameweek), min(fold.validation.gameweek))

    def test_validation_and_prediction_values_cannot_change_training_transform(self) -> None:
        dataset = example_dataset()
        before = prepare_folds(dataset, 6, 4)[0]
        dataset.history.loc[dataset.history.gameweek >= 7, ["recent_minutes", "total_points", "availability_pct"]] = 999
        dataset.prediction[dataset.feature_columns] = 9999
        after = prepare_folds(dataset, 6, 4)[0]
        self.assertEqual(before.processor.to_dict(), after.processor.to_dict())
        np.testing.assert_array_equal(before.x_train, after.x_train)
        np.testing.assert_array_equal(before.y_train, after.y_train)

    def test_invalid_artifacts_are_rejected(self) -> None:
        for error in ["duplicate", "target", "input", "count", "forecast", "infinite"]:
            with self.subTest(error=error):
                dataset = example_dataset()
                if error == "duplicate":
                    dataset.history.loc[1, ["element_id", "gameweek"]] = dataset.history.loc[0, ["element_id", "gameweek"]].to_numpy()
                elif error == "target":
                    dataset.history.loc[0, "total_points"] = np.nan
                elif error == "input":
                    dataset.manifest["feature_columns"].append("element_id")
                elif error == "count":
                    dataset.manifest["history_rows"] += 1
                elif error == "forecast":
                    dataset.prediction["total_points"] = 0
                else:
                    dataset.history.loc[0, "recent_minutes"] = np.inf
                with self.assertRaises(ValueError):
                    validate_dataset(dataset)

    def test_incomplete_and_nonconsecutive_weeks_do_not_form_folds(self) -> None:
        self.assertEqual(prepare_folds(example_dataset(9), 6, 4), [])
        dataset = example_dataset()
        dataset.history = dataset.history.loc[dataset.history.gameweek != 3]
        dataset.manifest["finished_gameweeks"].remove(3)
        dataset.manifest["history_rows"] = len(dataset.history)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            prepare_folds(dataset, 6, 4)


class BaselineTests(unittest.TestCase):
    def test_baseline_preserves_raw_points_and_uses_training_only_fallback(self) -> None:
        train = pd.DataFrame({"total_points": [-2, 6]})
        rows = pd.DataFrame({"total_points_mean_last_5": [np.nan, -1, 10], "total_points": [999, 999, 999]})
        predicted, fallback = baseline_predictions(train, rows, "total_points", "total_points_mean_last_5")
        self.assertEqual(fallback, 2)
        self.assertEqual(predicted.tolist(), [2, -1, 10])

    def test_metrics_are_hand_calculated_and_reject_broadcasting(self) -> None:
        metrics = regression_metrics(np.array([0, 2]), np.array([1, -1]))
        self.assertEqual(metrics["mae"], 2)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(5))
        self.assertEqual(metrics["mean_error"], -1)
        for actual, predicted in [([], []), ([1], [[1]]), ([np.nan], [1])]:
            with self.assertRaises(ValueError):
                regression_metrics(np.array(actual), np.array(predicted))

    def test_report_scores_only_holdouts_and_aggregates_actual_row_errors(self) -> None:
        report, rows = evaluate_baseline(example_dataset(), 6, 4)
        self.assertEqual(report["status"], "evaluated")
        self.assertEqual(len(rows), 16)
        self.assertEqual(sorted(rows.gameweek.unique()), list(range(7, 15)))
        self.assertEqual(len(report["per_gameweek"]), 8)
        self.assertEqual(report["folds"][0]["fallback_points"], 4.25)
        self.assertEqual(report["folds"][0]["fallback_rows"], 4)
        self.assertEqual(report["metrics"], regression_metrics(rows.total_points.to_numpy(), rows.predicted_points.to_numpy()))
        _, earlier = evaluate_baseline(example_dataset(10), 6, 4)
        pd.testing.assert_frame_equal(rows[rows.fold == 1].reset_index(drop=True), earlier)

    def test_insufficient_history_has_no_scores_or_fake_predictions(self) -> None:
        report, rows = evaluate_baseline(example_dataset(5), 6, 4)
        self.assertEqual(report["status"], "awaiting_data")
        self.assertEqual(report["additional_finished_gameweeks_needed"], 5)
        self.assertIsNone(report["metrics"])
        self.assertEqual(report["folds"], [])
        self.assertTrue(rows.empty)
        json.dumps(report, allow_nan=False)


class SavedModelDataTests(unittest.TestCase):
    @unittest.skipUnless((PROJECT_ROOT / "data/feature_manifest.json").exists(), "requires saved features")
    def test_saved_artifacts_load_offline_and_report_without_changing_inputs(self) -> None:
        settings = copy.deepcopy(load_config())
        with tempfile.TemporaryDirectory() as directory:
            settings["paths"]["baseline_metrics_file"] = str(Path(directory) / "metrics.json")
            settings["paths"]["baseline_validation_file"] = str(Path(directory) / "validation.parquet")
            # A subsequent awaiting-data run must replace stale holdout predictions.
            pd.DataFrame({"stale": [1]}).to_parquet(settings["paths"]["baseline_validation_file"])
            with patch.object(requests.Session, "get", side_effect=AssertionError("network used")), redirect_stdout(io.StringIO()):
                dataset = load_feature_dataset(settings)
                report = run_baseline(settings)
            self.assertEqual(len(dataset.history), dataset.manifest["history_rows"])
            stored = json.loads(Path(settings["paths"]["baseline_metrics_file"]).read_text())
            self.assertEqual(stored, report)
            rows = pd.read_parquet(settings["paths"]["baseline_validation_file"])
            self.assertIn("predicted_points", rows.columns)
            if report["status"] == "awaiting_data":
                self.assertTrue(rows.empty)
                self.assertIsNone(report["metrics"])


def model_settings(directory: str) -> dict:
    settings = copy.deepcopy(load_config())
    root = Path(directory)
    settings["paths"].update({
        "model_dir": str(root / "models"), "model_metrics_file": str(root / "metrics.json"),
        "model_validation_file": str(root / "validation.parquet"), "latest_model_file": str(root / "latest.json"),
        "predictions_file": str(root / "predictions.json"),
    })
    settings["model"].update({"hidden_layers": [8], "dropout": 0.2, "epochs": 5,
                              "early_stopping_patience": 2, "batch_size": 8, "learning_rate": 0.01})
    return settings


class NetworkTests(unittest.TestCase):
    def test_dynamic_width_and_single_row_output_keep_target_shape(self) -> None:
        for width in [4, 110]:
            model = PointsMLP(width, [8, 4], 0.2)
            self.assertEqual(model(torch.zeros(1, width)).shape, (1,))
            self.assertEqual(model(torch.zeros(3, width)).shape, (3,))
            self.assertEqual(model(torch.zeros(0, width)).shape, (0,))
            with self.assertRaises(ValueError):
                model(torch.zeros(3, width + 1))

    def test_output_can_be_negative_and_inference_disables_dropout(self) -> None:
        model = PointsMLP(4, [8], 0.9)
        with torch.no_grad():
            model.layers[-1].weight.zero_()
            model.layers[-1].bias.fill_(-2)
        inputs = np.ones((3, 4), dtype=np.float32)
        np.testing.assert_array_equal(predict_array(model, inputs), [-2, -2, -2])
        np.testing.assert_array_equal(predict_array(model, inputs), predict_array(model, inputs))
        self.assertFalse(model.training)

    def test_optimization_learns_a_numeric_relationship_reproducibly(self) -> None:
        options = training_options(load_config())
        options.update({"hidden_layers": [8], "dropout": 0.0, "epochs": 80,
                        "learning_rate": 0.03, "loss": "mse", "batch_size": 32})
        inputs = np.linspace(-1, 1, 64, dtype=np.float32).reshape(-1, 1)
        targets = 2 * inputs[:, 0] + 1
        first = fit_model(inputs, targets, options)
        self.assertLess(first.history[-1]["training_loss"], first.history[0]["training_loss"] * 0.1)
        second = fit_model(inputs, targets, options)
        np.testing.assert_array_equal(predict_array(first.model, inputs), predict_array(second.model, inputs))


class ModelTrainingTests(unittest.TestCase):
    def test_default_training_refuses_incomplete_folds_without_writing_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = model_settings(directory)
            with patch("src.models.train.load_feature_dataset", return_value=example_dataset(5)):
                with self.assertRaises(ModelNotReadyError):
                    run_training(settings)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_experimental_real_input_flow_roundtrips_weights_schema_and_predictions(self) -> None:
        dataset = example_dataset(5)
        with tempfile.TemporaryDirectory() as directory:
            settings = model_settings(directory)
            with patch("src.models.train.load_feature_dataset", return_value=dataset), redirect_stdout(io.StringIO()):
                report = run_training(settings, experimental=True)
            self.assertEqual(report["training_status"], "experimental")
            self.assertIsNone(report["validation_metrics"])
            self.assertIsNone(report["baseline_comparison"])
            self.assertEqual(report["refit_epochs"], 5)
            self.assertTrue(pd.read_parquet(settings["paths"]["model_validation_file"]).empty)
            model, processor, checkpoint = load_checkpoint(Path(report["checkpoint"]))
            self.assertEqual(model.input_size, 4)
            self.assertEqual(processor.excluded_columns, ["availability_pct"])
            expected = predict_array(model, processor.transform(dataset.prediction).to_numpy(dtype=np.float32))
            with patch("src.models.predict.load_feature_dataset", return_value=dataset), redirect_stdout(io.StringIO()):
                result = run_prediction(settings)
                repeated = run_prediction(settings)
            self.assertLessEqual(result["generated_at"], repeated["generated_at"])
            self.assertEqual({**result, "generated_at": repeated["generated_at"]}, repeated)
            self.assertEqual(result["training_status"], "experimental")
            self.assertEqual(result["gameweek"], 6)
            self.assertEqual([row["element_id"] for row in result["predictions"]], [11, 22])
            np.testing.assert_array_equal([row["predicted_points"] for row in result["predictions"]], expected)
            self.assertEqual(json.loads(Path(settings["paths"]["predictions_file"]).read_text()), repeated)
            wrong_season = copy.deepcopy(dataset)
            wrong_season.manifest["season_id"] = "different"
            with patch("src.models.predict.load_feature_dataset", return_value=wrong_season):
                with self.assertRaisesRegex(ValueError, "season or target"):
                    run_prediction(settings)
            checkpoint["input_columns"] = ["wrong"]
            torch.save(checkpoint, report["checkpoint"])
            with self.assertRaisesRegex(ValueError, "schemas differ"):
                load_checkpoint(Path(report["checkpoint"]))

    def test_evaluated_folds_score_matching_rows_and_restore_best_epoch(self) -> None:
        dataset = example_dataset()
        with tempfile.TemporaryDirectory() as directory:
            settings = model_settings(directory)
            with patch("src.models.train.load_feature_dataset", return_value=dataset), redirect_stdout(io.StringIO()):
                report = run_training(settings)
            self.assertEqual(report["training_status"], "evaluated")
            self.assertEqual(len(report["folds"]), 2)
            for fold in report["folds"]:
                self.assertLess(max(fold["train_gameweeks"]), min(fold["validation_gameweeks"]))
                self.assertEqual(fold["excluded_unobserved_columns"], ["availability_pct"])
                best = next(row for row in fold["epoch_history"] if row["epoch"] == fold["best_epoch"])
                self.assertEqual(fold["validation_metrics"], best["validation_metrics"])
            expected_epochs = max(1, int(np.median([fold["best_epoch"] for fold in report["folds"]])))
            self.assertEqual(report["refit_epochs"], expected_epochs)
            rows = pd.read_parquet(settings["paths"]["model_validation_file"])
            self.assertEqual(sorted(rows.gameweek.unique()), list(range(7, 15)))
            self.assertEqual(len(rows), 16)
            self.assertEqual(report["validation_metrics"], regression_metrics(rows.total_points.to_numpy(), rows.predicted_points.to_numpy()))
            self.assertEqual(report["baseline_metrics"], regression_metrics(rows.total_points.to_numpy(), rows.baseline_points.to_numpy()))
            # The final fit may use a snapshot observed after all validation training windows.
            _, processor, _ = load_checkpoint(Path(report["checkpoint"]))
            self.assertIn("availability_pct", processor.columns)

    def test_early_stopping_does_not_keep_the_last_worse_weights(self) -> None:
        options = training_options(load_config())
        options.update({"hidden_layers": [], "dropout": 0.0, "epochs": 30, "batch_size": 4,
                        "learning_rate": 0.1, "loss": "mse", "early_stopping_patience": 2})
        inputs = np.ones((4, 1), dtype=np.float32)
        targets = np.full(4, 10, dtype=np.float32)
        validation_targets = np.full(4, -10, dtype=np.float32)
        result = fit_model(inputs, targets, options, validation=(inputs, validation_targets))
        self.assertEqual(result.best_epoch, 1)
        self.assertEqual(result.epochs_run, 3)
        self.assertEqual(regression_metrics(validation_targets, predict_array(result.model, inputs)),
                         result.history[0]["validation_metrics"])


if __name__ == "__main__":
    unittest.main()

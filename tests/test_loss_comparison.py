"""Metric selection, paired-seed aggregation, and loss-experiment isolation."""

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

from test_backtest import dataset_example, small_settings
from src.models.compare_losses import evaluate_loss_comparison, run_loss_comparison, summarize_runs
from src.models.data import prepare_backtest
from src.models.network import predict_array
from src.models.predict import load_checkpoint
from src.models.train import fit_model, training_options


class LossSelectionTests(unittest.TestCase):
    def test_rmse_and_default_mae_can_select_different_epochs_and_restore_weights(self):
        options = training_options(small_settings())
        options.update(dropout=0, epochs=3, early_stopping_patience=1)
        x, y = np.arange(8, dtype=np.float32).reshape(-1, 1), np.arange(8, dtype=np.float32)
        scores = [{"mae": 1.0, "rmse": 3.0, "mean_error": 0.0},
                  {"mae": 2.0, "rmse": 2.0, "mean_error": 0.0},
                  {"mae": 3.0, "rmse": 4.0, "mean_error": 0.0}]
        with patch("src.models.train.regression_metrics", side_effect=scores):
            mae = fit_model(x, y, options, validation=(x, y))
        with patch("src.models.train.regression_metrics", side_effect=scores):
            rmse = fit_model(x, y, options, validation=(x, y), selection_metric="rmse")
        self.assertEqual((mae.best_epoch, rmse.best_epoch), (1, 2))
        for result in [mae, rmse]:
            reference = fit_model(x, y, options, epochs=result.best_epoch)
            np.testing.assert_array_equal(predict_array(result.model, x), predict_array(reference.model, x))
        with self.assertRaises(ValueError):
            fit_model(x, y, options, selection_metric="accuracy")

    def test_select_loss_by_all_seed_means_rather_than_the_best_single_run(self):
        runs = []
        for loss, values in [("huber", [1.0, 3.0]), ("mse", [1.9, 1.9])]:
            for seed, value in zip([7, 42], values):
                runs.append({"loss": loss, "seed": seed, "summary_metrics": dict.fromkeys(
                    ["rmse", "mae", "mean_error", "top_k_actual_points", "prediction_std"], value)})
        result = summarize_runs(runs, ["huber", "mse"], [7, 42])
        self.assertEqual(result["preferred_loss"], "mse")
        self.assertEqual(result["by_loss"]["huber"]["rmse"], {"mean": 2.0, "std": 1.0})
        self.assertAlmostEqual(result["paired_differences"][0]["mse_minus_huber_rmse"], 0.9)
        runs[2]["summary_metrics"]["rmse"] = runs[3]["summary_metrics"]["rmse"] = 2.0
        self.assertIsNone(summarize_runs(runs, ["huber", "mse"], [7, 42])["preferred_loss"])


class LossExperimentTests(unittest.TestCase):
    def test_pairs_share_options_and_test_changes_cannot_affect_any_result(self):
        settings = small_settings()
        options = training_options(settings)
        with redirect_stdout(io.StringIO()):
            before = evaluate_loss_comparison(dataset_example(), options, settings["backtest"], settings["loss_comparison"])
            altered = dataset_example()
            altered.history.loc[altered.history.gameweek == 5, ["total_points", "minutes_mean_last_3"]] = 999
            altered.prediction[altered.feature_columns] = 999
            after = evaluate_loss_comparison(altered, options, settings["backtest"], settings["loss_comparison"])
        self.assertEqual(before.report, after.report)
        pd.testing.assert_frame_equal(before.predictions, after.predictions)
        self.assertEqual(before.processor.to_dict(), after.processor.to_dict())
        self.assertEqual(len(before.fits), 6)
        self.assertEqual(before.report["validation_rows"], 2)
        self.assertEqual(before.report["selection_metric"], "validation_rmse")
        for run in before.report["runs"]:
            changed = {key: value for key, value in run["options"].items() if key not in ["loss", "seed"]}
            self.assertEqual(changed, {key: value for key, value in options.items() if key not in ["loss", "seed"]})
            for key, tensor in before.fits[run["run_id"]].model.state_dict().items():
                self.assertTrue(torch.equal(tensor, after.fits[run["run_id"]].model.state_dict()[key]))

    def test_invalid_experiment_is_rejected_before_training(self):
        settings = small_settings()
        for experiment in [{"losses": ["huber", "huber"], "seeds": [7, 42]},
                           {"losses": ["huber", "mse"], "seeds": [7, 7]},
                           {"losses": ["huber", "mse"], "seeds": [True, 42]}]:
            with self.subTest(experiment=experiment), self.assertRaises(ValueError):
                evaluate_loss_comparison(dataset_example(), training_options(settings), settings["backtest"], experiment)

    def test_report_and_checkpoints_roundtrip_without_model_promotion(self):
        settings = small_settings()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings["loss_comparison"]["output_dir"] = str(root / "comparison")
            settings["paths"].update(players_file=str(root / "players.parquet"), latest_model_file=str(root / "latest.json"),
                                      predictions_file=str(root / "predictions.json"))
            pd.DataFrame({"element_id": [1, 2], "web_name": ["<script>bad()</script>", "Player B"]}).to_parquet(root / "players.parquet")
            for name in ["latest.json", "predictions.json"]:
                (root / name).write_text("keep")
            with patch("src.models.compare_losses.load_feature_dataset", return_value=dataset_example()), redirect_stdout(io.StringIO()):
                report = run_loss_comparison(settings)
            output = root / "comparison"
            self.assertEqual(json.loads((output / "metrics.json").read_text()), report)
            self.assertEqual(len(list(output.glob("*.pt"))), 6)
            self.assertEqual(len(pd.read_csv(output / "runs.csv")), 6)
            for name in ["latest.json", "predictions.json"]:
                self.assertEqual((root / name).read_text(), "keep")
            page = (output / "report.html").read_text()
            self.assertIn("&lt;script&gt;", page)
            self.assertNotIn("<script>bad()", page)
            self.assertNotIn("{{", page)
            self.assertIn("data:image/png;base64,", page)
            self.assertTrue((output / "comparison.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            rows = pd.read_parquet(output / "predictions.parquet")
            prepared = prepare_backtest(dataset_example(), [1, 2, 3], [4], [5])
            for run in report["runs"]:
                model, processor, checkpoint = load_checkpoint(Path(run["checkpoint"]))
                self.assertEqual(checkpoint["selected_epoch"], run["best_epoch"])
                self.assertEqual(checkpoint["selection_metric"], "validation_rmse")
                self.assertEqual(processor.to_dict(), json.loads((output / "preprocessor.json").read_text()))
                np.testing.assert_array_equal(predict_array(model, prepared.x_validation), rows.loc[rows.run_id == run["run_id"], "predicted_points"].to_numpy())


if __name__ == "__main__":
    unittest.main()

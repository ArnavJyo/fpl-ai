"""Paired validation experiment: python -m src.models.compare_losses.

Change only Huber versus MSE, repeat fixed seeds, and select by mean validation
RMSE. Saves separate artifacts; never promotes a model or scores GW5. Read
docs/MODEL_LOSS_COMPARISON.md alongside this module.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.common import PROJECT_ROOT, ensure_parent, load_config
from src.features.preparation import FeaturePreprocessor
from src.models.backtest import attach_player_names, score_period
from src.models.baseline import baseline_predictions
from src.models.data import KEY, FeatureDataset, load_feature_dataset, prepare_backtest
from src.models.network import predict_array
from src.models.train import FitResult, fit_model, save_json, training_options


@dataclass
class LossComparisonResult:
    report: dict
    predictions: pd.DataFrame
    processor: FeaturePreprocessor
    fits: dict[str, FitResult]


def spread(values: np.ndarray) -> dict:
    """Describe prediction range; std uses ddof=0, not uncertainty in a mean."""
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0)),
            "min": float(values.min()), "max": float(values.max()),
            "p90": float(np.quantile(values, 0.9))}


def summarize_runs(runs: list[dict], losses: list[str], seeds: list[int]) -> dict:
    """Average independent runs, never choose a lucky seed or average weights."""
    summaries = {}
    for loss in losses:
        group = [run for run in runs if run["loss"] == loss]
        summaries[loss] = {}
        for metric in ["rmse", "mae", "mean_error", "top_k_actual_points", "prediction_std"]:
            values = np.array([run["summary_metrics"][metric] for run in group], dtype=float)
            summaries[loss][metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    # A numerical tie is reported honestly rather than broken by seed luck.
    first, second = losses
    first_rmse, second_rmse = [summaries[loss]["rmse"]["mean"] for loss in losses]
    preferred = None if np.isclose(first_rmse, second_rmse, rtol=0, atol=1e-8) else (
        first if first_rmse < second_rmse else second)
    paired = []
    for seed in seeds:
        scores = {run["loss"]: run["summary_metrics"]["rmse"] for run in runs if run["seed"] == seed}
        paired.append({"seed": seed, "mse_minus_huber_rmse": scores["mse"] - scores["huber"]})
    return {"by_loss": summaries, "preferred_loss": preferred,
            "selection_rule": "Lowest mean GW-validation RMSE across every configured seed; no best-seed selection.",
            "paired_differences": paired, "std_definition": "Population standard deviation across seed runs (ddof=0), not a confidence interval."}


def evaluate_loss_comparison(dataset: FeatureDataset, options: dict, split: dict, experiment: dict) -> LossComparisonResult:
    """Fit one training schema, then six fresh models with paired seeds.

    The existing chronological splitter also prepares test arrays; those arrays
    are never passed to a model, scored, or used by summarize_runs here.
    """
    losses, seeds = experiment["losses"], experiment["seeds"]
    if not isinstance(losses, list) or len(losses) != 2 or set(losses) != {"huber", "mse"}:
        raise ValueError("Compare exactly huber and mse once each.")
    if (not isinstance(seeds, list) or len(seeds) < 2 or
            any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ValueError("Use at least two distinct integer seeds in [0, 2**32).")
    top_k = split["ranking_top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("ranking_top_k must be a positive integer.")
    prepared = prepare_backtest(dataset, split["train_gameweeks"], split["validation_gameweeks"], split["test_gameweeks"])
    frame = prepared.validation
    baseline, fallback = baseline_predictions(prepared.train, frame, dataset.target,
                                               dataset.manifest["baseline_feature_column"])
    runs, predictions, fits = [], [], {}
    for loss in losses:
        for seed in seeds:
            run_id = f"{loss}_seed{seed}"
            run_options = {**copy.deepcopy(options), "loss": loss, "seed": seed}
            fitted = fit_model(prepared.x_train, prepared.y_train, run_options,
                               validation=(prepared.x_validation, prepared.y_validation),
                               selection_metric="rmse", progress=run_id)
            rows = frame[[*KEY, dataset.target]].copy()
            rows["predicted_points"] = predict_array(fitted.model, prepared.x_validation)
            rows["baseline_points"] = baseline
            rows["run_id"], rows["loss"], rows["seed"] = run_id, loss, seed
            rows["nn_error"] = rows.predicted_points - rows[dataset.target]
            scores = score_period(rows, dataset.target, top_k)
            prediction_spread = spread(rows.predicted_points.to_numpy())
            # Average per-week top-k totals if the split later includes >1 week.
            top_points = float(np.mean([week["ranking"]["nn"]["actual_points_total"] for week in scores["per_gameweek"]]))
            runs.append({"run_id": run_id, "loss": loss, "seed": seed, "options": run_options,
                         "best_epoch": fitted.best_epoch, "epochs_run": fitted.epochs_run,
                         "epoch_history": fitted.history, "validation": scores,
                         "prediction_spread": prediction_spread,
                         "summary_metrics": {**scores["nn"], "top_k_actual_points": top_points,
                                             "prediction_std": prediction_spread["std"]}})
            predictions.append(rows)
            fits[run_id] = fitted
    baseline_scores = runs[0]["validation"]
    report = {
        "status": "exploratory_loss_comparison", "season_id": dataset.manifest["season_id"],
        "target": dataset.target, "train_gameweeks": prepared.train_gameweeks,
        "validation_gameweeks": prepared.validation_gameweeks, "unscored_gameweeks": prepared.test_gameweeks,
        "train_rows": len(prepared.train), "validation_rows": len(frame),
        "base_options": copy.deepcopy(options), "losses": losses, "seeds": seeds,
        "selection_metric": "validation_rmse", "minimum_improvement": 0.0001,
        "input_columns": prepared.processor.output_columns,
        "excluded_unobserved_columns": prepared.processor.excluded_columns,
        "baseline_feature_column": dataset.manifest["baseline_feature_column"],
        "baseline_fallback_points": fallback, "ranking_top_k": top_k,
        "baseline": {"metrics": baseline_scores["baseline"],
                     "per_gameweek": [{"gameweek": week["gameweek"], "ranking": week["ranking"]["baseline"]}
                                      for week in baseline_scores["per_gameweek"]],
                     "prediction_spread": spread(baseline)},
        "actual_spread": spread(prepared.y_validation), "runs": runs,
        "comparison": summarize_runs(runs, losses, seeds),
        "historical_schedule_assumption": dataset.manifest.get("historical_schedule_assumption"),
        "limitations": [
            "The same validation week selects epochs and the preferred loss; this is development evidence, not an untouched test score.",
            "Three seeds measure sensitivity to initialization and training randomness on this split, not generalization across gameweeks.",
            "MAE, prediction spread, and top-k actual points are diagnostics; only mean validation RMSE selects the preferred loss.",
            "Huber delta and architecture are fixed. This compares these settings, not every possible Huber or MSE model.",
            "No GW5 assessment, GW6 forecast, refit, or automatic model promotion is performed. Reserve later weeks for final evaluation.",
            "Top-k rankings ignore positions, budget, and club constraints; they are not valid squads.",
        ],
    }
    return LossComparisonResult(report, pd.concat(predictions, ignore_index=True), prepared.processor, fits)


def run_loss_comparison(settings: dict | None = None) -> dict:
    """Write an isolated, reproducible comparison; preserve current outputs."""
    settings = load_config() if settings is None else settings
    dataset = load_feature_dataset(settings)
    result = evaluate_loss_comparison(dataset, training_options(settings), settings["backtest"], settings["loss_comparison"])
    directory = PROJECT_ROOT / settings["loss_comparison"]["output_dir"]
    ensure_parent(directory / "metrics.json")
    report = result.report
    result.predictions = attach_player_names(result.predictions, settings)
    for run in report["runs"]:
        checkpoint_path = directory / f"model_{run['run_id']}.pt"
        torch.save({"format_version": 1, "model_state_dict": result.fits[run["run_id"]].model.state_dict(),
                    "model_options": run["options"], "input_size": len(result.processor.output_columns),
                    "input_columns": result.processor.output_columns, "preprocessor": result.processor.to_dict(),
                    "target": dataset.target, "season_id": report["season_id"],
                    "trained_gameweeks": report["train_gameweeks"], "training_status": report["status"],
                    "validation_gameweeks": report["validation_gameweeks"], "selected_epoch": run["best_epoch"],
                    "selection_metric": report["selection_metric"], "validation_metrics": run["validation"]["nn"],
                    "baseline_comparison": {f"beats_baseline_{metric}": run["validation"]["nn"][metric] < report["baseline"]["metrics"][metric]
                                            for metric in ["mae", "rmse"]}}, checkpoint_path)
        run["checkpoint"] = str(checkpoint_path)
    save_json(directory / "preprocessor.json", result.processor.to_dict())
    result.predictions.to_csv(directory / "predictions.csv", index=False)
    result.predictions.to_parquet(directory / "predictions.parquet", index=False)
    pd.DataFrame([{ "run_id": run["run_id"], "loss": run["loss"], "seed": run["seed"],
                    "best_epoch": run["best_epoch"], "epochs_run": run["epochs_run"], **run["summary_metrics"]}
                  for run in report["runs"]]).to_csv(directory / "runs.csv", index=False)
    from src.models.loss_report import write_loss_report

    write_loss_report(report, result.predictions, directory)
    save_json(directory / "metrics.json", report)
    print(f"Preferred loss by mean validation RMSE: {report['comparison']['preferred_loss']} (exploratory only)")
    print(f"Report: {directory / 'report.html'}")
    return report


if __name__ == "__main__":
    run_loss_comparison()

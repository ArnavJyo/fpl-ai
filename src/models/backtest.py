"""Exploratory historical assessment: python -m src.models.backtest.

Train a fresh model on GW1-3, select its epoch on GW4, then assess GW5.
All artifacts go to backtest_dir; production weights and predictions are untouched.
See docs/MODEL_BACKTEST.md for the code-reading guide and metric definitions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.common import PROJECT_ROOT, ensure_parent, load_config
from src.features.preparation import FeaturePreprocessor
from src.models.baseline import baseline_predictions, regression_metrics
from src.models.data import KEY, FeatureDataset, load_feature_dataset, prepare_backtest
from src.models.network import PointsMLP, predict_array
from src.models.train import fit_model, save_json, training_options


@dataclass
class BacktestResult:
    report: dict
    predictions: pd.DataFrame
    model: PointsMLP
    processor: FeaturePreprocessor


def ranking_metrics(rows: pd.DataFrame, score: str, target: str, top_k: int) -> dict:
    """Assess one week's ranking; tied scores are ordered by player ID."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1 or rows.empty:
        raise ValueError("Ranking needs rows and a positive integer top_k.")
    if rows.gameweek.nunique() != 1 or rows.element_id.duplicated().any():
        raise ValueError("Rank each gameweek separately with unique player IDs.")
    if not np.isfinite(rows[[score, target]].to_numpy(dtype=float)).all():
        raise ValueError("Ranking scores and targets must be finite.")
    count = min(top_k, len(rows))
    selected = rows.sort_values([score, "element_id"], ascending=[False, True]).head(count)
    ideal = rows.sort_values([target, "element_id"], ascending=[False, True]).head(count)
    selected_ids, ideal_ids = set(selected.element_id), set(ideal.element_id)
    actual_total, ideal_total = float(selected[target].sum()), float(ideal[target].sum())
    predicted_ranks, actual_ranks = rows[score].rank(), rows[target].rank()
    correlation = (float(predicted_ranks.corr(actual_ranks))
                   if predicted_ranks.nunique() > 1 and actual_ranks.nunique() > 1 else None)
    return {
        "top_k": count, "selected_element_ids": [int(player) for player in selected.element_id],
        "actual_points_total": actual_total, "actual_points_mean": actual_total / count,
        "optimal_actual_points_total": ideal_total, "points_shortfall": ideal_total - actual_total,
        "overlap_with_actual_top_k": len(selected_ids & ideal_ids),
        "precision_at_k": len(selected_ids & ideal_ids) / count, "spearman_rank_correlation": correlation,
    }


def score_period(rows: pd.DataFrame, target: str, top_k: int) -> dict:
    """Use identical player rows for NN and baseline errors and rankings."""
    actual = rows[target].to_numpy()
    return {
        "rows": len(rows), "nn": regression_metrics(actual, rows.predicted_points.to_numpy()),
        "baseline": regression_metrics(actual, rows.baseline_points.to_numpy()),
        "per_gameweek": [
            {"gameweek": int(week), "rows": len(group),
             "nn": regression_metrics(group[target].to_numpy(), group.predicted_points.to_numpy()),
             "baseline": regression_metrics(group[target].to_numpy(), group.baseline_points.to_numpy()),
             "ranking": {"nn": ranking_metrics(group, "predicted_points", target, top_k),
                         "baseline": ranking_metrics(group, "baseline_points", target, top_k)}}
            for week, group in rows.groupby("gameweek", sort=True)
        ],
    }


def evaluate_backtest(dataset: FeatureDataset, options: dict, split_options: dict) -> BacktestResult:
    """Keep test inputs and outcomes outside the entire optimization loop."""
    prepared = prepare_backtest(dataset, split_options["train_gameweeks"],
                                split_options["validation_gameweeks"], split_options["test_gameweeks"])
    top_k = split_options["ranking_top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("ranking_top_k must be a positive integer.")
    # Only training and validation arrays are passed to fit_model. No all-history
    # refit: the saved backtest weights must never learn from validation or test.
    fitted = fit_model(prepared.x_train, prepared.y_train, options,
                       validation=(prepared.x_validation, prepared.y_validation), progress="Exploratory backtest")
    predictions, fallback = [], None
    for name, frame, inputs in [("validation", prepared.validation, prepared.x_validation),
                                ("test", prepared.test, prepared.x_test)]:
        rows = frame[[*KEY, dataset.target]].copy()
        rows["predicted_points"] = predict_array(fitted.model, inputs)
        baseline, fallback = baseline_predictions(prepared.train, frame, dataset.target,
                                                   dataset.manifest["baseline_feature_column"])
        rows["baseline_points"] = baseline
        rows["baseline_used_fallback"] = frame[dataset.manifest["baseline_feature_column"]].isna()
        rows["split"] = name
        rows["nn_error"] = rows.predicted_points - rows[dataset.target]
        rows["baseline_error"] = rows.baseline_points - rows[dataset.target]
        predictions.append(rows)
    combined = pd.concat(predictions, ignore_index=True)
    validation = score_period(combined.loc[combined.split == "validation"], dataset.target, top_k)
    test = score_period(combined.loc[combined.split == "test"], dataset.target, top_k)
    report = {
        "status": "exploratory_backtest", "season_id": dataset.manifest["season_id"], "target": dataset.target,
        "train_gameweeks": prepared.train_gameweeks, "validation_gameweeks": prepared.validation_gameweeks,
        "test_gameweeks": prepared.test_gameweeks, "train_rows": len(prepared.train),
        "options": options, "input_columns": prepared.processor.output_columns,
        "excluded_unobserved_columns": prepared.processor.excluded_columns,
        "selection_metric": "validation_mae", "best_epoch": fitted.best_epoch, "epochs_run": fitted.epochs_run,
        "epoch_history": fitted.history, "baseline_feature_column": dataset.manifest["baseline_feature_column"],
        "baseline_fallback_points": fallback, "ranking_top_k": top_k,
        "validation": validation, "test": test,
        "test_comparison": {
            f"beats_baseline_{metric}": test["nn"][metric] < test["baseline"][metric] for metric in ["mae", "rmse"]
        },
        "historical_schedule_assumption": dataset.manifest.get("historical_schedule_assumption"),
        "limitations": [
            "One small historical split and one seed; not evidence of stable future performance.",
            "GW5 is excluded from fitting and epoch selection in this run, but these weeks were already available during project development.",
            "Repeated changes based on test results turn this period into development data; reserve later weeks for final evaluation.",
            "Past-only raw features may use outcomes from earlier validation/test weeks for subsequent one-week predictions.",
            "Top-k ranking ignores squad positions, prices, and club constraints; it is not a squad recommendation.",
        ],
    }
    return BacktestResult(report, combined, fitted.model, prepared.processor)


def attach_player_names(rows: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Current names are display labels only; they never enter model inputs."""
    result = rows.copy()
    path = PROJECT_ROOT / settings["paths"]["players_file"]
    names = {}
    if path.exists():
        players = pd.read_parquet(path, columns=["element_id", "web_name"])
        if players.element_id.duplicated().any():
            raise ValueError("Player display labels must have unique IDs.")
        names = players.set_index("element_id").web_name.to_dict()
    result["player_name"] = result.element_id.map(names).fillna(result.element_id.astype(str))
    return result


def run_backtest(settings: dict | None = None) -> dict:
    settings = load_config() if settings is None else settings
    dataset = load_feature_dataset(settings)
    result = evaluate_backtest(dataset, training_options(settings), settings["backtest"])
    result.predictions = attach_player_names(result.predictions, settings)
    directory = PROJECT_ROOT / settings["paths"]["backtest_dir"]
    checkpoint_path = ensure_parent(directory / "model.pt")
    report = result.report
    checkpoint = {
        "format_version": 1, "model_state_dict": result.model.state_dict(), "model_options": report["options"],
        "input_size": len(result.processor.output_columns), "input_columns": result.processor.output_columns,
        "preprocessor": result.processor.to_dict(), "target": dataset.target, "season_id": report["season_id"],
        "trained_gameweeks": report["train_gameweeks"], "training_status": report["status"],
        "validation_gameweeks": report["validation_gameweeks"], "test_gameweeks": report["test_gameweeks"],
        "selected_epoch": report["best_epoch"], "validation_metrics": report["validation"]["nn"],
        "baseline_comparison": report["test_comparison"],
    }
    torch.save(checkpoint, checkpoint_path)
    result.predictions.to_parquet(directory / "predictions.parquet", index=False)
    result.predictions.to_csv(directory / "predictions.csv", index=False)
    report["checkpoint"] = str(checkpoint_path)
    # Presentation is kept separate from fitting and metrics. It cannot change
    # selected weights; see backtest_report.py and docs/MODEL_BACKTEST.md.
    from src.models.backtest_report import write_backtest_report

    write_backtest_report(report, result.predictions, directory)
    save_json(directory / "metrics.json", report)
    for metric in ["mae", "rmse"]:
        print(f"Test {metric.upper()}: NN={report['test']['nn'][metric]:.3f}, baseline={report['test']['baseline'][metric]:.3f}")
    print(f"Selected epoch {report['best_epoch']} on validation; assessed GW{report['test_gameweeks']} without refitting.")
    print(f"Backtest report: {directory / 'report.html'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_backtest()


if __name__ == "__main__":
    main()

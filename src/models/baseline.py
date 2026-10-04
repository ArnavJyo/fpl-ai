"""Evaluate the rolling-points baseline: python -m src.models.baseline."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common import PROJECT_ROOT, ensure_parent, load_config
from src.models.data import KEY, FeatureDataset, load_feature_dataset, prepare_folds


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Metrics need nonempty, matching one-dimensional arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics need finite actual and predicted values.")
    errors = predicted - actual
    return {"mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.square(errors).mean())),
            "mean_error": float(errors.mean())}


def baseline_predictions(train: pd.DataFrame, rows: pd.DataFrame, target: str, feature: str) -> tuple[pd.Series, float]:
    """Use raw prior points, falling back to the training-set target mean."""
    if train.empty or not np.isfinite(train[target].to_numpy(dtype=float)).all():
        raise ValueError("The baseline needs finite training targets.")
    fallback = float(train[target].mean())
    values = rows[feature].astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("The baseline feature must not contain infinity.")
    return values.fillna(fallback), fallback


def evaluate_baseline(dataset: FeatureDataset, min_train_gameweeks: int, validation_window_gws: int) -> tuple[dict, pd.DataFrame]:
    """Report holdout scores only when complete configured folds exist."""
    folds = prepare_folds(dataset, min_train_gameweeks, validation_window_gws)
    feature = dataset.manifest["baseline_feature_column"]
    if feature not in dataset.feature_columns:
        raise ValueError("The baseline column must be a declared raw feature.")
    weeks = dataset.manifest["finished_gameweeks"]
    report = {
        "model": "rolling_points_mean", "season_id": dataset.manifest["season_id"],
        "target": dataset.target, "baseline_feature_column": feature,
        "fallback_strategy": "training_target_mean", "status": "evaluated" if folds else "awaiting_data",
        "finished_gameweeks": weeks, "history_rows": len(dataset.history),
        "prediction_gameweek": dataset.manifest["prediction_gameweek"],
        "prediction_input_rows": len(dataset.prediction),
        "min_train_gameweeks": min_train_gameweeks, "validation_window_gws": validation_window_gws,
        "minimum_finished_gameweeks": min_train_gameweeks + validation_window_gws,
        "additional_finished_gameweeks_needed": max(0, min_train_gameweeks + validation_window_gws - len(weeks)),
        "historical_schedule_assumption": dataset.manifest.get("historical_schedule_assumption"),
        "folds": [], "metrics": None, "per_gameweek": [],
    }
    predictions = []
    for number, fold in enumerate(folds, start=1):
        predicted, fallback = baseline_predictions(fold.train, fold.validation, dataset.target, feature)
        rows = fold.validation[[*KEY, dataset.target]].copy()
        rows["predicted_points"] = predicted
        rows["used_fallback"] = fold.validation[feature].isna()
        rows["fold"] = number
        predictions.append(rows)
        report["folds"].append({
            "fold": number, "train_gameweeks": fold.train_gameweeks, "validation_gameweeks": fold.validation_gameweeks,
            "train_rows": len(fold.train), "validation_rows": len(fold.validation),
            "input_columns": fold.processor.output_columns,
            "excluded_unobserved_columns": fold.processor.excluded_columns,
            "fallback_points": fallback, "fallback_rows": int(rows.used_fallback.sum()),
            "metrics": regression_metrics(rows[dataset.target].to_numpy(), rows.predicted_points.to_numpy()),
        })
    columns = [*KEY, dataset.target, "predicted_points", "used_fallback", "fold"]
    held_out = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=columns)
    if not held_out.empty:
        report["metrics"] = regression_metrics(held_out[dataset.target].to_numpy(), held_out.predicted_points.to_numpy())
        report["per_gameweek"] = [
            {"gameweek": int(week), "rows": len(rows),
             **regression_metrics(rows[dataset.target].to_numpy(), rows.predicted_points.to_numpy())}
            for week, rows in held_out.groupby("gameweek", sort=True)
        ]
    return report, held_out


def run_baseline(settings: dict | None = None) -> dict:
    settings = load_config() if settings is None else settings
    dataset = load_feature_dataset(settings)
    model = settings["model"]
    report, predictions = evaluate_baseline(dataset, model["min_gameweeks_for_training"], model["validation_window_gws"])
    paths = settings["paths"]
    # Write the empty table as well, so a previous run cannot leave stale scores.
    predictions.to_parquet(ensure_parent(PROJECT_ROOT / paths["baseline_validation_file"]), index=False)
    ensure_parent(PROJECT_ROOT / paths["baseline_metrics_file"]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Historical rows: {len(dataset.history):,}; finished GWs: {dataset.manifest['finished_gameweeks']}")
    print(f"Upcoming GW{report['prediction_gameweek']} inputs: {len(dataset.prediction):,} rows (not points predictions)")
    if report["status"] == "awaiting_data":
        print(f"No evaluation yet: need {report['minimum_finished_gameweeks']} finished GWs "
              f"({report['min_train_gameweeks']} training + {report['validation_window_gws']} validation); "
              f"{report['additional_finished_gameweeks_needed']} more required.")
    else:
        print(f"Baseline: {len(report['folds'])} folds; MAE={report['metrics']['mae']:.3f}; RMSE={report['metrics']['rmse']:.3f}")
    print(f"Report: {paths['baseline_metrics_file']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_baseline()


if __name__ == "__main__":
    main()

"""Train on real features: python -m src.models.train [--experimental]."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.common import PROJECT_ROOT, ensure_parent, load_config
from src.features.preparation import FeaturePreprocessor
from src.models.baseline import baseline_predictions, regression_metrics
from src.models.data import KEY, load_feature_dataset, prepare_folds
from src.models.network import PointsMLP, predict_array


class ModelNotReadyError(ValueError):
    """The configured training/validation periods are not available yet."""


@dataclass
class FitResult:
    model: PointsMLP
    best_epoch: int
    epochs_run: int
    history: list[dict]


def validate_options(options: dict) -> None:
    if options["type"] != "mlp" or options.get("predict_horizon_gws", 1) != 1:
        raise ValueError("This trainer supports an MLP predicting one gameweek.")
    for key in ["batch_size", "epochs", "early_stopping_patience", "num_threads"]:
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ["learning_rate", "huber_delta"]:
        if not np.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if not np.isfinite(options["weight_decay"]) or options["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative.")
    if options["loss"] not in {"huber", "mse"}:
        raise ValueError("Supported regression losses are huber and mse.")
    # Validate the architecture before any training or file writes.
    PointsMLP(1, options["hidden_layers"], options["dropout"])


def training_options(settings: dict) -> dict:
    options = {"seed": 42, "num_threads": 1, "weight_decay": 0.0001, "huber_delta": 1.0,
               **settings["model"]}
    validate_options(options)
    return options


def fit_model(
    x_train: np.ndarray, y_train: np.ndarray, options: dict,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
    epochs: int | None = None, progress: str | None = None,
    selection_metric: str = "mae",
) -> FitResult:
    """Learn weights; restore the best validation epoch (MAE by default).

    Loss controls gradient updates; selection_metric independently chooses the
    saved epoch. The loss experiment uses RMSE for both Huber and MSE runs.
    """
    if selection_metric not in {"mae", "rmse"}:
        raise ValueError("selection_metric must be mae or rmse.")
    validate_options(options)
    if x_train.ndim != 2 or len(x_train) == 0 or y_train.shape != (len(x_train),):
        raise ValueError("Training needs a nonempty feature matrix and one target per row.")
    inputs = torch.as_tensor(x_train, dtype=torch.float32)
    targets = torch.as_tensor(y_train, dtype=torch.float32)
    if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
        raise ValueError("Training values must be finite in float32.")
    if validation is not None:
        x_val, y_val = validation
        if x_val.ndim != 2 or x_val.shape[1] != x_train.shape[1] or len(x_val) == 0 or y_val.shape != (len(x_val),):
            raise ValueError("Validation needs matching feature width and one target per row.")
        if not np.isfinite(x_val).all() or not np.isfinite(y_val).all():
            raise ValueError("Validation values must be finite.")
    limit = options["epochs"] if epochs is None else epochs
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("The epoch limit must be a positive integer.")
    torch.set_num_threads(options["num_threads"])
    torch.manual_seed(options["seed"])
    model = PointsMLP(x_train.shape[1], options["hidden_layers"], options["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    loss_function = nn.HuberLoss(delta=options["huber_delta"]) if options["loss"] == "huber" else nn.MSELoss()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=options["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(options["seed"]), num_workers=0)
    history, best_state = [], None
    best_score, best_epoch, stale_epochs = float("inf"), 0, 0
    for epoch in range(1, limit + 1):
        model.train()
        total_loss = 0.0
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch_inputs)
            loss = loss_function(predicted, batch_targets)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became nonfinite.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_inputs)
        record = {"epoch": epoch, "training_loss": total_loss / len(inputs)}
        if validation is not None:
            metrics = regression_metrics(y_val, predict_array(model, x_val))
            record["validation_metrics"] = metrics
            if metrics[selection_metric] < best_score - 0.0001:
                best_score, best_epoch, stale_epochs = metrics[selection_metric], epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale_epochs += 1
        else:
            best_epoch = epoch
        history.append(record)
        if progress and (epoch == 1 or epoch % 10 == 0 or epoch == limit):
            message = f"{progress}: epoch {epoch}/{limit}; training loss={record['training_loss']:.4f}"
            if validation is not None:
                message += f"; validation {selection_metric.upper()}={metrics[selection_metric]:.4f}"
            print(message, flush=True)
        if validation is not None and stale_epochs >= options["early_stopping_patience"]:
            if progress:
                print(f"{progress}: early stopping; restoring epoch {best_epoch}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return FitResult(model, best_epoch, len(history), history)


def save_json(path: Path, value: dict) -> None:
    ensure_parent(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_training(settings: dict | None = None, experimental: bool = False) -> dict:
    """Evaluate complete folds, then refit all history with its own preprocessor."""
    settings = load_config() if settings is None else settings
    options = training_options(settings)
    dataset = load_feature_dataset(settings)
    folds = prepare_folds(dataset, options["min_gameweeks_for_training"], options["validation_window_gws"])
    weeks = dataset.manifest["finished_gameweeks"]
    if not folds and not experimental:
        minimum = options["min_gameweeks_for_training"] + options["validation_window_gws"]
        raise ModelNotReadyError(
            f"Have {len(weeks)} finished GWs; evaluation needs {minimum} "
            f"({options['min_gameweeks_for_training']} training + {options['validation_window_gws']} validation). "
            "Use --experimental to fit available real history without validation."
        )
    status = "evaluated" if folds else "experimental"
    report = {
        "model": "mlp", "training_status": status, "season_id": dataset.manifest["season_id"],
        "target": dataset.target, "trained_gameweeks": weeks, "history_rows": len(dataset.history),
        "prediction_gameweek": dataset.manifest["prediction_gameweek"], "options": options,
        "selection_metric": "validation_mae", "folds": [], "validation_metrics": None,
        "baseline_metrics": None, "baseline_comparison": None, "per_gameweek": [],
        "historical_schedule_assumption": dataset.manifest.get("historical_schedule_assumption"),
        "evaluation_scope": "One-gameweek predictions using past-only inputs; validation selects epochs. No untouched test period evaluated.",
    }
    validation_rows = []
    for number, fold in enumerate(folds, start=1):
        result = fit_model(fold.x_train, fold.y_train, options,
                           validation=(fold.x_validation, fold.y_validation), progress=f"Fold {number}")
        predicted = predict_array(result.model, fold.x_validation)
        baseline, fallback = baseline_predictions(fold.train, fold.validation, dataset.target,
                                                   dataset.manifest["baseline_feature_column"])
        rows = fold.validation[[*KEY, dataset.target]].copy()
        rows["predicted_points"], rows["baseline_points"], rows["fold"] = predicted, baseline, number
        validation_rows.append(rows)
        report["folds"].append({
            "fold": number, "train_gameweeks": fold.train_gameweeks, "validation_gameweeks": fold.validation_gameweeks,
            "train_rows": len(fold.train), "validation_rows": len(fold.validation),
            "input_columns": fold.processor.output_columns, "excluded_unobserved_columns": fold.processor.excluded_columns,
            "best_epoch": result.best_epoch, "epochs_run": result.epochs_run, "epoch_history": result.history,
            "validation_metrics": regression_metrics(fold.y_validation, predicted),
            "baseline_metrics": regression_metrics(fold.y_validation, baseline.to_numpy()), "baseline_fallback_points": fallback,
        })
    held_out = (pd.concat(validation_rows, ignore_index=True) if validation_rows else
                pd.DataFrame(columns=[*KEY, dataset.target, "predicted_points", "baseline_points", "fold"]))
    if not held_out.empty:
        actual = held_out[dataset.target].to_numpy()
        report["validation_metrics"] = regression_metrics(actual, held_out.predicted_points.to_numpy())
        report["baseline_metrics"] = regression_metrics(actual, held_out.baseline_points.to_numpy())
        report["baseline_comparison"] = {
            f"beats_baseline_{metric}": report["validation_metrics"][metric] < report["baseline_metrics"][metric]
            for metric in ["mae", "rmse"]
        }
        report["per_gameweek"] = [
            {"gameweek": int(week), "rows": len(rows),
             "nn": regression_metrics(rows[dataset.target].to_numpy(), rows.predicted_points.to_numpy()),
             "baseline": regression_metrics(rows[dataset.target].to_numpy(), rows.baseline_points.to_numpy())}
            for week, rows in held_out.groupby("gameweek", sort=True)
        ]
        refit_epochs = max(1, int(np.median([fold["best_epoch"] for fold in report["folds"]])))
    else:
        refit_epochs = options["epochs"]
        print("Experimental fit: real historical rows, fixed epochs, no validation or baseline comparison.", flush=True)
    # The evaluation preprocessors are never reused for the all-history refit.
    processor = FeaturePreprocessor.fit(dataset.history, dataset.feature_columns)
    x_train = processor.transform(dataset.history).to_numpy(dtype=np.float32)
    y_train = dataset.history[dataset.target].to_numpy(dtype=np.float32)
    final = fit_model(x_train, y_train, options, epochs=refit_epochs, progress="All-history fit")
    report.update({"refit_epochs": refit_epochs, "input_columns": processor.output_columns,
                   "excluded_unobserved_columns": processor.excluded_columns, "epoch_history": final.history,
                   "training_metrics": regression_metrics(y_train, predict_array(final.model, x_train))})
    paths = settings["paths"]
    suffix = "_experimental" if status == "experimental" else ""
    checkpoint_path = PROJECT_ROOT / paths["model_dir"] / f"model_gw{max(weeks)}{suffix}.pt"
    report["checkpoint"] = str(checkpoint_path)
    checkpoint = {
        "format_version": 1, "model_state_dict": final.model.state_dict(), "model_options": options,
        "input_size": len(processor.output_columns), "input_columns": processor.output_columns,
        "preprocessor": processor.to_dict(), "target": dataset.target, "season_id": report["season_id"],
        "trained_gameweeks": weeks, "training_status": status, "refit_epochs": refit_epochs,
        "validation_metrics": report["validation_metrics"], "baseline_comparison": report["baseline_comparison"],
        "torch_version": str(torch.__version__), "created_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint, ensure_parent(checkpoint_path))
    held_out.to_parquet(ensure_parent(PROJECT_ROOT / paths["model_validation_file"]), index=False)
    save_json(PROJECT_ROOT / paths["model_metrics_file"], report)
    # A repository-relative pointer survives downloaded CI artifacts and clones.
    pointer_path = str(checkpoint_path.relative_to(PROJECT_ROOT)) if checkpoint_path.is_relative_to(PROJECT_ROOT) else str(checkpoint_path)
    save_json(PROJECT_ROOT / paths["latest_model_file"], {"checkpoint": pointer_path, "training_status": status})
    print(f"Saved {status} model: {checkpoint_path}", flush=True)
    if report["validation_metrics"] is not None:
        print(f"Validation MAE: NN={report['validation_metrics']['mae']:.3f}, baseline={report['baseline_metrics']['mae']:.3f}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimental", action="store_true", help="allow fixed-epoch real-data fitting before complete validation folds exist")
    args = parser.parse_args()
    try:
        run_training(experimental=args.experimental)
    except ModelNotReadyError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

"""Predict upcoming points from a saved model: python -m src.models.predict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common import PROJECT_ROOT, ensure_parent, load_config
from src.features.preparation import FeaturePreprocessor
from src.models.data import KEY, load_feature_dataset
from src.models.network import PointsMLP, predict_array


def load_checkpoint(path: Path) -> tuple[PointsMLP, FeaturePreprocessor, dict]:
    """Restore weights and their exact preprocessing schema together."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported model checkpoint format.")
    processor = FeaturePreprocessor.from_dict(checkpoint["preprocessor"])
    if checkpoint["input_columns"] != processor.output_columns or checkpoint["input_size"] != len(processor.output_columns):
        raise ValueError("Checkpoint architecture and preprocessing schemas differ.")
    options = checkpoint["model_options"]
    model = PointsMLP(checkpoint["input_size"], options["hidden_layers"], options["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, processor, checkpoint


def run_prediction(settings: dict | None = None, checkpoint_path: Path | None = None) -> dict:
    settings = load_config() if settings is None else settings
    dataset = load_feature_dataset(settings)
    paths = settings["paths"]
    if checkpoint_path is None:
        latest_path = PROJECT_ROOT / paths["latest_model_file"]
        if not latest_path.exists():
            raise ValueError("No model checkpoint found. Run src.models.train first.")
        checkpoint_path = PROJECT_ROOT / json.loads(latest_path.read_text())["checkpoint"]
    else:
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    model, processor, checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["season_id"] != dataset.manifest["season_id"] or checkpoint["target"] != dataset.target:
        raise ValueError("Model season or target differs from the feature dataset.")
    next_week = dataset.manifest["prediction_gameweek"]
    if next_week is None:
        raise ValueError("No upcoming gameweek is available in the saved features.")
    if max(checkpoint["trained_gameweeks"]) >= next_week:
        raise ValueError("Model training must precede the prediction gameweek.")
    inputs = processor.transform(dataset.prediction).to_numpy(dtype=np.float32)
    predicted = predict_array(model, inputs)
    records = [
        {"element_id": int(player), "gameweek": int(week), "predicted_points": float(points)}
        for (player, week), points in zip(dataset.prediction[KEY].itertuples(index=False, name=None), predicted)
    ]
    result = {
        "season_id": checkpoint["season_id"], "gameweek": next_week, "model_checkpoint": str(checkpoint_path),
        "training_status": checkpoint["training_status"], "trained_gameweeks": checkpoint["trained_gameweeks"],
        "validation_metrics": checkpoint["validation_metrics"], "baseline_comparison": checkpoint["baseline_comparison"],
        "predictions": records,
    }
    destination = ensure_parent(PROJECT_ROOT / paths["predictions_file"])
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"GW{next_week}: {len(records):,} NN points predictions; status={result['training_status']}")
    print(f"Saved: {destination}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="use this checkpoint instead of models/latest.json")
    args = parser.parse_args()
    run_prediction(checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()

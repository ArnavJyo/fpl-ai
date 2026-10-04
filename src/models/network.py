"""Reusable points-regression network; no model is constructed on import."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


class PointsMLP(nn.Module):
    """Configured hidden layers and one unrestricted points output per row."""

    def __init__(self, input_size: int, hidden_layers: Sequence[int], dropout: float = 0.0):
        super().__init__()
        widths = [input_size, *hidden_layers]
        if any(isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in widths):
            raise ValueError("Input and hidden layer sizes must be positive integers.")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be between zero (inclusive) and one (exclusive).")
        self.input_size = input_size
        layers = []
        previous_size = input_size
        for hidden_size in hidden_layers:
            layers.extend([nn.Linear(previous_size, hidden_size), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous_size = hidden_size
        # No ReLU here: points can be negative, and this is regression.
        layers.append(nn.Linear(previous_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.input_size:
            raise ValueError(f"Expected a batch with shape (rows, {self.input_size}).")
        # (rows, 1) -> (rows,), matching our one-dimensional target arrays.
        return self.layers(inputs).squeeze(-1)


def predict_array(model: PointsMLP, inputs: np.ndarray) -> np.ndarray:
    """Disable dropout and gradients for repeatable CPU inference."""
    values = torch.as_tensor(inputs, dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("Prediction inputs must be finite in float32.")
    model.eval()
    with torch.no_grad():
        predictions = model(values).cpu().numpy()
    if not np.isfinite(predictions).all():
        raise ValueError("The model produced nonfinite predictions.")
    return predictions


def main() -> None:
    """Inspect the architecture using the real saved feature schema."""
    from src.common import load_config
    from src.features.preparation import FeaturePreprocessor
    from src.models.data import load_feature_dataset

    settings = load_config()
    dataset = load_feature_dataset(settings)
    processor = FeaturePreprocessor.fit(dataset.history, dataset.feature_columns)
    options = settings["model"]
    model = PointsMLP(len(processor.output_columns), options["hidden_layers"], options["dropout"])
    print(model)
    print(f"Input columns: {model.input_size}; parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Architecture inspection only. Train with: python -m src.models.train")


if __name__ == "__main__":
    main()

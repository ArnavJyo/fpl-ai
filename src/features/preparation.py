"""Time-ordered validation and reusable, training-fitted numeric transforms."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def walk_forward_splits(
    frame: pd.DataFrame, min_train_gameweeks: int, validation_window_gws: int,
) -> list[dict[str, list[int]]]:
    """Expanding training windows followed by complete, nonoverlapping holdouts."""
    if min_train_gameweeks < 1 or validation_window_gws < 1:
        raise ValueError("Training and validation windows must be positive.")
    weeks = sorted(int(week) for week in frame.gameweek.unique())
    if weeks and weeks != list(range(weeks[0], weeks[-1] + 1)):
        raise ValueError("Walk-forward validation needs consecutive recorded gameweeks.")
    return [
        {"train_gameweeks": weeks[:end], "validation_gameweeks": weeks[end:end + validation_window_gws]}
        for end in range(min_train_gameweeks, len(weeks) - validation_window_gws + 1, validation_window_gws)
    ]


@dataclass
class FeaturePreprocessor:
    """Fit medians/scales on training only and preserve missing-value indicators.

    Columns with no observed training values are excluded for that fit, even
    if validation or prediction has values. The stored schema fixes ordering.
    """

    columns: list[str]
    excluded_columns: list[str]
    medians: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]

    @classmethod
    def fit(cls, train: pd.DataFrame, columns: list[str]) -> FeaturePreprocessor:
        if train.empty or len(set(columns)) != len(columns):
            raise ValueError("Fit requires nonempty training data and unique feature names.")
        numeric = train[columns].astype(float)
        if np.isinf(numeric.to_numpy()).any():
            raise ValueError("Features must not contain infinity.")
        active = [column for column in columns if numeric[column].notna().any()]
        if not active:
            raise ValueError("No observed training features are available.")
        excluded = [column for column in columns if column not in active]
        medians = numeric[active].median()
        filled = numeric[active].fillna(medians)
        scales = filled.std(ddof=0).replace(0, 1)
        return cls(active, excluded, medians.to_dict(), filled.mean().to_dict(), scales.to_dict())

    @property
    def output_columns(self) -> list[str]:
        return [*self.columns, *[f"{column}__missing" for column in self.columns]]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame[self.columns].astype(float)
        if np.isinf(numeric.to_numpy()).any():
            raise ValueError("Features must not contain infinity.")
        missing = numeric.isna().astype(float).add_suffix("__missing")
        scaled = (numeric.fillna(self.medians) - pd.Series(self.means)) / pd.Series(self.scales)
        return pd.concat([scaled, missing], axis=1)[self.output_columns]

    def to_dict(self) -> dict:
        return {**asdict(self), "output_columns": self.output_columns}

    @classmethod
    def from_dict(cls, state: dict) -> FeaturePreprocessor:
        return cls(**{key: state[key] for key in ("columns", "excluded_columns", "medians", "means", "scales")})

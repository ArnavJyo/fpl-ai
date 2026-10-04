"""Load raw model inputs and prepare training-only chronological folds."""

from __future__ import annotations

import json
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.common import PROJECT_ROOT, load_config
from src.features.preparation import FeaturePreprocessor, walk_forward_splits

KEY = ["element_id", "gameweek"]


@dataclass
class FeatureDataset:
    history: pd.DataFrame
    prediction: pd.DataFrame
    manifest: dict

    @property
    def target(self) -> str:
        return self.manifest["target"]

    @property
    def feature_columns(self) -> list[str]:
        return self.manifest["feature_columns"]


@dataclass
class PreparedFold:
    train_gameweeks: list[int]
    validation_gameweeks: list[int]
    train: pd.DataFrame
    validation: pd.DataFrame
    processor: FeaturePreprocessor
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray


@dataclass
class PreparedBacktest:
    """Three chronological periods sharing one training-fitted preprocessor."""

    train_gameweeks: list[int]
    validation_gameweeks: list[int]
    test_gameweeks: list[int]
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    processor: FeaturePreprocessor
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def validate_dataset(dataset: FeatureDataset) -> None:
    """Reject mislabeled rows, nonfinite values, and inconsistent artifacts."""
    target, columns = dataset.target, dataset.feature_columns
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("The manifest needs a nonempty, unique feature list.")
    if set(columns).intersection([*KEY, target]):
        raise ValueError("Player IDs, gameweeks, and the target cannot be model inputs.")
    if dataset.history.empty:
        raise ValueError("Model preparation needs historical rows.")
    for name, frame in [("history", dataset.history), ("prediction", dataset.prediction)]:
        required = [*KEY, *columns, *([target] if name == "history" else [])]
        missing = sorted(set(required).difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")
        if not frame.columns.is_unique:
            raise ValueError(f"{name} has duplicate column names.")
        if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
            raise ValueError(f"{name} needs unique, nonmissing player/gameweek keys.")
        keys = frame[KEY].to_numpy(dtype=float)
        if not np.isfinite(keys).all() or not (keys == np.floor(keys)).all() or (keys < 1).any():
            raise ValueError(f"{name} needs positive integer player/gameweek keys.")
        if (frame.gameweek > 38).any():
            raise ValueError(f"{name} gameweeks must be between 1 and 38.")
        if np.isinf(frame[columns].to_numpy(dtype=float)).any():
            raise ValueError(f"{name} features must not contain infinity.")
    if not np.isfinite(dataset.history[target].to_numpy(dtype=float)).all():
        raise ValueError("Historical targets must be finite and nonmissing.")
    if target in dataset.prediction.columns:
        raise ValueError("Prediction inputs must not contain the target.")
    weeks = sorted(int(week) for week in dataset.history.gameweek.unique())
    if weeks != dataset.manifest["finished_gameweeks"]:
        raise ValueError("Historical gameweeks do not match the feature manifest.")
    if len(dataset.history) != dataset.manifest["history_rows"] or len(dataset.prediction) != dataset.manifest["prediction_rows"]:
        raise ValueError("Feature row counts do not match the manifest.")
    next_week = dataset.manifest["prediction_gameweek"]
    if next_week is None:
        if not dataset.prediction.empty:
            raise ValueError("Prediction rows exist without a prediction gameweek.")
    elif next_week <= max(weeks) or dataset.prediction.empty or not dataset.prediction.gameweek.eq(next_week).all():
        raise ValueError("Prediction inputs must cover the manifest's future gameweek.")


def load_feature_dataset(settings: dict | None = None) -> FeatureDataset:
    """Use raw features, never the matrix fitted on all historical weeks."""
    settings = load_config() if settings is None else settings
    paths = settings["paths"]
    manifest = json.loads((PROJECT_ROOT / paths["feature_manifest_file"]).read_text())
    if manifest["target"] != settings["features"]["target"]:
        raise ValueError("Configured target differs from the saved feature target.")
    dataset = FeatureDataset(
        pd.read_parquet(PROJECT_ROOT / paths["features_file"]),
        pd.read_parquet(PROJECT_ROOT / paths["prediction_features_file"]),
        manifest,
    )
    validate_dataset(dataset)
    dataset.history = dataset.history.sort_values(["gameweek", "element_id"]).reset_index(drop=True)
    dataset.prediction = dataset.prediction.sort_values(["gameweek", "element_id"]).reset_index(drop=True)
    return dataset


def _float32(values: pd.DataFrame | pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Prepared inputs and targets must be finite in float32.")
    return array


def prepare_folds(dataset: FeatureDataset, min_train_gameweeks: int, validation_window_gws: int) -> list[PreparedFold]:
    """Keep whole weeks together and fit a separate schema on each training set."""
    validate_dataset(dataset)
    splits = walk_forward_splits(dataset.history, min_train_gameweeks, validation_window_gws)
    folds = []
    for split in splits:
        train = dataset.history.loc[dataset.history.gameweek.isin(split["train_gameweeks"])].reset_index(drop=True)
        validation = dataset.history.loc[dataset.history.gameweek.isin(split["validation_gameweeks"])].reset_index(drop=True)
        processor = FeaturePreprocessor.fit(train, dataset.feature_columns)
        folds.append(PreparedFold(
            **split, train=train, validation=validation, processor=processor,
            x_train=_float32(processor.transform(train)), y_train=_float32(train[dataset.target]),
            x_validation=_float32(processor.transform(validation)), y_validation=_float32(validation[dataset.target]),
        ))
    return folds


def prepare_backtest(
    dataset: FeatureDataset, train_gameweeks: list[int],
    validation_gameweeks: list[int], test_gameweeks: list[int],
) -> PreparedBacktest:
    """Fit on train only; validation selects epochs and test only assesses them.

    Each period must contain complete, consecutive recorded gameweeks. Periods
    cannot overlap or run backwards. This explicit split does not modify the
    configured walk-forward windows used by normal model training.
    """
    validate_dataset(dataset)
    available = set(dataset.manifest["finished_gameweeks"])
    periods = []
    for name, weeks in [("train", train_gameweeks), ("validation", validation_gameweeks), ("test", test_gameweeks)]:
        if not weeks or any(isinstance(week, bool) or not isinstance(week, int) for week in weeks):
            raise ValueError(f"{name} gameweeks must be a nonempty list of integers.")
        ordered = sorted(weeks)
        if len(set(ordered)) != len(ordered) or ordered != list(range(ordered[0], ordered[-1] + 1)):
            raise ValueError(f"{name} gameweeks must be unique and consecutive.")
        if not set(ordered).issubset(available):
            raise ValueError(f"{name} gameweeks are not all present in the saved history.")
        periods.append(ordered)
    train_weeks, validation_weeks, test_weeks = periods
    if max(train_weeks) >= min(validation_weeks) or max(validation_weeks) >= min(test_weeks):
        raise ValueError("Backtest periods must follow train < validation < test without overlap.")
    frames = [dataset.history.loc[dataset.history.gameweek.isin(weeks)].sort_values(KEY).reset_index(drop=True)
              for weeks in periods]
    train, validation, test = frames
    processor = FeaturePreprocessor.fit(train, dataset.feature_columns)
    return PreparedBacktest(
        train_gameweeks=train_weeks, validation_gameweeks=validation_weeks, test_gameweeks=test_weeks,
        train=train, validation=validation, test=test, processor=processor,
        x_train=_float32(processor.transform(train)), y_train=_float32(train[dataset.target]),
        x_validation=_float32(processor.transform(validation)), y_validation=_float32(validation[dataset.target]),
        x_test=_float32(processor.transform(test)), y_test=_float32(test[dataset.target]),
    )

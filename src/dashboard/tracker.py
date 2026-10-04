"""Archive forecasts before deadlines; score only finished weeks afterward."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common import PROJECT_ROOT
from src.models.baseline import regression_metrics
from src.models.train import save_json


def archive_directory(settings: dict, season_id: str) -> Path:
    """Separate seasons without depending on filesystem-safe deadline strings."""
    season_key = hashlib.sha256(season_id.encode()).hexdigest()[:12]
    return PROJECT_ROOT / settings["paths"]["outputs_dir"] / "forecast_archive" / season_key


def archive_forecast(settings: dict, predictions: dict, now: datetime | None = None) -> Path:
    """Save the latest pre-deadline forecast; refuse retrospectively generated ones."""
    now = datetime.now(timezone.utc) if now is None else now
    bootstrap = json.loads((PROJECT_ROOT / settings["paths"]["raw_data_dir"] / "bootstrap_static.json").read_text())
    if bootstrap["events"][0]["deadline_time"] != predictions["season_id"]:
        raise ValueError("Forecast archive season differs from cached bootstrap.")
    event = next(event for event in bootstrap["events"] if event["id"] == predictions["gameweek"])
    deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
    if now.tzinfo is None or now >= deadline:
        raise ValueError("Forecast deadline has passed. Refresh data; a retrospective forecast cannot enter the tracker.")
    path = archive_directory(settings, predictions["season_id"]) / f"gw{predictions['gameweek']:02d}.json"
    save_json(path, {**predictions, "generated_at": now.isoformat(), "deadline": deadline.isoformat()})
    return path


def performance_tracker(settings: dict, season_id: str, finished_gameweeks: list[int]) -> list[dict]:
    """Join genuinely archived predictions to labels; report missing rows honestly."""
    history = pd.read_parquet(PROJECT_ROOT / settings["paths"]["features_file"])
    target = settings["features"]["target"]
    result = []
    for path in sorted(archive_directory(settings, season_id).glob("gw*.json")):
        saved = json.loads(path.read_text())
        if saved["season_id"] != season_id or saved["gameweek"] not in finished_gameweeks:
            continue
        generated = datetime.fromisoformat(saved["generated_at"])
        deadline = datetime.fromisoformat(saved["deadline"])
        if generated.tzinfo is None or deadline.tzinfo is None or generated >= deadline:
            raise ValueError(f"Archive {path.name} is not a pre-deadline forecast.")
        predicted = pd.DataFrame(saved["predictions"])
        actual = history.loc[history.gameweek == saved["gameweek"], ["element_id", "gameweek", target]]
        joined = predicted.merge(actual, on=["element_id", "gameweek"], how="inner", validate="one_to_one")
        result.append({"gameweek": saved["gameweek"], "prediction_rows": len(predicted), "scored_rows": len(joined),
                       "unmatched_rows": len(predicted) - len(joined), "training_status": saved["training_status"],
                       "generated_at": saved["generated_at"], "metrics": regression_metrics(joined[target].to_numpy(), joined.predicted_points.to_numpy()) if len(joined) else None})
    return result

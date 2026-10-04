"""Build all feature artifacts offline: python -m src.features.pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common import ensure_parent, load_config, path_for, read_table, write_table
from src.data.season_state import season_state
from src.data.parse_bootstrap import parse_players
from src.data.parse_fixtures import parse_fixtures
from src.features.build_features import KEY, SNAPSHOT_COLUMNS, assemble_features, player_snapshot
from src.features.context import build_feature_history, build_team_results
from src.features.preparation import FeaturePreprocessor, walk_forward_splits


def read_fixture_stats(raw_dir: Path) -> pd.DataFrame:
    """Read per-fixture xG from cached summaries, without fetching."""
    records = []
    for path in sorted(raw_dir.glob("element_summary_*.json")):
        element_id = int(path.stem.removeprefix("element_summary_"))
        for game in json.loads(path.read_text()).get("history", []):
            records.append({
                "element_id": element_id, "fixture_id": game["fixture"],
                "expected_goals": float(game["expected_goals"]),
            })
    return pd.DataFrame(records, columns=["element_id", "fixture_id", "expected_goals"])


def load_snapshots(bootstrap: dict) -> pd.DataFrame:
    """Reject snapshots collected after a target deadline or from other seasons."""
    path = path_for("player_snapshots_file")
    columns = [*KEY, *SNAPSHOT_COLUMNS, "observed_at", "season_id"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    snapshots = pd.read_parquet(path)
    season_id = bootstrap["events"][0]["deadline_time"]
    snapshots = snapshots.loc[snapshots.season_id == season_id].copy()
    deadlines = {event["id"]: pd.Timestamp(event["deadline_time"]) for event in bootstrap["events"]}
    for row in snapshots[["gameweek", "observed_at"]].drop_duplicates().itertuples(index=False):
        observed = pd.Timestamp(row.observed_at)
        if observed.tzinfo is None or row.gameweek not in deadlines or observed >= deadlines[row.gameweek]:
            raise ValueError("Historical player snapshot was not captured before its target deadline.")
    if snapshots.duplicated(KEY).any():
        raise ValueError("Archived snapshots have duplicate player/gameweek keys.")
    return snapshots


def save_json(path: Path, value: dict) -> None:
    ensure_parent(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def matrix(frame: pd.DataFrame, processor: FeaturePreprocessor, target: str | None) -> pd.DataFrame:
    metadata = frame[[*KEY, *([target] if target else [])]].reset_index(drop=True)
    return pd.concat([metadata, processor.transform(frame).reset_index(drop=True)], axis=1)


def run_features() -> dict:
    """Rebuild raw features, forecast inputs, validation folds, and preprocessing."""
    settings = load_config()
    options = settings["features"]
    if options.get("categorical", []) != ["position"]:
        raise ValueError("This feature schema supports categorical: [position].")
    target = options["target"]
    raw_dir = path_for("raw_data_dir")
    bootstrap_path = raw_dir / "bootstrap_static.json"
    bootstrap = json.loads(bootstrap_path.read_text())
    state = season_state(bootstrap["events"])
    source_time = pd.Timestamp(bootstrap_path.stat().st_mtime, unit="s", tz="UTC")
    players = read_table("players_file")
    fixtures = read_table("fixtures_file")
    if not players.equals(parse_players(bootstrap)):
        raise ValueError("Player table does not match the cached bootstrap. Rebuild ingestion first.")
    fixture_path = raw_dir / "fixtures.json"
    if not fixtures.equals(parse_fixtures(json.loads(fixture_path.read_text()))):
        raise ValueError("Fixture table does not match its cached source. Rebuild ingestion first.")
    stats = read_table("gw_stats_file")
    spine = read_table("spine_file")
    finished = list(state.finished_gameweeks)
    stats = stats.loc[stats.gameweek.isin(finished)].copy()
    if stats.empty:
        raise ValueError("Feature engineering needs at least one finished gameweek of performance data.")
    if sorted(stats.gameweek.unique().tolist()) != finished:
        raise ValueError("Performance data does not cover all finished gameweeks. Rebuild ingestion first.")
    spine = spine.loc[spine.gameweek.isin(finished)].copy()
    historical = build_feature_history(stats, spine, fixtures)
    fixture_stats = read_fixture_stats(raw_dir)
    results = build_team_results(fixtures.loc[fixtures.gameweek.isin(finished)], spine, fixture_stats)
    snapshots = load_snapshots(bootstrap)
    next_week = state.next_gameweek
    if next_week is not None:
        deadline = pd.Timestamp(next(event["deadline_time"] for event in bootstrap["events"] if event["id"] == next_week))
        if source_time >= deadline or pd.Timestamp.now(tz="UTC") >= deadline:
            raise ValueError("Upcoming deadline has passed for the saved season state. Refresh ingestion first.")
        current = player_snapshot(players, next_week)
        current["observed_at"] = source_time.isoformat()
        current["season_id"] = bootstrap["events"][0]["deadline_time"]
        previous = snapshots.loc[snapshots.gameweek != next_week]
        snapshots = pd.concat([previous, current], ignore_index=True) if not previous.empty else current
    features, prediction, columns = assemble_features(
        historical, players, fixtures, results, snapshots, options["rolling_windows"],
        options["rolling_stats"], target, next_week,
    )
    processor = FeaturePreprocessor.fit(features, columns)
    folds = walk_forward_splits(
        features, settings["model"]["min_gameweeks_for_training"], settings["model"]["validation_window_gws"],
    )
    fold_records = []
    for number, split in enumerate(folds, start=1):
        train = features.loc[features.gameweek.isin(split["train_gameweeks"])].reset_index(drop=True)
        validation = features.loc[features.gameweek.isin(split["validation_gameweeks"])].reset_index(drop=True)
        fitted = FeaturePreprocessor.fit(train, columns)
        directory = path_for("feature_folds_dir") / f"fold_{number}"
        ensure_parent(directory / "train.parquet")
        matrix(train, fitted, target).to_parquet(directory / "train.parquet", index=False)
        matrix(validation, fitted, target).to_parquet(directory / "validation.parquet", index=False)
        save_json(directory / "preprocessor.json", fitted.to_dict())
        fold_records.append({**split, "directory": str(directory.relative_to(path_for("feature_folds_dir"))),
                             "train_rows": len(train), "validation_rows": len(validation)})
    write_table(features, "features_file")
    write_table(prediction, "prediction_features_file")
    write_table(matrix(features, processor, target), "training_matrix_file")
    write_table(matrix(prediction, processor, None), "prediction_matrix_file")
    write_table(snapshots, "player_snapshots_file")
    prices = historical.loc[historical.price_tenths.notna(), [*KEY, "price_tenths"]].copy()
    prices["source"] = "historical_fixture"
    if next_week is not None:
        current_prices = players[["element_id", "price_tenths"]].copy()
        current_prices["gameweek"] = next_week
        current_prices["source"] = "pre_deadline_snapshot"
        prices = pd.concat([prices, current_prices], ignore_index=True)
    prices["price_tenths"] = prices.price_tenths.astype("int64")
    prices["gameweek"] = prices.gameweek.astype("int64")
    write_table(prices.sort_values(KEY).reset_index(drop=True), "prices_file")
    save_json(path_for("feature_preprocessor_file"), processor.to_dict())
    report = {
        "season_id": bootstrap["events"][0]["deadline_time"],
        "target": target, "prediction_gameweek": next_week,
        "finished_gameweeks": finished, "history_rows": len(features), "prediction_rows": len(prediction),
        "feature_columns": columns, "active_feature_columns": processor.columns,
        "excluded_unobserved_columns": processor.excluded_columns,
        "baseline_feature_column": f"{target}_mean_last_{max(options['rolling_windows'])}",
        "matrix_columns": processor.output_columns, "folds": fold_records,
        "training_ready": len(finished) >= settings["model"]["min_gameweeks_for_training"],
        "validation_ready": bool(folds),
        "bootstrap_cache_observed_at": source_time.isoformat(),
        "bootstrap_cache_age_hours": max(0, (pd.Timestamp.now(tz="UTC") - source_time).total_seconds() / 3600),
        "full_history_preprocessor_fit_gameweeks": finished,
        "historical_schedule_assumption": "Saved fixture assignment is assumed to match its pre-deadline schedule; past schedules are not archived.",
        "snapshot_timestamp_source": "Cached bootstrap file modification time; this is a local provenance proxy.",
    }
    save_json(path_for("feature_manifest_file"), report)
    print(f"Prediction GW: {next_week}; {len(processor.columns)} active inputs; {len(folds)} walk-forward folds")
    if not report["validation_ready"]:
        minimum = settings["model"]["min_gameweeks_for_training"] + settings["model"]["validation_window_gws"]
        print(f"Validation needs {minimum} finished GWs under current settings; {len(finished)} are available.")
    if processor.excluded_columns:
        print(f"Excluded {len(processor.excluded_columns)} inputs without historical observations; retained in raw forecast context.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_features()


if __name__ == "__main__":
    main()

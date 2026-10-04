"""Data → features → model → forecasts → ILP → dashboard.

Run `python run_pipeline.py --end-to-end --refresh --allow-experimental` to
refresh and fit; use `--offline --skip-training` for saved inputs/checkpoints.
The existing data-only CLI remains available without --end-to-end.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from src.common import PROJECT_ROOT, load_config
from src.dashboard.build_dashboard import run_dashboard
from src.dashboard.tracker import archive_forecast
from src.data.stages import fetch_bootstrap, make_client, run_bootstrap, run_fixtures, run_gameweek_stats, run_spine
from src.features.pipeline import run_features
from src.models.baseline import run_baseline
from src.models.predict import run_prediction
from src.models.train import run_training, save_json
from src.optimizer.manager import integer
from src.optimizer.solve import run_optimization


def runtime_settings() -> dict:
    """Allow numeric team ID and exact JSON state from CI environment variables."""
    settings = copy.deepcopy(load_config())
    value = os.environ.get("FPL_TEAM_ID")
    if value:
        if not value.isdecimal() or int(value) < 1:
            raise ValueError("FPL_TEAM_ID must be a positive numeric entry ID.")
        settings["manager"]["team_id"] = int(value)
    integer(settings["manager"]["team_id"], "manager.team_id", 1) if settings["manager"]["team_id"] is not None else None
    return settings


def run_end_to_end(settings: dict | None = None, refresh: bool = False, offline: bool = False,
                   skip_training: bool = False, allow_experimental: bool = False,
                   workers: int = 6, manager_path: Path | None = None) -> dict:
    """Stop on stage failure and record it; never report downstream success.

    The model uses the raw feature pipeline, so the old inner-joined history
    diagnostic is not a dependency. This preserves blank/double-week rows.
    """
    settings = runtime_settings() if settings is None else settings
    integer(workers, "workers", 1)
    if offline and refresh:
        raise ValueError("--offline and --refresh cannot be used together.")
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "stages": [],
              "offline": offline, "refresh": refresh, "team_id": settings["manager"]["team_id"],
              "allow_experimental": allow_experimental, "skip_training": skip_training}
    report_path = PROJECT_ROOT / settings["paths"]["pipeline_report_file"]

    def stage(name: str, action):
        started = time.monotonic()
        record = {"name": name, "status": "running"}
        report["stages"].append(record)
        print(f"\n{name}", flush=True)
        try:
            value = action()
            record.update(status="complete", duration_seconds=round(time.monotonic() - started, 3))
            return value
        except Exception as exc:
            record.update(status="failed", error=str(exc), duration_seconds=round(time.monotonic() - started, 3))
            raise

    try:
        if not offline:
            def ingest():
                client = make_client(refresh=refresh)
                try:
                    bootstrap = fetch_bootstrap(client)
                    run_bootstrap(bootstrap)
                    run_fixtures(client)
                    run_gameweek_stats(client, bootstrap)
                    run_spine(client, bootstrap, workers=workers)
                finally:
                    client.session.close()
            stage("Refresh source tables", ingest)
            stage("Build past-only features", run_features)
        stage("Score baseline on available full folds", lambda: run_baseline(settings))
        if not skip_training:
            stage("Train points model", lambda: run_training(settings, experimental=allow_experimental))
        elif not allow_experimental:
            pointer = json.loads((PROJECT_ROOT / settings["paths"]["latest_model_file"]).read_text())
            if pointer.get("training_status") == "experimental":
                raise ValueError("Saved model is experimental. Pass --allow-experimental explicitly to use its projections.")
        predictions = stage("Generate upcoming forecasts", lambda: run_prediction(settings))
        if not allow_experimental and predictions["training_status"] == "experimental":
            raise ValueError("Saved model is experimental. Pass --allow-experimental explicitly to use its projections.")
        stage("Archive forecast before deadline", lambda: archive_forecast(settings, predictions))
        stage("Optimize squad and lineup", lambda: run_optimization(settings, manager_path,
                                                                    fetch_manager=not offline, refresh=refresh))
        dashboard = stage("Build dashboard and performance tracker", lambda: run_dashboard(settings))
        report.update(status="complete", gameweek=predictions["gameweek"], training_status=predictions["training_status"], dashboard=str(dashboard))
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(report_path, report)
    print(f"\nEnd-to-end run complete: {report['dashboard']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-to-end", action="store_true", help="accepted by the root entry point")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true", help="use saved features; make no API requests")
    parser.add_argument("--skip-training", action="store_true", help="load the current saved model")
    parser.add_argument("--allow-experimental", action="store_true", help="explicitly allow a model before full folds exist")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--manager-state", type=Path)
    args = parser.parse_args()
    try:
        state_json = os.environ.get("FPL_MANAGER_STATE_JSON")
        if state_json:
            # Never echo the secret; this file is ignored and excluded from CI caches.
            state = json.loads(state_json)
            save_json(PROJECT_ROOT / load_config()["paths"]["manager_state_file"], state)
        run_end_to_end(refresh=args.refresh, offline=args.offline, skip_training=args.skip_training,
                       allow_experimental=args.allow_experimental, workers=args.workers, manager_path=args.manager_state)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Pipeline stopped: {exc}\n")


if __name__ == "__main__":
    main()

"""Build the offline dashboard: python -m src.dashboard.build_dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common import PROJECT_ROOT, ensure_parent, load_config
from src.dashboard.tracker import performance_tracker
from src.models.train import save_json
from src.optimizer.solve import candidate_table


def safe_script_json(value: dict) -> str:
    """Keep untrusted names from ending an embedded JSON script element."""
    return json.dumps(value, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def run_dashboard(settings: dict | None = None) -> Path:
    """Render only after squad/prediction season and gameweek match."""
    settings = load_config() if settings is None else settings
    paths = settings["paths"]
    predictions = json.loads((PROJECT_ROOT / paths["predictions_file"]).read_text())
    squad = json.loads((PROJECT_ROOT / paths["squad_file"]).read_text())
    manifest = json.loads((PROJECT_ROOT / paths["feature_manifest_file"]).read_text())
    for source in [squad, {**manifest, "gameweek": manifest["prediction_gameweek"]}]:
        if source["season_id"] != predictions["season_id"] or source["gameweek"] != predictions["gameweek"]:
            raise ValueError("Dashboard sources belong to different seasons/gameweeks.")
    candidates = candidate_table(pd.read_parquet(PROJECT_ROOT / paths["players_file"]), predictions,
                                 pd.read_parquet(PROJECT_ROOT / paths["prediction_features_file"]),
                                 pd.read_parquet(PROJECT_ROOT / paths["teams_file"]))
    columns = ["element_id", "web_name", "position", "short_name", "price_tenths", "predicted_points", "availability_pct", "fixture_count"]
    players = candidates[columns].sort_values("predicted_points", ascending=False).to_dict(orient="records")
    data = {"squad": squad, "players": players, "tracker": performance_tracker(settings, predictions["season_id"], manifest["finished_gameweeks"]),
            "source_observed_at": manifest.get("bootstrap_cache_observed_at"), "finished_gameweeks": manifest["finished_gameweeks"]}
    template = (Path(__file__).resolve().parent / "templates/dashboard.html").read_text()
    page = template.replace("{{DASHBOARD_DATA}}", safe_script_json(data))
    directory = PROJECT_ROOT / paths["dashboard_dir"]
    path = ensure_parent(directory / "index.html")
    path.write_text(page)
    save_json(directory / "data.json", data)
    # Downloads are next to the HTML so a Pages deployment needs just this folder.
    save_json(directory / "squad.json", squad)
    save_json(directory / "predictions.json", predictions)
    pd.DataFrame(players).to_csv(directory / "players.csv", index=False)
    print(f"Dashboard: {path}")
    return path


if __name__ == "__main__":
    run_dashboard()

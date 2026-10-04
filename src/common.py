"""Shared config loading and path helpers for the FPL AI pipeline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with SETTINGS_PATH.open() as fh:
        return yaml.safe_load(fh)


def path_for(key: str) -> Path:
    """Resolve a path from the `paths` block of settings.yaml."""
    rel = load_config()["paths"][key]
    return PROJECT_ROOT / rel


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# Canonical position labels. element_type is 1-indexed on the FPL API.
POSITION_NAMES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def position_name(element_type: int) -> str:
    return POSITION_NAMES[element_type]


def read_table(path_key: str) -> pd.DataFrame:
    """Load a table using its name in the configuration's paths block."""
    return pd.read_parquet(path_for(path_key))


def write_table(frame: pd.DataFrame, path_key: str) -> None:
    """Overwrite a derived table, creating its destination directory if needed."""
    destination = ensure_parent(path_for(path_key))
    frame.to_parquet(destination, index=False)
    print(f"{destination.name}: {len(frame):,} rows × {len(frame.columns)} columns")

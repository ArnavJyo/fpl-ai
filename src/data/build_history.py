"""Join saved performance and context on player ID + gameweek.

Rebuild the complete table; there is no append or API-fetch path."""

from __future__ import annotations

# Keep direct-file commands working; package imports use the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hashlib

import pandas as pd

KEY = ["element_id", "gameweek"]


def build_history(stats: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    """Require unique input keys and keep only rows represented on both sides."""
    history = stats.merge(spine, on=KEY, how="inner", validate="one_to_one")
    return history.sort_values(KEY).reset_index(drop=True)


def unmatched_keys(
    stats: pd.DataFrame, spine: pd.DataFrame,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Identify rows the inner join will discard from each input."""
    stats_keys = set(stats[KEY].itertuples(index=False, name=None))
    spine_keys = set(spine[KEY].itertuples(index=False, name=None))
    return stats_keys - spine_keys, spine_keys - stats_keys


def content_hash(frame: pd.DataFrame) -> str:
    """Fingerprint sorted table values and index, not the Parquet file bytes."""
    hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()[:16]


def main() -> None:
    from src.data.pipeline import stage_main

    stage_main("history")


if __name__ == "__main__":
    main()

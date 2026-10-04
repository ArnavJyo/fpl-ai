"""Interpret finished, current, and next gameweeks from bootstrap events."""

from __future__ import annotations

# Keep direct-file commands working; package imports use the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class SeasonState:
    finished_gameweeks: tuple[int, ...]
    next_gameweek: int | None
    current_gameweek: int | None
    next_deadline: str | None
    season_over: bool

    @property
    def last_finished(self) -> int | None:
        return self.finished_gameweeks[-1] if self.finished_gameweeks else None

    @property
    def is_pre_season(self) -> bool:
        return not self.finished_gameweeks and self.next_gameweek is not None

    def describe(self) -> str:
        parts = [
            f"finished: {self.finished_gameweeks or 'none'}",
            f"current: {self.current_gameweek}",
            f"next: {self.next_gameweek}",
        ]
        if self.next_deadline:
            parts.append(f"deadline: {self.next_deadline}")
        if self.season_over:
            parts.append("SEASON OVER")
        if self.is_pre_season:
            parts.append("PRE-SEASON")
        return " | ".join(parts)


def season_state(events: list[dict[str, Any]]) -> SeasonState:
    ordered = sorted(events, key=lambda event: event["id"])
    finished = tuple(event["id"] for event in ordered if event.get("finished"))

    next_events = [event for event in ordered if event.get("is_next")]
    if next_events:
        upcoming = next_events[0]
    else:
        unplayed = [event for event in ordered if not event.get("finished")]
        upcoming = unplayed[0] if unplayed else None

    current = next((event["id"] for event in ordered if event.get("is_current")), None)

    return SeasonState(
        finished_gameweeks=finished,
        next_gameweek=upcoming["id"] if upcoming else None,
        current_gameweek=current,
        next_deadline=upcoming.get("deadline_time") if upcoming else None,
        season_over=upcoming is None,
    )


def main() -> None:
    from src.data.pipeline import stage_main

    stage_main("season-state")


if __name__ == "__main__":
    main()

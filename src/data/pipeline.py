"""Command-line entry point and explicit execution order for the data pipeline.

Run everything: python -m src.data.pipeline
Refresh sources: python -m src.data.pipeline --refresh
Run one stage:  python -m src.data.pipeline --stage history
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from src.common import load_config
from src.data.season_state import season_state
from src.data.stages import (
    fetch_bootstrap,
    make_client,
    run_bootstrap,
    run_fixtures,
    run_gameweek_stats,
    run_history,
    run_spine,
    run_verification,
)

STAGE_NAMES = (
    "all", "bootstrap", "fixtures", "season-state", "gameweek-stats",
    "spine", "history", "verify", "fetch", "features",
)


def run_pipeline(refresh: bool = False, workers: int = 12) -> bool:
    """Rebuild the six derived tables in dependency order, then verify them."""
    client = make_client(refresh=refresh)
    try:
        bootstrap = fetch_bootstrap(client)
        run_bootstrap(bootstrap)
        run_fixtures(client)
        run_gameweek_stats(client, bootstrap)
        run_spine(client, bootstrap, workers=workers)
        run_history()
        print(f"API/cache: {client.stats()}")
        return run_verification()
    finally:
        client.session.close()


def run_one_stage(stage: str, args: argparse.Namespace) -> bool:
    """Run a standalone stage without executing its upstream dependencies."""
    if stage == "features":
        from src.features.pipeline import run_features

        run_features()
        return True
    if stage == "history":
        run_history()
        return True
    if stage == "verify":
        return run_verification()

    client = make_client(refresh=args.refresh)
    try:
        if stage == "fixtures":
            run_fixtures(client)
        else:
            bootstrap = fetch_bootstrap(client)
            if stage == "bootstrap":
                run_bootstrap(bootstrap)
            elif stage == "season-state":
                print(season_state(bootstrap["events"]).describe())
            elif stage == "gameweek-stats":
                run_gameweek_stats(client, bootstrap, gameweeks=args.gameweek)
            elif stage == "spine":
                run_spine(client, bootstrap, workers=args.workers)
            elif stage == "fetch":
                api = load_config()["fpl_api"]
                client.get(api["fixtures"], "fixtures")
                client.get(api["event_live"].format(event_id=1), "event_1_live")
        print(f"API/cache: {client.stats()}")
        return True
    finally:
        client.session.close()


def main(argv: Sequence[str] | None = None, default_stage: str = "all") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGE_NAMES, default=default_stage)
    parser.add_argument("--refresh", action="store_true", help="refetch this run's API inputs")
    parser.add_argument("--workers", type=int, default=12, help="concurrent summary requests")
    parser.add_argument("--gameweek", type=int, nargs="*", help="weeks for --stage gameweek-stats")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.gameweek is not None and args.stage != "gameweek-stats":
        parser.error("--gameweek requires --stage gameweek-stats")

    if args.stage == "all":
        passed = run_pipeline(refresh=args.refresh, workers=args.workers)
    else:
        passed = run_one_stage(args.stage, args)
    raise SystemExit(0 if passed else 1)


def stage_main(stage: str) -> None:
    """Keep the original per-file commands as small wrappers around this CLI."""
    main(default_stage=stage)


if __name__ == "__main__":
    main()

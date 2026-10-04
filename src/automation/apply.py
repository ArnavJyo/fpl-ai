"""Apply one verified plan per entry/season/GW, with durable write receipts.

python -m src.automation.apply --preview   # live reads; saves a private plan
python -m src.automation.apply --apply     # user-authorized account changes

A timeout is ambiguous: record it BEFORE sending a POST and read back state.
Do not blindly retry transfers or consume a second chip. A confirmed partial
transfer commit can resume at lineup saving; an ambiguous mismatch stops.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from src.common import PROJECT_ROOT
from src.automation.client import AuthenticatedFplClient, FplActionError, private_json
from src.automation.plan import TRANSFER_CHIPS, build_plan, team_fingerprint, timestamp, validate_forecast


def journal_path(settings: dict, season_id: str, gameweek: int) -> Path:
    key = hashlib.sha256(f"{settings['manager']['team_id']}:{season_id}:{gameweek}".encode()).hexdigest()[:20]
    return PROJECT_ROOT / settings["automation"]["actions_dir"] / f"receipt-{key}.json"


def lineup_matches(payload: dict, plan: dict) -> bool:
    try:
        fields = ["element", "position", "is_captain", "is_vice_captain"]
        actual = [{k: p[k] for k in fields} for p in payload["picks"]]
        wanted = plan["lineup_request"]["picks"]
        return sorted(actual, key=lambda p: p["position"]) == wanted and chip_matches(payload, plan)
    except (KeyError, TypeError):
        return False


def chip_matches(payload: dict, plan: dict) -> bool:
    active = {c["name"] for c in payload.get("chips", []) if c.get("status_for_entry") == "active"}
    wanted = {plan["chip"]} if plan["chip"] else set()
    return active == wanted


def transfers_match(payload: dict, plan: dict) -> bool:
    """Confirm IDs, exact bank, expected transfer count, and zero new hits."""
    expected = {p["element"] for p in plan["lineup_request"]["picks"]}
    if {p["element"] for p in payload["picks"]} != expected or payload["transfers"]["bank"] != plan["bank_after_tenths"]:
        return False
    if plan["chip"] in TRANSFER_CHIPS:
        # Live FPL preserves the ordinary free-transfer counter during Free Hit
        # and Wildcard; 13 chip transfers can correctly leave `made` at zero.
        # Confirm exact squad/bank and the active unlimited-transfer chip.
        return chip_matches(payload, plan)
    if payload["transfers"]["made"] != plan["source_transfer_made"] + len(plan["transfer_request"]["transfers"]):
        return False
    if len(plan["transfer_request"]["transfers"]) > plan["free_transfers_before"]:
        return False
    # Scoring chips are applied at the later lineup step.
    return not any(c.get("status_for_entry") == "active" and c.get("name") in TRANSFER_CHIPS for c in payload.get("chips", []))


def ensure_open(client, plan: dict, policy: dict) -> dict:
    bootstrap = client.request("GET", "/bootstrap-static/")
    validate_forecast({"season_id": plan["season_id"], "gameweek": plan["gameweek"],
                       "generated_at": plan["forecast_generated_at"], "training_status": plan["training_status"],
                       "source_snapshot_observed_at": plan["source_snapshot_observed_at"]}, bootstrap, policy)
    return bootstrap


def execute_plan(client, plan: dict, settings: dict) -> dict:
    """A journal plus server read-backs protects reruns and partial failures."""
    path = journal_path(settings, plan["season_id"], plan["gameweek"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a") as lock, ExitStack() as cleanup:
        fcntl.flock(lock, fcntl.LOCK_EX)
        client.verify_identity()
        repository = settings["automation"].get("receipt_repository")
        store = None
        if repository:
            from src.automation.receipts import ReceiptStore
            store = ReceiptStore(repository, path.name)
            cleanup.callback(store.close)
            remote = store.load()
            if remote:
                private_json(path, remote)
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt["status"] == "applied":
                print(f"GW{plan['gameweek']} already applied; no account changes repeated.")
                return receipt
            # Resume the exact durable plan, never make a new transfer decision
            # after the earlier operation may already have changed the account.
            plan = receipt["plan"]
        else:
            receipt = {"status": "prepared", "plan": plan, "events": []}

        def record(status: str, message: str):
            receipt["status"] = status
            receipt["events"].append({"at": datetime.now(timezone.utc).isoformat(), "message": message})
            if store:
                store.save(receipt)
            private_json(path, receipt)

        policy = settings["automation"]
        if plan["team_id"] != settings["manager"]["team_id"] or plan["additional_hit_points"] != 0 or policy["max_additional_hit_points"] != 0:
            raise FplActionError("Plan does not match the configured team or zero-hit authorization.")
        if plan["chip"] and plan["chip"] not in policy["allowed_chips"] and plan["chip"] != plan["source_active_chip"]:
            raise FplActionError("The pending plan's chip is no longer allowed by configuration.")
        bootstrap = ensure_open(client, plan, policy)
        current = client.my_team()
        if receipt["status"] == "lineup_pending":
            if lineup_matches(current, plan) and transfers_match(current, plan):
                record("applied", "Recovered an earlier lineup save by reading the live team.")
                return receipt
            raise FplActionError("An earlier lineup save is ambiguous. Inspect the live team; no write was repeated.")
        if receipt["status"] == "transfers_pending":
            if not transfers_match(current, plan):
                raise FplActionError("An earlier transfer response was ambiguous. Live state differs; no transfer was repeated.")
            record("transfers_verified", "Recovered an earlier transfer commit by reading the live team.")
        elif receipt["status"] == "prepared":
            if team_fingerprint(current) != plan["source_fingerprint"]:
                raise FplActionError("Live team changed since planning; regenerate the plan before any writes.")
            prices = {e["id"]: e for e in bootstrap["elements"]}
            for transfer in plan["transfer_request"]["transfers"]:
                player = prices.get(transfer["element_in"])
                if not player or player["now_cost"] != transfer["purchase_price"] or player["status"] in {"s", "u"} or player.get("chance_of_playing_next_round") == 0:
                    raise FplActionError("A purchase price or availability changed; regenerate the plan before writes.")
            needs_transfers = bool(plan["transfer_request"]["transfers"]) or (plan["chip"] in TRANSFER_CHIPS and plan["chip"] != plan["source_active_chip"])
            if needs_transfers:
                record("transfers_pending", "Sending a single transfer/chip commit; automatic POST retries disabled.")
                try:
                    client.request("POST", "/transfers/", plan["transfer_request"])
                except FplActionError:
                    if not transfers_match(client.my_team(), plan):
                        raise
                current = client.my_team()
                if not transfers_match(current, plan):
                    raise FplActionError("Transfer response did not match live squad, bank, allowance or chip. Lineup save stopped.")
            record("transfers_verified", "Squad, bank, transfer count and transfer chip read back successfully.")
        elif receipt["status"] != "transfers_verified":
            raise FplActionError("Unknown action receipt status; stopped without another write.")

        current = client.my_team()
        if not transfers_match(current, plan):
            raise FplActionError("Team changed after transfer verification; lineup save stopped.")
        ensure_open(client, plan, policy)
        if not lineup_matches(current, plan):
            record("lineup_pending", "Saving XI, bench order, captain, vice captain and scoring chip.")
            try:
                client.request("POST", f"/my-team/{plan['team_id']}/", plan["lineup_request"])
            except FplActionError:
                if not lineup_matches(client.my_team(), plan):
                    raise
        current = client.my_team()
        if not lineup_matches(current, plan) or not transfers_match(current, plan):
            raise FplActionError("FPL live lineup/captain/chip read-back did not match; no success was reported.")
        record("applied", "Confirmed live squad, XI, bench order, captain, vice captain, chip and bank.")
        print(f"Applied GW{plan['gameweek']}: {len(plan['transfer_request']['transfers'])} free transfers; chip={plan['chip'] or 'none'}; captain saved.")
        return receipt


def run_live_actions(settings: dict, apply: bool = False, client=None) -> dict:
    own_client = client is None
    client = AuthenticatedFplClient.from_settings(settings) if own_client else client
    try:
        client.verify_identity()
        forecast = json.loads((PROJECT_ROOT / settings["paths"]["predictions_file"]).read_text())
        path = journal_path(settings, forecast["season_id"], forecast["gameweek"])
        if apply and path.exists():
            receipt = json.loads(path.read_text())
            result = execute_plan(client, receipt["plan"], settings)
        else:
            plan = build_plan(forecast, client.request("GET", "/bootstrap-static/"), client.request("GET", "/fixtures/"), client.my_team(), settings)
            private_json(PROJECT_ROOT / settings["automation"]["actions_dir"] / "latest_plan.json", plan)
            print(f"Live plan GW{plan['gameweek']}: {len(plan['transfer_request']['transfers'])} free transfers, chip={plan['chip'] or 'none'}.")
            result = execute_plan(client, plan, settings) if apply else {"status": "preview", "plan": plan}
        # Refresh the exact manager state for the dashboard after verified writes.
        from src.optimizer.manager import import_my_team
        from src.automation.plan import chip_state
        payload = client.my_team()
        active, _ = chip_state(payload, forecast["gameweek"])
        state = import_my_team(payload, settings["manager"]["team_id"], forecast["season_id"], forecast["gameweek"], active)
        private_json(PROJECT_ROOT / settings["paths"]["manager_state_file"], state)
        return result
    finally:
        if own_client:
            client.close()


def scheduled_window(settings: dict, bootstrap: dict, now: datetime | None = None) -> bool:
    from src.data.season_state import season_state
    deadline = season_state(bootstrap["events"]).next_deadline
    if not deadline:
        return False
    remaining = (timestamp(deadline) - (now or datetime.now(timezone.utc))).total_seconds()
    policy = settings["automation"]
    return policy["deadline_buffer_seconds"] < remaining <= policy["scheduled_apply_window_minutes"] * 60


def main() -> None:
    from src.pipeline import runtime_settings
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--scheduled-window", action="store_true", help="exit 0 inside the configured deadline window, else 2")
    args = parser.parse_args()
    settings = runtime_settings()
    try:
        if args.scheduled_window:
            import requests
            response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=30)
            response.raise_for_status()
            raise SystemExit(0 if scheduled_window(settings, response.json()) else 2)
        run_live_actions(settings, apply=args.apply)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"FPL execution stopped: {exc}\n")


if __name__ == "__main__":
    main()

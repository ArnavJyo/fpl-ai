"""Connect Google/FPL through normal Chrome; export its FPL session locally.

Run: python -m src.automation.login [--enable-actions]. Sign in yourself in
the dedicated Chrome window. No passwords or Google credentials are copied.
FPL's own access token is saved privately and never printed. Profiles/sessions are
ignored by Git. --enable-actions supplies the session to a GitHub secret.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import re
import subprocess
import time
from datetime import datetime, timezone

from src.common import PROJECT_ROOT, load_config
from src.automation.client import AuthenticatedFplClient, FplActionError, private_json


def existing_chrome_session(profile_name: str, team_id: int) -> dict:
    """Read only FPL's current OIDC record in an explicitly selected profile.

    Do not copy a Chrome profile or export Google/other-site credentials. Deleted
    or superseded login records never count as the current signed-in session.
    """
    from pathlib import Path
    from ccl_chromium_reader import ccl_chromium_localstorage
    if not re.fullmatch(r"Default|Profile [0-9]+", profile_name):
        raise FplActionError("Choose a Chrome folder such as 'Profile 1'; arbitrary paths are not accepted.")
    directory = Path.home() / "Library/Application Support/Google/Chrome" / profile_name / "Local Storage/leveldb"
    key = "oidc.user:https://account.premierleague.com/as:bfcbaf69-aade-4c1b-8f00-c1cb8a193030"
    with ccl_chromium_localstorage.LocalStoreDb(directory) as database:
        records = list(database.iter_records_for_script_key("https://fantasy.premierleague.com", key,
                                                           include_deletions=True, raise_on_no_result=False))
        if not records:
            raise FplActionError("That Chrome profile has no FPL login; open My Team and sign in first.")
        latest = max(records, key=lambda r: r.leveldb_seq_number)
        if not latest.is_live or not latest.value:
            raise FplActionError("That profile's FPL login was cleared; historical credentials were not used.")
        stored = json.loads(latest.value)
    return {"team_id": team_id, "saved_at": datetime.now(timezone.utc).isoformat(), "cookies": [],
            "access_token": stored["access_token"], "refresh_token": stored.get("refresh_token"),
            "expires_at": stored.get("expires_at", 0), "client_id": "bfcbaf69-aade-4c1b-8f00-c1cb8a193030"}


def save_connection(settings: dict, auth: dict, enable_actions: bool, repository: str) -> None:
    # An open Chrome tab may still have an older, unexpired access token whose
    # refresh token the bot already rotated. Keep the newer verified vault pair.
    from src.automation.session import configured_vault
    vault = configured_vault(settings)
    if vault:
        try:
            try:
                current = vault.load()
            except FplActionError:
                current = None  # A new verified login can repair an incomplete renewal.
            if current and current.get("expires_at", 0) >= auth.get("expires_at", 0):
                auth = current
        finally:
            vault.close()
    client = AuthenticatedFplClient(auth, settings["manager"]["team_id"])
    try:
        client.verify_identity()
        client.my_team()
        auth = client.auth  # Verification may renew a near-expiry rotating token.
    finally:
        client.close()
    private_json(PROJECT_ROOT / settings["automation"]["auth_file"], auth)
    print(f"Connected FPL entry {auth['team_id']}. Session saved locally and ignored by Git.", flush=True)
    if enable_actions:
        from src.automation.session import connect_actions_session
        connect_actions_session(settings, auth, repository)
        print("GitHub Actions renewable session connected. Automatic execution enabled.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-actions", action="store_true")
    parser.add_argument("--repository", default="ArnavJyo/fpl-ai")
    parser.add_argument("--timeout-minutes", type=int, default=15)
    parser.add_argument("--chrome-profile", help="import the existing FPL session from a specifically chosen macOS Chrome folder, e.g. 'Profile 1'")
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    settings = load_config()
    team_id = settings["manager"]["team_id"]
    if args.chrome_profile:
        save_connection(settings, existing_chrome_session(args.chrome_profile, team_id), args.enable_actions, args.repository)
        return
    profile = PROJECT_ROOT / ".local/fpl-browser"
    profile.mkdir(parents=True, exist_ok=True)
    os.chmod(profile.parent, 0o700)
    os.chmod(profile, 0o700)
    chrome = next((p for p in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", shutil.which("google-chrome"),
                              shutil.which("chromium")] if p and os.path.isfile(p)), None)
    if not chrome:
        raise FplActionError("Install Google Chrome for the interactive sign-in.")
    port_file = profile / "DevToolsActivePort"
    port_file.unlink(missing_ok=True)
    process = subprocess.Popen([chrome, f"--user-data-dir={profile}", "--remote-debugging-address=127.0.0.1",
                                "--remote-debugging-port=0", "--no-first-run", "https://fantasy.premierleague.com/my-team"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Sign in to FPL with Google in the new Chrome window, then open My Team. Waiting for your session…", flush=True)
    deadline = time.monotonic() + args.timeout_minutes * 60
    try:
        while not port_file.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                raise FplActionError("The dedicated Chrome window closed before login.")
            time.sleep(0.5)
        if not port_file.exists():
            raise FplActionError("Chrome did not start its local connection.")
        port = int(port_file.read_text().splitlines()[0])
        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
            context = browser.contexts[0]
            while time.monotonic() < deadline:
                page = next((p for p in context.pages if p.url.startswith("https://fantasy.premierleague.com")), None)
                if page:
                    try:
                        with page.expect_request(lambda r: r.url == "https://fantasy.premierleague.com/api/me/" and r.method == "GET", timeout=5000) as identity_request:
                            me = page.evaluate("""async () => {
                          const r = await fetch('/api/me/', {credentials: 'same-origin'});
                          return r.ok && r.headers.get('content-type')?.includes('application/json') ? await r.json() : null;
                        }""")
                        authorization = identity_request.value.header_value("x-api-authorization")
                        entry = ((me or {}).get("player") or {}).get("entry")
                        if entry:
                            if entry != team_id:
                                raise FplActionError(f"You signed into entry {entry}; the configured entry is {team_id}.")
                            auth = {"team_id": team_id, "saved_at": datetime.now(timezone.utc).isoformat(),
                                    "cookies": [c for c in context.cookies() if c["domain"].lstrip(".") in
                                                {"fantasy.premierleague.com", "premierleague.com"}]}
                            if authorization and authorization.startswith("Bearer "):
                                auth["access_token"] = authorization.removeprefix("Bearer ")
                            renewal = page.evaluate("""() => {
                              const u = JSON.parse(localStorage.getItem('oidc.user:https://account.premierleague.com/as:bfcbaf69-aade-4c1b-8f00-c1cb8a193030') || 'null');
                              return u ? {refresh_token: u.refresh_token, expires_at: u.expires_at} : {};
                            }""")
                            auth.update(renewal)
                            auth["client_id"] = "bfcbaf69-aade-4c1b-8f00-c1cb8a193030"
                            save_connection(settings, auth, args.enable_actions, args.repository)
                            return
                    except FplActionError:
                        raise
                    except Exception:
                        # Login redirects/navigation can interrupt a read. Never
                        # print page state or credentials while waiting.
                        pass
                time.sleep(2)
        raise FplActionError("Login timed out. Rerun the connection command when ready.")
    finally:
        process.terminate()


if __name__ == "__main__":
    try:
        main()
    except (FplActionError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from None

"""Connect Google/FPL through normal Chrome; export only FPL cookies locally.

Run: python -m src.automation.login [--enable-actions]. Sign in yourself in
the dedicated Chrome window. No passwords, Google cookies, or browser tokens
are copied into the project or printed. Browser profiles and FPL sessions are
ignored by Git. --enable-actions supplies the session to a GitHub secret.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone

from src.common import PROJECT_ROOT, load_config
from src.automation.client import AuthenticatedFplClient, FplActionError, private_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-actions", action="store_true")
    parser.add_argument("--repository", default="ArnavJyo/fpl-ai")
    parser.add_argument("--timeout-minutes", type=int, default=15)
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    settings = load_config()
    team_id = settings["manager"]["team_id"]
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
                        me = page.evaluate("""async () => {
                          const r = await fetch('/api/me/', {credentials: 'same-origin'});
                          return r.ok && r.headers.get('content-type')?.includes('application/json') ? await r.json() : null;
                        }""")
                        entry = ((me or {}).get("player") or {}).get("entry")
                        if entry:
                            if entry != team_id:
                                raise FplActionError(f"You signed into entry {entry}; the configured entry is {team_id}.")
                            auth = {"team_id": team_id, "saved_at": datetime.now(timezone.utc).isoformat(),
                                    "cookies": [c for c in context.cookies() if c["domain"].lstrip(".") in
                                                {"fantasy.premierleague.com", "premierleague.com"}]}
                            client = AuthenticatedFplClient(auth, team_id)
                            try:
                                client.verify_identity()
                                client.my_team()
                            finally:
                                client.close()
                            path = PROJECT_ROOT / settings["automation"]["auth_file"]
                            private_json(path, auth)
                            print(f"Connected FPL entry {team_id}. Session saved locally and ignored by Git.", flush=True)
                            if args.enable_actions:
                                subprocess.run(["gh", "secret", "set", "FPL_AUTH_JSON", "--repo", args.repository],
                                               input=json.dumps(auth), text=True, check=True)
                                subprocess.run(["gh", "variable", "set", "FPL_AUTOPILOT_ENABLED", "--body", "true",
                                                "--repo", args.repository], check=True)
                                print("GitHub Actions session connected. Automatic execution enabled.", flush=True)
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

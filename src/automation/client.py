"""Use an FPL browser session without storing a password or retrying writes.

The current official web app sends X-API-Authorization with its FPL access
token. Older cookie sessions additionally need X-CSRFToken. Keep this transport small so an
upstream API change fails clearly rather than guessing another write route.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests


class FplActionError(ValueError):
    """Authentication, state, or write verification failed."""


def private_json(path: Path, value: dict) -> None:
    """Atomically save private data with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


class AuthenticatedFplClient:
    base_url = "https://fantasy.premierleague.com/api"

    def __init__(self, auth: dict, team_id: int):
        self.auth = dict(auth)
        self.vault = None
        self.auth_path = None
        self.team_id = team_id
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "Origin": "https://fantasy.premierleague.com",
                                     "Referer": "https://fantasy.premierleague.com/en/my-team", "X-API-Language": "en"})
        if auth.get("team_id") != team_id:
            raise FplActionError("Saved FPL login belongs to a different configured team.")
        cookies = auth.get("cookies", [])
        self.access_token = auth.get("access_token")
        if self.access_token is not None and (not isinstance(self.access_token, str) or not self.access_token or any(c in self.access_token for c in '\r\n')):
            raise FplActionError("Saved FPL access token is invalid; reconnect the browser login.")
        if not isinstance(cookies, list) or (not cookies and not self.access_token):
            raise FplActionError("No authenticated FPL session. Run src.automation.login and sign in in the browser.")
        for cookie in cookies:
            domain = cookie["domain"].lstrip(".")
            if domain not in {"fantasy.premierleague.com", "premierleague.com"}:
                raise FplActionError("Session export contains a cookie outside the FPL domain.")
            expires = cookie.get("expires", -1)
            self.session.cookies.set(cookie["name"], cookie["value"], domain=cookie["domain"],
                                     path=cookie.get("path", "/"), secure=cookie.get("secure", True),
                                     expires=int(expires) if expires and expires > 0 else None)
        if not self.access_token:
            self.access_token = next((c["value"] for c in cookies if c["name"] == "ACCESS_TOKEN"), None)
        if self.access_token:
            self.session.headers["X-API-Authorization"] = "Bearer " + self.access_token

    @classmethod
    def from_settings(cls, settings: dict):
        from src.common import PROJECT_ROOT
        secret = os.environ.get("FPL_AUTH_JSON")
        source = PROJECT_ROOT / settings["automation"]["auth_file"]
        if not secret and not source.exists():
            raise FplActionError("FPL login is missing. Run .venv/bin/python -m src.automation.login.")
        try:
            auth = json.loads(secret) if secret else json.loads(source.read_text())
        except (ValueError, TypeError):
            raise FplActionError("Saved FPL session is invalid; sign in again.") from None
        from src.automation.session import configured_vault
        vault = configured_vault(settings)
        try:
            if vault:
                auth = vault.load() or auth
            client = cls(auth, settings["manager"]["team_id"])
        except Exception:
            if vault:
                vault.close()
            raise
        client.vault = vault
        client.auth_path = source
        return client

    def ensure_token(self) -> None:
        """Refresh before a request; never retry an account mutation after expiry."""
        if not self.auth.get("refresh_token") or self.auth.get("expires_at", 0) > time.time() + 120:
            return
        if self.auth.get("client_id") != "bfcbaf69-aade-4c1b-8f00-c1cb8a193030":
            raise FplActionError("Unsupported FPL login client; reconnect before renewal.")
        if self.vault:
            self.vault.begin_refresh(self.auth)
        try:
            response = requests.post("https://account.premierleague.com/as/token", data={"grant_type": "refresh_token",
                                     "client_id": self.auth["client_id"], "refresh_token": self.auth["refresh_token"]},
                                     timeout=30, allow_redirects=False)
            if not response.ok:
                raise ValueError
            data = response.json()
            token = data["access_token"]
            if not isinstance(token, str) or not token or any(c in token for c in '\r\n'):
                raise ValueError
            lifetime = int(data["expires_in"])
            if lifetime <= 0:
                raise ValueError
        except (requests.RequestException, ValueError, KeyError, TypeError):
            raise FplActionError("FPL session renewal failed; reconnect login. No account operation was retried.") from None
        self.auth.update(access_token=token, refresh_token=data.get("refresh_token", self.auth["refresh_token"]),
                         expires_at=time.time() + lifetime)
        # Persist rotating credentials before a subsequent FPL operation. The
        # repository holds ciphertext, and the key stays in the Actions secret.
        if self.auth_path:
            private_json(self.auth_path, self.auth)
        if self.vault:
            self.vault.save(self.auth)
        self.access_token = token
        self.session.headers["X-API-Authorization"] = "Bearer " + token

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        if path not in {"/me/", "/bootstrap-static/", "/fixtures/", "/transfers/", f"/my-team/{self.team_id}/"}:
            raise FplActionError("Unsupported FPL route.")
        self.ensure_token()
        headers = {}
        if method == "POST":
            csrf = next((c.value for c in self.session.cookies if c.name == "csrftoken"), None)
            if not csrf and not self.access_token:
                raise FplActionError("FPL session has no CSRF cookie; reconnect the browser login.")
            if csrf:
                headers["X-CSRFToken"] = csrf
        try:
            response = self.session.request(method, self.base_url + path, json=payload, headers=headers,
                                            timeout=30, allow_redirects=False)
        except requests.RequestException:
            # A timed-out POST may have succeeded. The journal and a GET decide
            # recovery; a second POST is never sent automatically here.
            raise FplActionError(f"FPL {method} response was not received. Read back team state before retrying.") from None
        if not response.ok or response.is_redirect:
            raise FplActionError(f"FPL {method} failed with HTTP {response.status_code}. No write retry was sent.")
        if response.status_code == 204 or not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            raise FplActionError("FPL returned HTML instead of JSON; the login may have expired.") from None

    def verify_identity(self) -> None:
        me = self.request("GET", "/me/")
        if (me.get("player") or {}).get("entry") != self.team_id:
            raise FplActionError("The signed-in FPL account does not own the configured entry.")

    def my_team(self) -> dict:
        return self.request("GET", f"/my-team/{self.team_id}/")

    def close(self) -> None:
        if self.vault:
            self.vault.close()
        self.session.close()

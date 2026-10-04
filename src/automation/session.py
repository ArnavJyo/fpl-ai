"""Renewable FPL sessions encrypted separately from account-write receipts.

FPL rotates refresh tokens. AES-GCM ciphertext lives on the private state
branch; its independent 256-bit key lives in a GitHub secret and an ignored
owner-only local file. Never commit plaintext tokens or that encryption key.
Content-SHA locking reserves renewal before using a refresh token, so another
runner cannot consume the same rotating token concurrently.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.automation.client import FplActionError, private_json
from src.automation.receipts import ReceiptStore
from src.common import PROJECT_ROOT


class SessionVault:
    def __init__(self, repository: str, team_id: int, key: str, store=None):
        try:
            decoded = bytes.fromhex(key)
            if len(decoded) != 32:
                raise ValueError
        except (TypeError, ValueError):
            raise FplActionError("FPL session encryption key is invalid; reconnect the browser login.") from None
        self.cipher = AESGCM(decoded)
        self.aad = f"fpl-ai-entry-{team_id}".encode()
        self.store = store if store is not None else ReceiptStore(repository, f"entry-{team_id}.json", session=True)

    def load(self) -> dict | None:
        value = self.store.load()
        if value is None:
            return None
        if value.get("status") != "ready":
            raise FplActionError("An earlier FPL session renewal is incomplete; reconnect login before account writes.")
        try:
            return json.loads(self.cipher.decrypt(base64.b64decode(value["nonce"]), base64.b64decode(value["ciphertext"]), self.aad))
        except Exception:
            raise FplActionError("Encrypted FPL session could not be verified; no account changes submitted.") from None

    def save(self, auth: dict, status="ready") -> None:
        nonce = os.urandom(12)
        encrypted = self.cipher.encrypt(nonce, json.dumps(auth, allow_nan=False).encode(), self.aad)
        envelope = {"format_version": 1, "status": status, "nonce": base64.b64encode(nonce).decode(),
                    "ciphertext": base64.b64encode(encrypted).decode()}
        self.store.save_value(envelope, "Persist encrypted FPL session: " + status)

    def begin_refresh(self, auth: dict):
        self.save(auth, status="renewing")

    def close(self):
        self.store.close()


def configured_vault(settings: dict) -> SessionVault | None:
    key_path = PROJECT_ROOT / settings["automation"].get("session_key_file", "data/fpl_session_key.json")
    key = os.environ.get("FPL_SESSION_KEY")
    if not key and key_path.exists():
        key = json.loads(key_path.read_text())["key"]
    if not key:
        return None
    repository = settings["automation"].get("receipt_repository")
    if not repository:
        raise FplActionError("Session renewal requires a configured private state repository.")
    return SessionVault(repository, settings["manager"]["team_id"], key)


def connect_actions_session(settings: dict, auth: dict, repository: str) -> None:
    """Call only after the signed-in FPL entry has been verified."""
    import subprocess
    path = PROJECT_ROOT / settings["automation"]["session_key_file"]
    key = json.loads(path.read_text())["key"] if path.exists() else os.urandom(32).hex()
    private_json(path, {"key": key})
    for name, value in [("FPL_SESSION_KEY", key), ("FPL_AUTH_JSON", json.dumps(auth))]:
        subprocess.run(["gh", "secret", "set", name, "--repo", repository], input=value, text=True, check=True)
    vault = SessionVault(repository, settings["manager"]["team_id"], key)
    try:
        vault.store.load()  # Acquire SHA; a new verified login can repair a pending/old session.
        vault.save(auth)
    finally:
        vault.close()
    subprocess.run(["gh", "variable", "set", "FPL_AUTOPILOT_ENABLED", "--body", "true", "--repo", repository], check=True)

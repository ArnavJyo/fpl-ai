"""Persist FPL action receipts in the private repo before any account write.

The fpl-automation-state branch stores plans/receipts, never browser sessions.
GitHub content SHAs provide optimistic locking in addition to job concurrency.
Local execution uses the existing gh login; hosted execution uses GH_TOKEN.
Failure to persist stops BEFORE submitting an FPL operation.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess

import requests

from src.automation.client import FplActionError


class ReceiptStore:
    branch = "fpl-automation-state"

    def __init__(self, repository: str, filename: str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or not re.fullmatch(r"receipt-[a-f0-9]+\.json", filename):
            raise FplActionError("Invalid receipt repository or filename.")
        token = os.environ.get("GH_TOKEN")
        if not token:
            result = subprocess.run(["gh", "auth", "token", "--hostname", "github.com"], capture_output=True, text=True)
            if result.returncode or not result.stdout.strip():
                raise FplActionError("GitHub login is required to persist team receipts before live execution.")
            token = result.stdout.strip()
        self.root = "https://api.github.com/repos/" + repository
        self.path = "receipts/" + filename
        self.sha = None
        self.branch_exists = False
        self.session = requests.Session()
        self.session.headers.update({"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
                                     "X-GitHub-Api-Version": "2022-11-28"})

    def request(self, method: str, route: str, body: dict | None = None, allow_missing=False):
        try:
            response = self.session.request(method, self.root + route, json=body, timeout=30, allow_redirects=False)
        except requests.RequestException:
            raise FplActionError("GitHub receipt persistence failed; no further FPL writes will be sent.") from None
        if allow_missing and response.status_code == 404:
            return None
        if not response.ok:
            raise FplActionError(f"GitHub receipt persistence returned HTTP {response.status_code}; stopped before another FPL write.")
        return response.json()

    def load(self) -> dict | None:
        branch = self.request("GET", f"/git/ref/heads/{self.branch}", allow_missing=True)
        self.branch_exists = branch is not None
        if not branch:
            return None
        result = self.request("GET", f"/contents/{self.path}?ref={self.branch}", allow_missing=True)
        if not result:
            return None
        self.sha = result["sha"]
        return json.loads(base64.b64decode(result["content"]))

    def save(self, receipt: dict) -> None:
        if not self.branch_exists:
            # Branch creation is done before committing a pending action. No FPL
            # write happens until both operations receive successful responses.
            main = self.request("GET", "/git/ref/heads/main")
            self.request("POST", "/git/refs", {"ref": "refs/heads/" + self.branch, "sha": main["object"]["sha"]})
            self.branch_exists = True
        content = base64.b64encode(json.dumps(receipt, indent=2, allow_nan=False).encode()).decode()
        body = {"branch": self.branch, "message": f"Record GW{receipt['plan']['gameweek']} execution: {receipt['status']}", "content": content}
        if self.sha:
            body["sha"] = self.sha
        result = self.request("PUT", "/contents/" + self.path, body)
        self.sha = result["content"]["sha"]

    def close(self):
        self.session.close()

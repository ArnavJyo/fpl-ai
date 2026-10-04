"""Fetch FPL JSON using a disk cache and bounded transient-failure retries.

Existing cache files have no expiry. Pass refresh=True to download again.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 4
USER_AGENT = "fpl-ai/0.1 (personal FPL analysis)"
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_transient(exc: requests.RequestException) -> bool:
    """Retry connection failures, timeouts, and selected server/rate-limit errors."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in RETRY_STATUS_CODES
    return False


class FplClient:
    def __init__(self, base_url: str, raw_dir: Path, refresh: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.raw_dir = raw_dir
        self.refresh = refresh
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.hits = 0
        self.misses = 0

    def get(self, path: str, cache_name: str) -> Any:
        """Return the parsed JSON for `path`, using the cache when possible."""
        cache_file = self.raw_dir / f"{cache_name}.json"
        if cache_file.exists() and not self.refresh:
            self.hits += 1
            return json.loads(cache_file.read_text())

        self.misses += 1
        payload = self._download(path)
        cache_file.write_text(payload)
        return json.loads(payload)

    def get_many(self, items: list[tuple[str, str]], max_workers: int = 12) -> list[Any]:
        """Fetch (path, cache_name) pairs concurrently, preserving input order."""
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda item: self.get(*item), items))

    def _download(self, path: str) -> str:
        """Fetch one URL, retrying transient failures with exponential backoff."""
        url = f"{self.base_url}{path}"
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self.session.get(url, timeout=TIMEOUT_SECONDS)
                response.raise_for_status()
                return response.text
            except requests.RequestException as exc:
                if not _is_transient(exc) or attempt == MAX_ATTEMPTS - 1:
                    raise
                backoff = 2**attempt
                print(f"  ! {url} failed ({exc}); retrying in {backoff}s")
                time.sleep(backoff)
        raise AssertionError("unreachable")

    def stats(self) -> str:
        return f"{self.hits} from cache, {self.misses} fetched"


def main() -> None:
    # Direct-file commands need the project root for package imports.
    if __package__ in (None, ""):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data.pipeline import stage_main

    stage_main("fetch")


if __name__ == "__main__":
    main()

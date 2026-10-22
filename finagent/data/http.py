"""Polite, cached HTTP for public financial data sources.

Responses are cached on disk with a time-to-live so repeated runs (and
classroom demos on flaky Wi-Fi) do not hammer SEC EDGAR or Yahoo, and so a run
can be replayed from the same data it was originally made with.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import requests

from ..config import Settings

_BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"


class DataSourceError(RuntimeError):
    """A data provider could not return what was asked for."""


class CachedHTTP:
    # SEC EDGAR fair-access policy: at most 10 requests per second.
    _min_interval = {"sec.gov": 0.12}

    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.cache_dir / "http"
        self.root.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._last_call: dict[str, float] = {}
        self._lock = threading.Lock()

    def _headers(self, url: str) -> dict:
        if "sec.gov" in url:
            return {"User-Agent": self.settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"}
        return {"User-Agent": _BROWSER_UA}

    def _throttle(self, url: str) -> None:
        for host, interval in self._min_interval.items():
            if host in url:
                with self._lock:
                    wait = self._last_call.get(host, 0) + interval - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                    self._last_call[host] = time.monotonic()

    def _path(self, url: str) -> Path:
        return self.root / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".cache")

    def get_text(self, url: str, ttl_s: float) -> str:
        path = self._path(url)
        if path.exists() and (ttl_s < 0 or time.time() - path.stat().st_mtime < ttl_s):
            return path.read_text(encoding="utf-8")

        last_error: Exception | None = None
        for attempt in range(3):
            self._throttle(url)
            try:
                resp = self._session.get(url, headers=self._headers(url), timeout=30)
            except requests.RequestException as err:
                last_error = err
            else:
                if resp.status_code == 200:
                    path.write_text(resp.text, encoding="utf-8")
                    return resp.text
                if resp.status_code == 404:
                    raise DataSourceError(f"not found: {url}")
                last_error = DataSourceError(f"HTTP {resp.status_code} from {url}")
            time.sleep(1.5 * (attempt + 1))

        if path.exists():  # stale data beats no data when a provider is down
            return path.read_text(encoding="utf-8")
        raise DataSourceError(str(last_error))

    def get_json(self, url: str, ttl_s: float):
        return json.loads(self.get_text(url, ttl_s))


_shared: dict[tuple, CachedHTTP] = {}


def http_for(settings: Settings) -> CachedHTTP:
    key = (str(settings.cache_dir), settings.sec_user_agent)
    if key not in _shared:
        _shared[key] = CachedHTTP(settings)
    return _shared[key]

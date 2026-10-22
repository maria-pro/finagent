"""Polite, cached HTTP for public financial data sources.

Responses are cached on disk with a time-to-live so repeated runs (and
classroom demos on flaky Wi-Fi) do not hammer SEC EDGAR or Yahoo, and so a run
can be replayed from the same data it was originally made with.

Every fetch is also recorded - URL, when the data was retrieved, and whether it
came from the network, a fresh cache entry or a *stale* cache entry served
because the provider was down - so each evidence record in a run can say exactly
which data it was computed from and how old it was (see ``record_fetches``).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

import requests

from ..config import Settings

_BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"

_fetch_log: ContextVar[list | None] = ContextVar("finagent_fetch_log", default=None)


class DataSourceError(RuntimeError):
    """A data provider could not return what was asked for."""


@contextmanager
def record_fetches():
    """Collect a record of every URL fetched (or served from cache) inside the block."""
    log: list[dict] = []
    token = _fetch_log.set(log)
    try:
        yield log
    finally:
        _fetch_log.reset(token)


def replay_fetches(entries: list[dict]) -> None:
    """Re-attach earlier fetch records, e.g. when a tool reuses an index built in a previous call."""
    for entry in entries:
        _note(entry["url"], entry["fetched_at"], entry["status"])


def _note(url: str, fetched_at: str, status: str) -> None:
    log = _fetch_log.get()
    if log is not None and not any(e["url"] == url for e in log):
        log.append({"url": url, "fetched_at": fetched_at, "status": status})


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        # Parallel agents can fetch the same URL; write-then-rename means a reader
        # sees either the old file or the new one, never a half-written one.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def get_text(self, url: str, ttl_s: float) -> str:
        path = self._path(url)
        if path.exists() and (ttl_s < 0 or time.time() - path.stat().st_mtime < ttl_s):
            _note(url, _iso(path.stat().st_mtime), "cache")
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
                    self._write_atomic(path, resp.text)
                    _note(url, _iso(time.time()), "network")
                    return resp.text
                if resp.status_code == 404:
                    raise DataSourceError(f"not found: {url}")
                last_error = DataSourceError(f"HTTP {resp.status_code} from {url}")
            time.sleep(1.5 * (attempt + 1))

        if path.exists():
            # Stale data beats no data when a provider is down - but the run must say so.
            _note(url, _iso(path.stat().st_mtime), "stale-cache")
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

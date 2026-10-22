import pytest
import requests

from finagent.data import http as http_mod
from finagent.data.http import CachedHTTP, DataSourceError, record_fetches


class _Resp:
    def __init__(self, status, text=""):
        self.status_code, self.text = status, text


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setattr(http_mod.time, "sleep", lambda s: None)
    return CachedHTTP(settings)


def test_network_then_cache_are_recorded(client, monkeypatch):
    monkeypatch.setattr(client._session, "get", lambda url, **kw: _Resp(200, "payload"))
    with record_fetches() as log:
        assert client.get_text("https://example.org/a", ttl_s=3600) == "payload"
        assert client.get_text("https://example.org/a", ttl_s=3600) == "payload"  # second read: cache, deduplicated
    assert [e["status"] for e in log] == ["network"]
    with record_fetches() as log:
        client.get_text("https://example.org/a", ttl_s=3600)
    assert log[0]["status"] == "cache" and log[0]["fetched_at"].endswith("Z")


def test_outage_serves_stale_cache_and_says_so(client, monkeypatch):
    monkeypatch.setattr(client._session, "get", lambda url, **kw: _Resp(200, "old"))
    client.get_text("https://example.org/b", ttl_s=3600)

    def down(url, **kw):
        raise requests.ConnectionError("provider down")

    monkeypatch.setattr(client._session, "get", down)
    with record_fetches() as log:
        assert client.get_text("https://example.org/b", ttl_s=0) == "old"
    assert log[0]["status"] == "stale-cache"
    with pytest.raises(DataSourceError):
        client.get_text("https://example.org/never-cached", ttl_s=0)


def test_cache_writes_are_atomic(client, monkeypatch):
    monkeypatch.setattr(client._session, "get", lambda url, **kw: _Resp(200, "x" * 10_000))
    client.get_text("https://example.org/c", ttl_s=3600)
    leftovers = [p.name for p in client.root.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
    assert client._path("https://example.org/c").read_text() == "x" * 10_000

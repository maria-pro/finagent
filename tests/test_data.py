"""Data-source parsing from canned payloads, configuration, and the dense-retrieval path."""

import copy

import numpy as np
import pandas as pd
import pytest

from finagent import config
from finagent.data import prices
from finagent.retrieval import Chunk, FilingIndex

YAHOO = {
    "chart": {
        "result": [
            {
                "meta": {"longName": "NVIDIA Corporation", "currency": "USD", "exchangeTimezoneName": "America/New_York"},
                "timestamp": [1790000000, 1790086400, 1790172800],
                "indicators": {"quote": [{"close": [100.0, 101.0, 102.0]}], "adjclose": [{"adjclose": [99.0, 100.0, 101.0]}]},
            }
        ],
        "error": None,
    }
}
TREASURY = 'Date,"4 WEEKS COUPON EQUIVALENT","13 WEEKS COUPON EQUIVALENT"\n10/21/2026,3.99,4.17\n10/20/2026,4.00,4.18\n'


class FakeHTTP:
    def __init__(self, json_payload=None, text=None, fail=False):
        self.json_payload, self.text, self.fail, self.urls = json_payload, text, fail, []

    def get_json(self, url, ttl):
        self.urls.append(url)
        return self.json_payload

    def get_text(self, url, ttl):
        self.urls.append(url)
        if self.fail:
            raise RuntimeError("down")
        return self.text


def test_price_history_parses_adjusted_closes(monkeypatch, settings):
    fake = FakeHTTP(json_payload=YAHOO)
    monkeypatch.setattr(prices, "http_for", lambda s: fake)
    series = prices.adjusted_close(settings, "brk.b")
    assert list(series) == [99.0, 100.0, 101.0] and series.name == "BRK-B"
    assert series.attrs["meta"]["name"] == "NVIDIA Corporation"
    assert "BRK-B" in fake.urls[0]


def test_missing_adjusted_closes_are_recorded(monkeypatch, settings):
    payload = copy.deepcopy(YAHOO)
    del payload["chart"]["result"][0]["indicators"]["adjclose"]
    monkeypatch.setattr(prices, "http_for", lambda s: FakeHTTP(json_payload=payload))
    series = prices.adjusted_close(settings, "KO")
    assert list(series) == [100.0, 101.0, 102.0] and series.attrs["meta"]["adjusted"] is False
    assert "dividends are not included" in prices.price_basis(series)


def test_closing_prices_are_the_unadjusted_series(monkeypatch, settings):
    monkeypatch.setattr(prices, "http_for", lambda s: FakeHTTP(json_payload=YAHOO))
    assert list(prices.closing_price(settings, "NVDA")) == [100.0, 101.0, 102.0]
    assert prices.price_basis(prices.adjusted_close(settings, "NVDA")) == "returns from dividend- and split-adjusted closes"


def test_price_history_errors_are_explicit(monkeypatch, settings):
    monkeypatch.setattr(prices, "http_for", lambda s: FakeHTTP(json_payload={"chart": {"result": None, "error": "Not Found"}}))
    with pytest.raises(prices.DataSourceError):
        prices.price_history(settings, "ZZZZ")


def test_risk_free_series_reads_treasury_csvs_per_year(monkeypatch, settings):
    fake = FakeHTTP(text=TREASURY)
    monkeypatch.setattr(prices, "http_for", lambda s: fake)
    series = prices.risk_free_series(settings, pd.Timestamp.today() - pd.Timedelta(days=400))
    assert series.iloc[-1] == pytest.approx(0.0417) and "US Treasury" in series.attrs["source"]
    assert len(fake.urls) == 2  # this year's and last year's file
    assert prices.risk_free_rate(settings) == pytest.approx(0.0417)


def test_risk_free_series_falls_back_to_irx_then_empty(monkeypatch, settings):
    monkeypatch.setattr(prices, "http_for", lambda s: FakeHTTP(fail=True))
    irx = pd.Series([4.06, 4.07], index=pd.to_datetime(["2026-10-20", "2026-10-21"]))
    monkeypatch.setattr(prices, "adjusted_close", lambda s, t, r: irx)
    series = prices.risk_free_series(settings, "2026-10-01")
    assert series.iloc[-1] == pytest.approx(0.0407) and "IRX" in series.attrs["source"]

    def no_irx(*args):
        raise RuntimeError("no data")

    monkeypatch.setattr(prices, "adjusted_close", no_irx)
    assert prices.risk_free_series(settings, "2026-10-01").empty
    assert prices.risk_free_rate(settings) == 0.0


def test_dense_retrieval_is_fused_with_bm25(monkeypatch, settings):
    chunks = [Chunk("a#0", "a", "Revenue grew in the data center."), Chunk("b#0", "b", "Export controls may reduce demand in China.")]
    vectors = {"passage": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), "query": np.array([[0.0, 1.0]], dtype=np.float32)}
    monkeypatch.setattr(FilingIndex, "_embed", lambda self, texts, input_type: vectors[input_type])
    index = FilingIndex(chunks, settings.with_overrides(embed_model="nvidia/test-embed"), "acc-1")
    assert index.mode == "BM25 + nvidia/test-embed (RRF)"
    assert index.search("trade restrictions", k=1)[0][0].ref == "b#0"  # no word overlap: dense finds it
    cached = FilingIndex(chunks, settings.with_overrides(embed_model="nvidia/test-embed"), "acc-1")
    assert cached._dense is not None  # second build loads the saved vectors


def test_failed_embeddings_fall_back_to_bm25(monkeypatch, settings):
    def broken(self, texts, input_type):
        raise ConnectionError("embedding endpoint down")

    monkeypatch.setattr(FilingIndex, "_embed", broken)
    index = FilingIndex([Chunk("a#0", "a", "Export controls")], settings.with_overrides(embed_model="nvidia/test-embed"), "acc-2")
    assert index.mode == "BM25 (embeddings unavailable: ConnectionError)"
    assert index.search("export controls")[0][0].ref == "a#0"


def test_profiles_and_environment_overrides(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
    monkeypatch.setenv("BEDROCK_API_KEY", "test-key")
    bedrock = config.load_settings("bedrock")
    assert bedrock.base_url == "https://bedrock-runtime.ap-southeast-2.amazonaws.com/openai/v1"
    assert bedrock.api_key == "test-key" and bedrock.reasoning_style == "effort"
    monkeypatch.setenv("FINAGENT_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("FINAGENT_REASONING", "yes")
    monkeypatch.setenv("FINAGENT_TEMPERATURE", "0.6")
    local = config.load_settings("local")
    assert local.max_concurrency == 3 and local.reasoning is True and local.temperature == 0.6
    assert local.is_local and not bedrock.is_local
    with pytest.raises(ValueError):
        config.load_settings("nope")
    with pytest.raises(ValueError):
        config.parse_ablations("facts,typo")

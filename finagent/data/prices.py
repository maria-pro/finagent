"""Daily price history (dividend- and split-adjusted) and risk-free rates."""

from __future__ import annotations

import io
from urllib.parse import quote

import pandas as pd

from ..config import Settings
from .http import DataSourceError, http_for

_CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    "?range={range}&interval=1d&includeAdjustedClose=true&events=div%2Csplit"
)
_TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_bill_rates&field_tdr_date_value={year}&_format=csv"
)
_PRICE_TTL_S = 6 * 3600


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def price_history(settings: Settings, ticker: str, range_: str = "10y") -> pd.DataFrame:
    """Return a DataFrame indexed by trading date with ``close`` and ``adj_close`` columns."""
    ticker = normalize_ticker(ticker)
    payload = http_for(settings).get_json(_CHART_URL.format(ticker=quote(ticker), range=range_), _PRICE_TTL_S)
    chart = payload.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise DataSourceError(f"no price data for {ticker}: {chart.get('error')}")

    result = chart["result"][0]
    timestamps = result.get("timestamp") or []
    if not timestamps:
        raise DataSourceError(f"empty price history for {ticker}")
    bars = result["indicators"]["quote"][0]
    adj = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose") or bars["close"]
    tz = result["meta"].get("exchangeTimezoneName", "America/New_York")

    index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(tz).normalize().tz_localize(None)
    frame = pd.DataFrame({"close": bars["close"], "adj_close": adj}, index=index)
    frame.index.name = "date"
    frame = frame[~frame.index.duplicated(keep="last")].dropna()
    frame.attrs["meta"] = {
        "ticker": ticker,
        "name": result["meta"].get("longName") or result["meta"].get("shortName") or ticker,
        "currency": result["meta"].get("currency", "USD"),
        "exchange": result["meta"].get("fullExchangeName", ""),
    }
    return frame


def adjusted_close(settings: Settings, ticker: str, range_: str = "10y") -> pd.Series:
    frame = price_history(settings, ticker, range_)
    series = frame["adj_close"].rename(normalize_ticker(ticker))
    series.attrs = frame.attrs
    return series


def risk_free_series(settings: Settings, start) -> pd.Series:
    """Daily 13-week US Treasury bill yield (annualised decimal) from ``start`` to today.

    The primary source is the US Treasury's daily bill-rate CSVs (coupon-equivalent
    yield, one file per year); Yahoo's ^IRX index is the fallback. Returns an empty
    series if neither answers. Using the dated series - rather than today's rate for
    every day - matters for Sharpe ratios over windows in which rates moved a lot.
    """
    start = pd.Timestamp(start)
    today = pd.Timestamp.today()
    http = http_for(settings)
    parts = []
    for year in range(start.year, today.year + 1):
        # Files for older years no longer change; recent ones are refreshed daily.
        ttl = 24 * 3600 if year >= today.year - 1 else -1
        try:
            frame = pd.read_csv(io.StringIO(http.get_text(_TREASURY_URL.format(year=year), ttl)))
            column = next(c for c in frame.columns if "13 WEEKS COUPON" in c.upper())
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy() / 100.0
            parts.append(pd.Series(values, index=pd.to_datetime(frame["Date"], format="%m/%d/%Y")).dropna())
        except Exception:
            continue
    if parts:
        series = pd.concat(parts).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        series.attrs["source"] = "US Treasury daily bill rates (13-week, coupon equivalent)"
    else:
        try:
            series = adjusted_close(settings, "^IRX", "10y") / 100.0
            series.attrs["source"] = "Yahoo Finance ^IRX (13-week T-bill discount rate)"
        except Exception:
            return pd.Series(dtype=float)
    return series[series.index >= start - pd.Timedelta(days=14)]


def risk_free_rate(settings: Settings) -> float:
    """Latest 13-week US Treasury bill yield as a decimal; 0.0 if no source answers."""
    series = risk_free_series(settings, pd.Timestamp.today() - pd.Timedelta(days=30))
    return float(series.iloc[-1]) if len(series) else 0.0

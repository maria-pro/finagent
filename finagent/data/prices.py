"""Daily price history (dividend- and split-adjusted) and the risk-free rate."""

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


def risk_free_rate(settings: Settings) -> float:
    """Latest 13-week US Treasury bill yield as a decimal; 0.0 if no source answers.

    Primary source is the US Treasury's daily bill-rate CSV (coupon-equivalent
    yield); Yahoo's ^IRX index is the fallback.
    """
    http = http_for(settings)
    year = pd.Timestamp.today().year
    for y in (year, year - 1):
        try:
            text = http.get_text(
                "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                f"daily-treasury-rates.csv/{y}/all?type=daily_treasury_bill_rates&field_tdr_date_value={y}&_format=csv",
                24 * 3600,
            )
            frame = pd.read_csv(io.StringIO(text))
            column = next(c for c in frame.columns if "13 WEEKS COUPON" in c.upper())
            frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y")
            latest = frame.sort_values("Date")[column].dropna()
            if len(latest):
                return float(latest.iloc[-1]) / 100.0
        except Exception:
            continue
    try:
        return float(adjusted_close(settings, "^IRX", "1mo").iloc[-1]) / 100.0
    except Exception:
        return 0.0

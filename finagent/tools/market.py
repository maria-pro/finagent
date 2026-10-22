from __future__ import annotations

import pandas as pd

from .. import analytics as an
from ..data.prices import adjusted_close, risk_free_rate
from . import TICKER, Tool, ToolContext, parse_tickers, pct, rnd

_PERIOD_DAYS = {"1y": 365, "3y": 3 * 365, "5y": 5 * 365}


def _period(value: str | None, default: str) -> str:
    v = (value or default).lower().replace(" ", "").replace("years", "y").replace("year", "y")
    return v if v in _PERIOD_DAYS else default


def _window(series, period: str):
    start = series.index[-1] - pd.Timedelta(days=_PERIOD_DAYS[period])
    return series.loc[series.index >= start]


def price_performance(ctx: ToolContext, ticker: str, period: str = "1y") -> dict:
    ticker = parse_tickers(ticker)[0]
    period = _period(period, "1y")
    full = adjusted_close(ctx.settings, ticker)
    prices = _window(full, period)
    returns = an.daily_returns(prices)
    rf = risk_free_rate(ctx.settings)
    dd = an.max_drawdown(prices)
    last_year = _window(full, "1y")
    sma200 = full.rolling(200).mean().iloc[-1]
    out = {
        "ticker": ticker,
        "company": full.attrs.get("meta", {}).get("name", ticker),
        "as_of": prices.index[-1].date().isoformat(),
        "period": period,
        "last_price": rnd(prices.iloc[-1]),
        "total_return_pct": pct(an.total_return(prices)),
    }
    if period != "1y":
        out["annualized_return_pct"] = pct(an.cagr(prices))
    out.update(
        {
            "volatility_pct": pct(an.annualized_volatility(returns)),
            "sharpe_ratio": rnd(an.sharpe_ratio(returns, rf)),
            "max_drawdown_pct": pct(dd.max_drawdown),
            "drawdown_peak_date": dd.peak.date().isoformat(),
            "drawdown_trough_date": dd.trough.date().isoformat(),
            "high_52w": rnd(last_year.max()),
            "pct_below_52w_high": pct(1 - prices.iloc[-1] / last_year.max()),
            "sma_200": rnd(sma200),
            "price_vs_sma_200": "above" if prices.iloc[-1] > sma200 else "below",
            "rsi_14": rnd(an.rsi(full), 1),
            "risk_free_rate_pct": pct(rf),
        }
    )
    facts = [f"{ticker} {period} total return: {out['total_return_pct']}%"]
    if "annualized_return_pct" in out:
        facts.append(f"{ticker} {period} annualized return: {out['annualized_return_pct']}%")
    facts += [
        f"{ticker} {period} annualized volatility: {out['volatility_pct']}%",
        f"{ticker} {period} Sharpe ratio: {out['sharpe_ratio']}",
        f"{ticker} {period} max drawdown: {out['max_drawdown_pct']}% ({out['drawdown_peak_date']} to {out['drawdown_trough_date']})",
        f"{ticker} price: ${out['last_price']} on {out['as_of']}, {out['pct_below_52w_high']}% below the 52-week high of ${out['high_52w']}",
        f"{ticker} price is {out['price_vs_sma_200']} its 200-day average of ${out['sma_200']}",
    ]
    out["facts"] = facts
    return out


def benchmark_comparison(ctx: ToolContext, ticker: str, benchmark: str = "SPY", period: str = "3y") -> dict:
    ticker = parse_tickers(ticker)[0]
    benchmark = parse_tickers(benchmark or "SPY")[0]
    period = _period(period, "3y")
    asset = _window(adjusted_close(ctx.settings, ticker), period)
    bench = _window(adjusted_close(ctx.settings, benchmark), period)
    stats = an.benchmark_stats(an.daily_returns(asset), an.daily_returns(bench))
    asset_ret, bench_ret = an.total_return(asset), an.total_return(bench)
    out = {
        "ticker": ticker,
        "benchmark": benchmark,
        "period": period,
        "beta": rnd(stats.beta),
        "correlation": rnd(stats.correlation),
        "alpha_pct_per_year": pct(stats.alpha_annual),
        "tracking_error_pct": pct(stats.tracking_error),
        "information_ratio": rnd(stats.information_ratio),
        "ticker_return_pct": pct(asset_ret),
        "benchmark_return_pct": pct(bench_ret),
        "excess_return_pct": pct(asset_ret - bench_ret),
    }
    out["facts"] = [
        f"{ticker} {period} beta vs {benchmark}: {out['beta']}",
        f"{ticker} {period} correlation with {benchmark}: {out['correlation']}",
        f"{period} return: {ticker} {out['ticker_return_pct']}% vs {benchmark} {out['benchmark_return_pct']}% "
        f"(excess {out['excess_return_pct']}%)",
        f"{ticker} {period} annualized alpha vs {benchmark}: {out['alpha_pct_per_year']}%",
    ]
    return out


TOOLS = [
    Tool(
        "price_performance",
        "Price performance of a stock: total return, volatility, Sharpe ratio, max drawdown, 52-week high, trend.",
        {
            "type": "object",
            "properties": {"ticker": TICKER, "period": {"type": "string", "enum": ["1y", "3y", "5y"]}},
            "required": ["ticker"],
        },
        price_performance,
    ),
    Tool(
        "benchmark_comparison",
        "Compare a stock with a benchmark (default SPY): beta, correlation, alpha, excess return.",
        {
            "type": "object",
            "properties": {
                "ticker": TICKER,
                "benchmark": {"type": "string", "description": "Benchmark ticker, default SPY"},
                "period": {"type": "string", "enum": ["1y", "3y", "5y"]},
            },
            "required": ["ticker"],
        },
        benchmark_comparison,
    ),
]

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import analytics as an
from ..accel import array_backend
from ..data.prices import adjusted_close
from . import TICKERS, Tool, ToolContext, parse_tickers, parse_weights, pct, rnd

SIM_PATHS = 100_000
LOOKBACK = "5y"


def _returns(ctx: ToolContext, tickers: list[str], range_: str = LOOKBACK) -> pd.DataFrame:
    series = [adjusted_close(ctx.settings, t, range_) for t in tickers]
    frame = pd.concat(series, axis=1, join="inner")
    frame.columns = tickers
    returns = frame.pct_change().dropna()
    unadjusted = [t for t, s in zip(tickers, series) if not s.attrs.get("meta", {}).get("adjusted", True)]
    returns.attrs["price_basis"] = (
        f"returns from actual closes for {', '.join(unadjusted)} (no adjusted closes from the provider, so dividends are "
        "not included); dividend- and split-adjusted closes otherwise"
        if unadjusted
        else "returns from dividend- and split-adjusted closes"
    )
    return returns


def _label(portfolio: dict) -> str:
    if len(portfolio) == 1:
        return next(iter(portfolio))
    return "Portfolio (" + ", ".join(f"{w:.0%} {t}" for t, w in portfolio.items()) + ")"


def value_at_risk(ctx: ToolContext, tickers, weights=None, confidence: float = 0.95, horizon_days: int = 10) -> dict:
    tickers = parse_tickers(tickers)
    w = np.array(parse_weights(weights, len(tickers)))
    confidence = float(confidence)
    if confidence > 1:  # "95" -> 0.95
        confidence /= 100
    confidence = min(max(confidence, 0.8), 0.995)
    horizon_days = max(1, min(int(horizon_days), 60))

    returns = _returns(ctx, tickers)
    hist = an.historical_var(an.portfolio_returns(returns, w), confidence, horizon_days)
    xp, backend = array_backend()
    fhs = an.filtered_historical_simulation(
        returns.to_numpy(), w, horizon_days=horizon_days, n_paths=SIM_PATHS, confidence=confidence, xp=xp, backend=backend
    )
    years = round(len(returns) / an.TRADING_DAYS)
    out = {
        "portfolio": {t: rnd(x, 3) for t, x in zip(tickers, w)},
        "confidence_pct": rnd(confidence * 100, 1),
        "horizon_days": horizon_days,
        "lookback": f"{len(returns)} trading days",
        # Two different views of the same history: unconditional (every day weighs the
        # same) and conditional (today's volatility regime, via EWMA filtering).
        "historical_var_pct": pct(hist.var),
        "historical_cvar_pct": pct(hist.cvar),
        "fhs_var_pct": pct(fhs.var),
        "fhs_cvar_pct": pct(fhs.cvar),
        "fhs_prob_loss_over_10pct": pct(fhs.prob_loss_over_10pct),
        "current_vol_pct": pct(fhs.current_vol_annual),
        "average_vol_pct": pct(fhs.average_vol_annual),
        "simulated_paths": fhs.n_paths,
        "method": f"historical VaR (overlapping {horizon_days}-day returns) and {fhs.method}",
        "price_basis": returns.attrs["price_basis"],
        "compute_backend": fhs.backend,
        "compute_ms": rnd(fhs.runtime_ms, 1),
    }
    who = _label(out["portfolio"])
    c, h = f"{out['confidence_pct']:g}%", f"{horizon_days}-day"
    out["facts"] = [
        f"{who} {c} {h} Value at Risk, historical (based on {len(returns)} trading days of returns): {out['historical_var_pct']}% of value",
        f"{who} {c} {h} expected shortfall, historical: {out['historical_cvar_pct']}%",
        f"{who} {c} {h} Value at Risk, filtered historical simulation scaled to current volatility: {out['fhs_var_pct']}%",
        f"{who} {c} {h} expected shortfall, filtered historical simulation: {out['fhs_cvar_pct']}%",
        f"{who} current volatility {out['current_vol_pct']}% vs {years}-year average {out['average_vol_pct']}% (annualized)",
        f"{who} probability of losing more than 10% over {horizon_days} days (filtered historical simulation): {out['fhs_prob_loss_over_10pct']}%",
    ]
    return out


def stress_test(ctx: ToolContext, tickers, weights=None) -> dict:
    tickers = parse_tickers(tickers)
    w = np.array(parse_weights(weights, len(tickers)))
    returns = _returns(ctx, tickers, "10y")
    # Compound the weighted daily returns, i.e. rebalance to the stated weights every day. A
    # buy-and-hold index started ten years ago would have drifted far from them by any crash.
    portfolio_returns = an.portfolio_returns(returns, w)
    portfolio = (1 + portfolio_returns).cumprod().rename("portfolio")
    spy = adjusted_close(ctx.settings, "SPY", "10y")
    episodes = an.stress_test(portfolio, spy)
    beta = an.benchmark_stats(portfolio_returns.iloc[-756:], an.daily_returns(spy)).beta
    out = {
        "portfolio": {t: rnd(x, 3) for t, x in zip(tickers, w)},
        "episodes": [
            {"episode": e["episode"], "portfolio_return_pct": pct(e["return"]), "spy_return_pct": pct(e["benchmark_return"])}
            for e in episodes
        ],
        "beta_vs_spy_3y": rnd(beta),
        "beta_implied_move_if_spy_falls_20pct": pct(-0.20 * beta),
        "method": "daily returns compounded over each window" + (", rebalanced daily to the stated weights" if len(tickers) > 1 else ""),
        "price_basis": returns.attrs["price_basis"],
    }
    who = _label(out["portfolio"])
    out["facts"] = [
        f"{who} in the {e['episode']} ({e['start']} to {e['end']}): {pct(e['return'])}% vs SPY {pct(e['benchmark_return'])}%"
        for e in episodes
    ] + [f"{who} 3-year beta vs SPY: {out['beta_vs_spy_3y']}; a 20% SPY fall implies about {out['beta_implied_move_if_spy_falls_20pct']}%"]
    return out


TOOLS = [
    Tool(
        "value_at_risk",
        "Value at Risk for one stock, or for a portfolio when several tickers are given: historical VaR and "
        "filtered historical simulation scaled to current volatility.",
        {
            "type": "object",
            "properties": {
                "tickers": TICKERS,
                "weights": {"type": "array", "items": {"type": "number"}, "description": "Portfolio weights; equal if omitted"},
                "confidence": {"type": "number", "description": "e.g. 0.95"},
                "horizon_days": {"type": "integer", "description": "Trading days, default 10"},
            },
            "required": ["tickers"],
        },
        value_at_risk,
    ),
    Tool(
        "stress_test",
        "How one stock (or a portfolio of several tickers) performed in past market crashes vs SPY, plus beta-implied loss.",
        {
            "type": "object",
            "properties": {
                "tickers": TICKERS,
                "weights": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["tickers"],
        },
        stress_test,
    ),
]

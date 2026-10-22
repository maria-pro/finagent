from __future__ import annotations

import numpy as np
import pandas as pd

from .. import analytics as an
from ..accel import array_backend
from ..data.prices import adjusted_close
from . import TICKERS, Tool, ToolContext, parse_tickers, parse_weights, pct, rnd

MC_PATHS = 100_000


def _returns(ctx: ToolContext, tickers: list[str], range_: str = "5y") -> pd.DataFrame:
    frame = pd.concat([adjusted_close(ctx.settings, t, range_) for t in tickers], axis=1, join="inner")
    frame.columns = tickers
    return frame.pct_change().dropna()


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
    mc = an.monte_carlo_var(
        returns.to_numpy(), w, horizon_days=horizon_days, n_paths=MC_PATHS, confidence=confidence, xp=xp, backend=backend
    )
    out = {
        "portfolio": {t: rnd(x, 3) for t, x in zip(tickers, w)},
        "confidence_pct": rnd(confidence * 100, 1),
        "horizon_days": horizon_days,
        "lookback": f"{len(returns)} trading days",
        "historical_var_pct": pct(hist.var),
        "historical_cvar_pct": pct(hist.cvar),
        "monte_carlo_var_pct": pct(mc.var),
        "monte_carlo_cvar_pct": pct(mc.cvar),
        "prob_loss_over_10pct": pct(mc.prob_loss_over_10pct),
        "simulated_paths": mc.n_paths,
        "compute_backend": mc.backend,
        "compute_ms": rnd(mc.runtime_ms, 1),
    }
    who = _label(out["portfolio"])
    c, h = f"{out['confidence_pct']:g}%", f"{horizon_days}-day"
    out["facts"] = [
        f"{who} {c} {h} Value at Risk (historical): {out['historical_var_pct']}% of value",
        f"{who} {c} {h} expected shortfall (historical): {out['historical_cvar_pct']}%",
        f"{who} {c} {h} Value at Risk (Monte Carlo, {MC_PATHS:,} paths): {out['monte_carlo_var_pct']}%",
        f"{who} {c} {h} expected shortfall (Monte Carlo): {out['monte_carlo_cvar_pct']}%",
        f"{who} probability of losing more than 10% over {horizon_days} days: {out['prob_loss_over_10pct']}%",
    ]
    return out


def stress_test(ctx: ToolContext, tickers, weights=None) -> dict:
    tickers = parse_tickers(tickers)
    w = np.array(parse_weights(weights, len(tickers)))
    prices = pd.concat([adjusted_close(ctx.settings, t, "10y") for t in tickers], axis=1, join="inner")
    prices.columns = tickers
    portfolio = (prices / prices.iloc[0] @ w).rename("portfolio")
    spy = adjusted_close(ctx.settings, "SPY", "10y")
    episodes = an.stress_test(portfolio, spy)
    beta = an.benchmark_stats(an.daily_returns(portfolio.iloc[-756:]), an.daily_returns(spy.iloc[-756:])).beta
    out = {
        "portfolio": {t: rnd(x, 3) for t, x in zip(tickers, w)},
        "episodes": [
            {"episode": e["episode"], "portfolio_return_pct": pct(e["return"]), "spy_return_pct": pct(e["benchmark_return"])}
            for e in episodes
        ],
        "beta_vs_spy_3y": rnd(beta),
        "beta_implied_move_if_spy_falls_20pct": pct(-0.20 * beta),
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
        "Value at Risk for one stock, or for a portfolio when several tickers are given: historical and Monte Carlo VaR/CVaR.",
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

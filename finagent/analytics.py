"""Deterministic finance math. The language models never do arithmetic - they
call tools built on these functions, and every number they quote must come from
here (see ``provenance.py``)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Market stress episodes used for historical stress tests (peak-to-trough of the S&P 500).
STRESS_WINDOWS = {
    "2018 Q4 sell-off": ("2018-09-20", "2018-12-24"),
    "2020 COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate-hike bear market": ("2022-01-03", "2022-10-12"),
}


def daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def total_return(prices: pd.Series) -> float:
    return float(prices.iloc[-1] / prices.iloc[0] - 1.0)


def cagr(prices: pd.Series) -> float:
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    return float((prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1.0) if years > 0 else float("nan")


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    excess = returns - risk_free_annual / TRADING_DAYS
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


@dataclass
class Drawdown:
    max_drawdown: float  # negative fraction, e.g. -0.35
    peak: pd.Timestamp
    trough: pd.Timestamp


def max_drawdown(prices: pd.Series) -> Drawdown:
    running_peak = prices.cummax()
    dd = prices / running_peak - 1.0
    trough = dd.idxmin()
    peak = prices.loc[:trough].idxmax()
    return Drawdown(float(dd.min()), peak, trough)


def drawdown_series(prices: pd.Series) -> pd.Series:
    return prices / prices.cummax() - 1.0


def rsi(prices: pd.Series, window: int = 14) -> float:
    """Wilder's relative strength index of the latest bar."""
    delta = prices.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    rs = gain.iloc[-1] / loss.iloc[-1]
    return float(100 - 100 / (1 + rs))


@dataclass
class BenchmarkStats:
    beta: float
    correlation: float
    alpha_annual: float
    tracking_error: float
    information_ratio: float


def benchmark_stats(asset: pd.Series, bench: pd.Series) -> BenchmarkStats:
    """CAPM-style statistics from aligned daily returns."""
    frame = pd.concat([asset, bench], axis=1, join="inner").dropna()
    a, b = frame.iloc[:, 0], frame.iloc[:, 1]
    beta = float(np.cov(a, b, ddof=1)[0, 1] / np.var(b, ddof=1))
    alpha_daily = a.mean() - beta * b.mean()
    active = a - b
    te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ir = float(active.mean() * TRADING_DAYS / te) if te > 0 else float("nan")
    return BenchmarkStats(beta, float(a.corr(b)), float(alpha_daily * TRADING_DAYS), te, ir)


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return returns.dropna() @ weights


@dataclass
class VaRResult:
    var: float  # loss as a positive fraction of portfolio value
    cvar: float  # expected loss beyond VaR (expected shortfall)


def historical_var(returns: pd.Series, confidence: float = 0.95, horizon_days: int = 1) -> VaRResult:
    """Empirical VaR from overlapping ``horizon_days`` compounded returns (no sqrt-time scaling)."""
    if horizon_days > 1:
        horizon = (1 + returns).rolling(horizon_days).apply(np.prod, raw=True).dropna() - 1
    else:
        horizon = returns
    cutoff = np.quantile(horizon, 1 - confidence)
    tail = horizon[horizon <= cutoff]
    return VaRResult(float(-cutoff), float(-tail.mean()))


@dataclass
class MonteCarloResult:
    var: float
    cvar: float
    prob_loss_over_10pct: float
    percentiles: dict[str, float]
    n_paths: int
    horizon_days: int
    backend: str
    runtime_ms: float
    samples: np.ndarray | None = None  # horizon returns, kept for charts


def monte_carlo_var(
    returns: np.ndarray,
    weights: np.ndarray,
    *,
    horizon_days: int = 10,
    n_paths: int = 100_000,
    confidence: float = 0.95,
    seed: int = 7,
    xp=np,
    backend: str = "CPU (NumPy)",
    keep_samples: bool = False,
) -> MonteCarloResult:
    """Bootstrap Monte Carlo VaR.

    Each simulated day re-samples one historical day's *vector* of returns, which
    keeps the cross-asset correlation and fat tails of the real data instead of
    assuming a normal distribution. ``xp`` is the array module: NumPy on CPU or
    CuPy on an NVIDIA GPU - the code is identical.
    """
    started = time.perf_counter()
    hist = xp.asarray(returns, dtype=xp.float32)
    w = xp.asarray(weights, dtype=xp.float32)
    daily_portfolio = hist @ w  # (n_days,)
    rng = xp.random.default_rng(seed) if hasattr(xp.random, "default_rng") else xp.random
    idx = rng.integers(0, daily_portfolio.shape[0], size=(n_paths, horizon_days))
    log_growth = xp.log1p(daily_portfolio[idx]).sum(axis=1)
    horizon_returns = xp.expm1(log_growth)

    cutoff = xp.quantile(horizon_returns, 1 - confidence)
    tail = horizon_returns[horizon_returns <= cutoff]
    pct = {f"p{p}": float(xp.quantile(horizon_returns, p / 100)) for p in (1, 5, 50, 95, 99)}
    result = MonteCarloResult(
        var=float(-cutoff),
        cvar=float(-tail.mean()),
        prob_loss_over_10pct=float((horizon_returns < -0.10).mean()),
        percentiles=pct,
        n_paths=n_paths,
        horizon_days=horizon_days,
        backend=backend,
        runtime_ms=0.0,
        samples=None,
    )
    if keep_samples:
        result.samples = horizon_returns.get() if hasattr(horizon_returns, "get") else np.asarray(horizon_returns)
    result.runtime_ms = (time.perf_counter() - started) * 1000
    return result


def stress_test(prices: pd.Series, bench: pd.Series) -> list[dict]:
    """Return of ``prices`` vs ``bench`` over each historical stress window it has data for."""
    rows = []
    for name, (start, end) in STRESS_WINDOWS.items():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if prices.index[0] > s or bench.index[0] > s:
            continue
        p = prices.loc[s:e]
        b = bench.loc[s:e]
        if len(p) < 5 or len(b) < 5:
            continue
        rows.append({"episode": name, "start": start, "end": end, "return": total_return(p), "benchmark_return": total_return(b)})
    return rows

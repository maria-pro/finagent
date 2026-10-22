"""Deterministic finance math. The language models never do arithmetic - they
call tools built on these functions, and every number they quote must come from
here (see ``provenance.py``)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EWMA_LAMBDA = 0.94  # RiskMetrics decay for daily data

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


def daily_risk_free(risk_free_annual: pd.Series | float, index: pd.Index) -> pd.Series | float:
    """Per-day risk-free return aligned to ``index``; a dated yield series is forward-filled."""
    if isinstance(risk_free_annual, pd.Series):
        if risk_free_annual.empty:
            return 0.0
        rates = risk_free_annual.sort_index()
        aligned = rates.reindex(rates.index.union(index)).ffill().reindex(index).fillna(rates.iloc[0])
        return aligned / TRADING_DAYS
    return risk_free_annual / TRADING_DAYS


def sharpe_ratio(returns: pd.Series, risk_free_annual: pd.Series | float = 0.0) -> float:
    """Annualised Sharpe ratio. Pass the dated T-bill series so each day is measured
    against the rate that applied on that day, not today's rate."""
    excess = returns - daily_risk_free(risk_free_annual, returns.index)
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
    """Unconditional historical VaR from overlapping ``horizon_days`` compounded returns
    (no sqrt-time scaling). Every day in the window counts equally, whatever the
    volatility regime today."""
    if horizon_days > 1:
        horizon = (1 + returns).rolling(horizon_days).apply(np.prod, raw=True).dropna() - 1
    else:
        horizon = returns
    cutoff = np.quantile(horizon, 1 - confidence)
    tail = horizon[horizon <= cutoff]
    return VaRResult(float(-cutoff), float(-tail.mean()))


def ewma_variance(returns: np.ndarray, lam: float = EWMA_LAMBDA) -> np.ndarray:
    """EWMA variance per asset. Row ``t`` is the variance forecast for day ``t`` made
    at the close of day ``t-1``; the extra last row is the forecast for tomorrow."""
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim == 1:
        r = r[:, None]
    var = np.empty((r.shape[0] + 1, r.shape[1]))
    var[0] = r.var(axis=0, ddof=1)
    for t in range(r.shape[0]):
        var[t + 1] = lam * var[t] + (1 - lam) * r[t] ** 2
    return var


@dataclass
class SimulationResult:
    var: float
    cvar: float
    prob_loss_over_10pct: float
    percentiles: dict[str, float]
    n_paths: int
    horizon_days: int
    current_vol_annual: float  # portfolio EWMA volatility forecast, annualised
    average_vol_annual: float  # portfolio volatility over the whole window, annualised
    method: str
    backend: str
    runtime_ms: float
    samples: np.ndarray | None = None  # horizon returns, kept for charts


def filtered_historical_simulation(
    returns: np.ndarray,
    weights: np.ndarray,
    *,
    horizon_days: int = 10,
    n_paths: int = 100_000,
    confidence: float = 0.95,
    lam: float = EWMA_LAMBDA,
    seed: int = 7,
    xp=np,
    backend: str = "CPU (NumPy)",
    keep_samples: bool = False,
) -> SimulationResult:
    """Filtered historical simulation (Barone-Adesi et al., 1999) with EWMA volatility.

    Each asset's daily returns are divided by their EWMA volatility to get
    standardised residuals. Simulated days re-sample whole residual *vectors*
    (keeping cross-asset dependence and fat tails), rescale them by the current
    volatility forecast, and update each asset's volatility along the path, so
    volatility clustering carries into the horizon. Unlike an iid bootstrap of raw
    returns - which is just historical VaR resampled - the result is conditional on
    today's volatility regime, a genuinely different view.

    ``xp`` is the array module: NumPy on CPU or CuPy on an NVIDIA GPU; the code is
    identical. The filtering pass is cheap and always runs on the CPU.
    """
    started = time.perf_counter()
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim == 1:
        r = r[:, None]
    w = np.asarray(weights, dtype=np.float64)
    var = ewma_variance(r, lam)
    resid = r / np.sqrt(var[:-1])
    resid = resid / resid.std(axis=0, ddof=1)  # unit variance: removes the EWMA scale bias

    z_hist = xp.asarray(resid, dtype=xp.float32)
    weights_x = xp.asarray(w, dtype=xp.float32)
    variance = xp.tile(xp.asarray(var[-1], dtype=xp.float32), (n_paths, 1))
    rng = xp.random.default_rng(seed)
    log_growth = xp.zeros(n_paths, dtype=xp.float32)
    for _ in range(horizon_days):
        day = xp.sqrt(variance) * z_hist[rng.integers(0, z_hist.shape[0], size=n_paths)]
        log_growth += xp.log1p(xp.clip(day @ weights_x, -0.99, None))
        variance = lam * variance + (1 - lam) * day * day
    horizon_returns = xp.expm1(log_growth)

    cutoff = xp.quantile(horizon_returns, 1 - confidence)
    tail = horizon_returns[horizon_returns <= cutoff]
    portfolio = r @ w
    result = SimulationResult(
        var=float(-cutoff),
        cvar=float(-tail.mean()),
        prob_loss_over_10pct=float((horizon_returns < -0.10).mean()),
        percentiles={f"p{p}": float(xp.quantile(horizon_returns, p / 100)) for p in (1, 5, 50, 95, 99)},
        n_paths=n_paths,
        horizon_days=horizon_days,
        current_vol_annual=float(np.sqrt(ewma_variance(portfolio, lam)[-1, 0] * TRADING_DAYS)),
        average_vol_annual=float(portfolio.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        method=f"filtered historical simulation (EWMA lambda={lam})",
        backend=backend,
        runtime_ms=0.0,
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

"""Tools the specialist agents can call.

Design rules that make small models dependable:

* every tool does its own maths and returns finished numbers, already in the
  units an analyst would quote (percent, billions of dollars);
* outputs are compact - a 4B model on a CPU pays for every token it reads;
* arguments are forgiving (``"nvda"``, ``"$NVDA"`` and ``"NVDA, AMD"`` all work),
  because small models are inconsistent about formatting.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import Settings


@dataclass
class ToolContext:
    settings: Settings
    cache: dict = field(default_factory=dict)  # per-run memo, e.g. built filing indexes


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., dict]

    def spec(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def __call__(self, ctx: ToolContext, args: dict[str, Any]) -> dict:
        allowed = set(self.parameters.get("properties", {}))
        clean = {k: v for k, v in args.items() if k in allowed and v not in (None, "")}
        return self.fn(ctx, **clean)


def rnd(x: float | None, digits: int = 2) -> float | None:
    """Round for display; NaN/inf become None so JSON stays valid."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(xf) or math.isinf(xf) else round(xf, digits)


def pct(x: float | None, digits: int = 2) -> float | None:
    return rnd(None if x is None else float(x) * 100, digits)


def bn(x: float | None, digits: int = 2) -> float | None:
    return rnd(None if x is None else float(x) / 1e9, digits)


def parse_tickers(value: Any) -> list[str]:
    if isinstance(value, str):
        items = re.split(r"[,\s;/]+", value)
    elif isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        items = [str(value)]
    out = []
    for item in items:
        t = item.strip().upper().lstrip("$")
        t = re.sub(r"\.(US|O|N)$", "", t)
        if t and t not in out:
            out.append(t)
    return out


def parse_weights(value: Any, n: int) -> list[float]:
    if value in (None, "", []):
        return [1.0 / n] * n
    if isinstance(value, str):
        value = [v for v in re.split(r"[,\s;]+", value) if v]
    weights = [float(v) for v in value]
    if len(weights) != n or sum(weights) <= 0:
        return [1.0 / n] * n
    total = sum(weights)
    return [w / total for w in weights]


TICKER = {"type": "string", "description": "Stock ticker, e.g. NVDA"}
TICKERS = {"type": "array", "items": {"type": "string"}, "description": "One or more tickers, e.g. [\"NVDA\"]"}

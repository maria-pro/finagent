"""A safe arithmetic tool, so derived numbers (ratios, differences) are also
computed - and therefore traceable - rather than produced by the model."""

from __future__ import annotations

import ast
import math
import operator
import re

from . import Tool, ToolContext, rnd

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt, "log": math.log}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")


# "5%" is a percentage unless an operand follows the sign, as in "10 % 3" (modulo).
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d.(])")
_OF_RE = re.compile(r"\bof\b", re.IGNORECASE)


def normalize(expression: str) -> str:
    """Turn analyst shorthand into Python arithmetic: "$1,200 * 5%" -> "1200 * (5/100)",
    "50% of 200" -> "(50/100) * 200"; "%" between two operands stays modulo."""
    cleaned = expression.replace(",", "").replace("$", "").replace("^", "**")
    cleaned = _PERCENT_RE.sub(r"(\1/100)", cleaned)
    return _OF_RE.sub("*", cleaned)


def evaluate(expression: str) -> float:
    return float(_eval(ast.parse(normalize(expression), mode="eval")))


def calculator(ctx: ToolContext, expression: str) -> dict:
    return {"expression": expression, "result": rnd(evaluate(expression), 4)}


TOOLS = [
    Tool(
        "calculator",
        "Evaluate an arithmetic expression, e.g. '(130.5 - 60.9) / 60.9 * 100' or '15% of 240'. Use it for any derived number.",
        {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        calculator,
    ),
]

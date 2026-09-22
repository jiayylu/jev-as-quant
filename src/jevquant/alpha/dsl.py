"""A small, safe expression language for alphas.

    -pct(close, 5)                                  short-term reversal
    rank(ts_mean(volume, 5) / ts_mean(volume, 63))  volume surge
    ts_z(laya_tone, 63) * -pct(close, 5)            news tone vs price move

Expressions are parsed with `ast` and only whitelisted names, operators, functions and window
lengths are accepted, so an alpha proposed by Claude or a random search can never run
arbitrary code. Complexity is capped (nodes, depth) because every extra degree of freedom is
another way to fit noise.

Inputs are date x ticker DataFrames; time-series operators run down each column, cross-
sectional operators across each row (members only when a member mask is supplied).
"""
from __future__ import annotations

import ast
from typing import Callable

import numpy as np
import pandas as pd

WINDOWS = (1, 2, 3, 5, 10, 21, 42, 63, 126, 252)
MAX_NODES = 18
MAX_DEPTH = 8


def _roll(x: pd.DataFrame, n: int, how: str) -> pd.DataFrame:
    r = x.rolling(n, min_periods=max(2, int(n * 0.7)))
    return getattr(r, how)()


def _ts_rank(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=max(2, int(n * 0.7))).rank(pct=True)


def _decay(x: pd.DataFrame, n: int) -> pd.DataFrame:
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    return x.rolling(n, min_periods=n).apply(lambda a: float(np.dot(a, w)), raw=True)


def _ts_corr(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=max(3, int(n * 0.7))).corr(y)


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


TS_FUNCS: dict[str, Callable] = {
    "ts_mean": lambda x, n: _roll(x, n, "mean"),
    "ts_std": lambda x, n: _roll(x, n, "std"),
    "ts_sum": lambda x, n: _roll(x, n, "sum"),
    "ts_min": lambda x, n: _roll(x, n, "min"),
    "ts_max": lambda x, n: _roll(x, n, "max"),
    "ts_skew": lambda x, n: _roll(x, n, "skew"),
    "ts_rank": _ts_rank,
    "ts_z": lambda x, n: (x - _roll(x, n, "mean")) / _roll(x, n, "std"),
    "delay": lambda x, n: x.shift(n),
    "delta": lambda x, n: x - x.shift(n),
    "pct": lambda x, n: x / x.shift(n) - 1,
    "decay": _decay,
}
TS2_FUNCS = {"ts_corr": _ts_corr}
EW_FUNCS = {"abs": np.abs, "sign": np.sign, "log": _slog, "neg": lambda x: -x}
BIN_FUNCS = {"max": np.maximum, "min": np.minimum}


class AlphaError(ValueError):
    pass


def parse(expr: str) -> ast.Expression:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise AlphaError(f"syntax: {e.msg}") from e
    return tree


def complexity(expr: str) -> tuple[int, int]:
    """(nodes, depth) counting operations and inputs only: function names, window constants and
    AST context markers are not part of an alpha's complexity."""
    tree = parse(expr)

    def walk(n) -> tuple[int, int]:
        if isinstance(n, ast.Call):
            kids = [a for a in n.args if not isinstance(a, ast.Constant)]
        elif isinstance(n, ast.BinOp):
            kids = [n.left, n.right]
        elif isinstance(n, ast.UnaryOp):
            kids = [n.operand]
        elif isinstance(n, ast.Name):
            return 1, 1
        else:  # constants
            return 0, 0
        sub = [walk(k) for k in kids]
        return 1 + sum(c for c, _ in sub), 1 + max((d for _, d in sub), default=0)

    return walk(tree.body)


def validate(expr: str, inputs: set[str]) -> None:
    """Raise AlphaError unless `expr` only uses whitelisted syntax, names, functions, windows."""
    tree = parse(expr)
    func_names = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Expression, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)):
            continue
        if id(n) in func_names:
            continue
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            continue
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            continue
        if isinstance(n, ast.Name):
            if n.id not in inputs:
                raise AlphaError(f"unknown input {n.id!r}")
            continue
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            continue
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise AlphaError("only plain function calls")
            f = n.func.id
            if f in TS_FUNCS:
                if len(n.args) != 2 or not isinstance(n.args[1], ast.Constant) or n.args[1].value not in WINDOWS:
                    raise AlphaError(f"{f}(x, n) needs n in {WINDOWS}")
            elif f in TS2_FUNCS:
                if len(n.args) != 3 or not isinstance(n.args[2], ast.Constant) or n.args[2].value not in WINDOWS:
                    raise AlphaError(f"{f}(x, y, n) needs n in {WINDOWS}")
            elif f in ("rank", "zscore", "demean") or f in EW_FUNCS:
                if len(n.args) != 1:
                    raise AlphaError(f"{f}(x) takes one argument")
            elif f in BIN_FUNCS:
                if len(n.args) != 2:
                    raise AlphaError(f"{f}(x, y) takes two arguments")
            elif f == "clip":
                if len(n.args) != 3 or not all(isinstance(a, ast.Constant) for a in n.args[1:]):
                    raise AlphaError("clip(x, lo, hi) needs constant bounds")
            else:
                raise AlphaError(f"unknown function {f!r}")
            if n.keywords:
                raise AlphaError("no keyword arguments")
            continue
        raise AlphaError(f"disallowed syntax: {type(n).__name__}")
    nodes, depth = complexity(expr)
    if nodes > MAX_NODES or depth > MAX_DEPTH:
        raise AlphaError(f"too complex ({nodes} nodes, depth {depth}; max {MAX_NODES}, {MAX_DEPTH})")


class Evaluator:
    """Evaluates expressions on a dict of panels, memoizing shared sub-expressions."""

    def __init__(self, inputs: dict[str, pd.DataFrame], member: pd.DataFrame | None = None):
        self.inputs = inputs
        self.member = member
        self._memo: dict[str, pd.DataFrame] = {}

    def __call__(self, expr: str) -> pd.DataFrame:
        validate(expr, set(self.inputs))
        self._memo = {}  # sub-expressions are shared within one alpha only (panels are ~17 MB each)
        out = self._eval(parse(expr).body)
        self._memo = {}
        if not isinstance(out, pd.DataFrame):
            raise AlphaError("expression must depend on at least one input")
        return out.replace([np.inf, -np.inf], np.nan)

    def _cs(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.where(self.member) if self.member is not None else x

    def _eval(self, n):
        key = ast.dump(n)
        if key in self._memo:
            return self._memo[key]
        if isinstance(n, ast.Name):
            v = self.inputs[n.id]
        elif isinstance(n, ast.Constant):
            v = float(n.value)
        elif isinstance(n, ast.UnaryOp):
            v = -self._eval(n.operand) if isinstance(n.op, ast.USub) else self._eval(n.operand)
        elif isinstance(n, ast.BinOp):
            a, b = self._eval(n.left), self._eval(n.right)
            if isinstance(n.op, ast.Add):
                v = a + b
            elif isinstance(n.op, ast.Sub):
                v = a - b
            elif isinstance(n.op, ast.Mult):
                v = a * b
            else:
                v = a / (b.where(b != 0) if isinstance(b, pd.DataFrame) else (b if b != 0 else np.nan))
        else:  # Call
            f = n.func.id
            args = n.args
            if f in TS_FUNCS:
                v = TS_FUNCS[f](self._eval(args[0]), int(args[1].value))
            elif f in TS2_FUNCS:
                v = TS2_FUNCS[f](self._eval(args[0]), self._eval(args[1]), int(args[2].value))
            elif f == "rank":
                v = self._cs(self._eval(args[0])).rank(axis=1, pct=True) - 0.5
            elif f == "zscore":
                x = self._cs(self._eval(args[0]))
                v = x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)
            elif f == "demean":
                x = self._cs(self._eval(args[0]))
                v = x.sub(x.mean(axis=1), axis=0)
            elif f in EW_FUNCS:
                v = EW_FUNCS[f](self._eval(args[0]))
            elif f in BIN_FUNCS:
                v = BIN_FUNCS[f](self._eval(args[0]), self._eval(args[1]))
            elif f == "clip":
                v = self._eval(args[0]).clip(float(args[1].value), float(args[2].value))
            else:  # pragma: no cover - validate() rejects this
                raise AlphaError(f)
        self._memo[key] = v
        return v

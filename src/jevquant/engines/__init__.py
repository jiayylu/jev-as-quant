"""Decision engines: everything that can answer a typed question set about a state."""
from .base import Engine
from .cache import CachedEngine
from .cascade import CascadeEngine

__all__ = ["Engine", "CachedEngine", "CascadeEngine", "load_engine"]


def load_engine(spec: str, **kw) -> Engine:
    """Build an engine from a short spec: laya[:checkpoint], jev, claude-api, claude-code."""
    kind, _, arg = spec.partition(":")
    if kind == "laya":
        from .laya import LayaEngine
        return LayaEngine(checkpoint=arg or "typed-decisions", **kw)
    if kind == "jev":
        from .jev import JevEngine
        return JevEngine(model=arg or "jev-1.13.0", **kw)
    if kind == "claude-api":
        from .claude import ClaudeAPIEngine
        return ClaudeAPIEngine(model=arg or "claude-opus-5", **kw)
    if kind == "claude-code":
        from .claude import ClaudeCodeEngine
        return ClaudeCodeEngine(model=arg or "claude-opus-5", **kw)
    raise ValueError(f"unknown engine spec {spec!r}")

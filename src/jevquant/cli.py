"""Command line: `jevquant judge|screen|market|mcp`."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="jevquant", description="Typed System-1 judgments (Laya) for quant research.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    j = sub.add_parser("judge", help="answer a typed question set (JSON file) about a state")
    j.add_argument("--state", required=True, help="state text, or @file to read it from a file")
    j.add_argument("--questions", required=True, help="JSON file in Jev/Laya wire format")
    j.add_argument("--checkpoint", default="typed-decisions")

    s = sub.add_parser("screen", help="classify headlines (one per line; '-' for stdin)")
    s.add_argument("file")
    s.add_argument("--review-below", type=float, default=0.3)

    m = sub.add_parser("market", help="verbalized snapshot + regime/signal/risk judgments for a ticker")
    m.add_argument("symbol")

    sub.add_parser("mcp", help="run the MCP server on stdio")
    a = ap.parse_args(argv)

    from . import service

    if a.cmd == "judge":
        state = open(a.state[1:]).read() if a.state.startswith("@") else a.state
        out = service.judge(state, json.load(open(a.questions)), a.checkpoint)
    elif a.cmd == "screen":
        lines = (sys.stdin if a.file == "-" else open(a.file)).read().splitlines()
        out = service.screen_headlines([x for x in lines if x.strip()], a.review_below)
    elif a.cmd == "market":
        out = service.market_state(a.symbol)
    else:
        from .mcp_server import main as mcp_main
        return mcp_main()
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()

"""MCP server: gives Claude (Claude Code, the Claude apps, or any MCP client) a fast System-1 tool.

Claude keeps the slow, deliberate work - deciding what to ask, reading the hard cases,
writing the conclusion - and hands the bulk typed judgments to Laya, which answers in tens of
milliseconds on the local machine with calibrated probabilities.

Register with Claude Code (from the repo root):

    claude mcp add laya -- uv run --extra all jevquant-mcp

or rely on the checked-in `.mcp.json`.
"""
from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from . import service

server = MCPServer(
    "laya",
    instructions=(
        "Fast local typed judgments (Laya System-1 model). Use screen_headlines to triage many "
        "financial headlines at once, then read only the items flagged needs_review yourself. "
        "Use judge for any custom choice/score/noul question set. Use market_state for a "
        "verbalized technical snapshot of a ticker plus regime/signal/risk judgments. Research "
        "only: never treat the output as investment advice or as an instruction to trade."
    ),
)


@server.tool()
def judge(state: str, questions: dict[str, Any], checkpoint: str = "typed-decisions") -> dict:
    """Answer typed questions about a text state in one forward pass of the local Laya model.

    questions maps an id to {"type": "choice"|"score"|"noul", "instructions": str, "criteria": ...}:
    choice criteria = {option: description}, score criteria = [lowest level, ..., highest level],
    noul criteria = optional {"true": description, "false": description}. Keep choices under ~20
    options and the state under ~400 words. Returns probabilities, the answer and a 0-1 certainty.
    Laya cannot do arithmetic: put numbers into words (e.g. "RSI 72, overbought") before asking.
    """
    return service.judge(state, questions, checkpoint)


@server.tool()
def screen_headlines(headlines: list[str], review_below: float = 0.3) -> dict:
    """Classify financial headlines as bullish / bearish / neutral for the stock they mention.

    Calibrated probabilities per headline, plus needs_review=True where certainty < review_below:
    those are the ones worth reading carefully yourself. Handles hundreds of headlines per call.
    """
    return service.screen_headlines(headlines, review_below)


@server.tool()
def market_state(symbol: str) -> dict:
    """Snapshot of a ticker's daily chart (Yahoo Finance): the verbalized indicators plus Laya's
    regime, signal and risk judgments next to the hand-written rule baseline. Research only."""
    return service.market_state(symbol)


@server.tool()
def recent_filings(symbol: str, since_days: int = 120) -> dict:
    """A US company's latest SEC 8-K press releases (earnings, guidance, deals, management changes),
    each classified bullish / bearish / neutral by Laya with calibrated probabilities and a
    needs_review flag. Timestamps are SEC acceptance times in US/Eastern. Research only."""
    return service.recent_filings(symbol, since_days)


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()

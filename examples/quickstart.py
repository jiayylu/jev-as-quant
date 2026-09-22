"""Five-minute tour: one typed question set, three engines, one cascade.

    uv run --extra laya --extra claude python examples/quickstart.py
"""
from jevquant import Choice, Noul, Score
from jevquant.engines import CascadeEngine, load_engine

state = ("Price is 6.1% above its 50-day average. The 50-day average is rising (+1.2%). "
         "RSI(14) is 74, overbought zone. Volatility is normal: 18% annualized, 1.1x its one-year norm. "
         "Trading volume is heavy (1.9x its 20-day average).")
questions = {
    "action": Choice("A disciplined trend-following trader reads this chart. What should they do?",
                     {"buy": "trend and momentum point up", "hold": "mixed evidence", "sell": "trend points down"}),
    "trend": Score("Direction and strength of the price trend",
                   ["strong downtrend", "mild downtrend", "no clear trend", "mild uptrend", "strong uptrend"]),
    "stretched": Noul("The move looks stretched and a pullback is likely."),
}

laya = load_engine("laya:typed-decisions")          # local, open weights
d = laya.system_one(state, questions)
print(f"Laya  {d.latency_ms:6.0f} ms  action={d['action'].choice} {d['action'].probabilities}")
print(f"      trend={d['trend'].score:.2f}/4  stretched={d['stretched'].noul:.2f}  certainty={d.min_certainty:.2f}")

# Hand the uncertain cases to Claude with the same questions (needs ANTHROPIC_API_KEY;
# use load_engine("claude-code") instead to go through a logged-in Claude Code CLI).
try:
    claude = load_engine("claude-api")
    cascade = CascadeEngine(fast=laya, slow=claude, threshold=0.3)
    d2 = cascade.system_one(state, questions)
    print(f"Cascade escalated={d2.escalated} engine={d2.engine} action={d2['action'].choice}")
except Exception as e:  # no credentials: the Laya part above still ran
    print("Claude step skipped:", type(e).__name__, str(e)[:120])

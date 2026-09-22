---
name: laya-triage
description: Triage many financial headlines or tickers fast with the local Laya System-1 model (MCP server "laya"), then reason carefully only about the items it is unsure of. Use when asked to scan, screen, sort or classify a batch of market news or a list of tickers.
---

# Laya triage: System 1 screens, you (System 2) decide the hard cases

1. Send the whole batch to `screen_headlines` (or `market_state` per ticker). Do not read the
   items one by one first; Laya answers hundreds of headlines in seconds.
2. Accept answers with `needs_review: false` as they are. Their probabilities are calibrated on
   labeled financial tweets (see reports/e3_news.json in the jev-as-quant repo).
3. For each `needs_review: true` item, read the text yourself and decide. Say what Laya thought
   and why you agree or disagree.
4. Report the result as a table: item, label, probability, and who decided (Laya or you).
5. Laya cannot do arithmetic. For `judge`, turn numbers into words first ("RSI 72, overbought").

This is research tooling. Never present the output as investment advice or as an instruction to
buy or sell, and never place orders.

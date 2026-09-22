# 演示：Claude Code 通过 MCP 调用 Laya

这是 2026-09-22 在本机实际跑出来的结果，展示"Laya 作为 Claude 的一部分"的工作方式：Claude 把整批新闻一次交给本地 Laya，
Laya 返回校准后的概率和 `needs_review` 标记，Claude 只对被标记的（以及它自己觉得可疑的）条目做判断。

## 命令

```bash
claude -p --model claude-opus-5 --effort low \
  --strict-mcp-config --mcp-config mcp.json \
  --tools "" --allowedTools "mcp__laya__screen_headlines" \
  "Use the laya screen_headlines tool once on all of these headlines. Then give me a table ... (6 条新闻，见 examples/headlines.txt)"
```

`mcp.json` 和仓库里的 `.mcp.json` 一样，只是换成了本地虚拟环境里的 Python。

## Claude 的回答（原样）

I ran Laya once on all 6 headlines. It flagged 1 of them for review.

| # | Headline | Laya label (p) | needs_review | My label & reason |
|---|---|---|---|---|
| 1 | $AAPL – Apple beats estimates and raises FY revenue outlook | bullish (0.78) | no | — |
| 2 | $SAN: Deutsche Bank cuts to Hold | neutral (0.67) | **yes** | **bearish**: this is a downgrade. "Cut to Hold" means the rating was lowered from Buy, which usually hurts the stock. |
| 3 | Tesla to hold annual shareholder meeting on June 12 | neutral (0.90) | no | — |
| 4 | Regulators open probe into bank's AML controls | neutral (0.76) | no | — |
| 5 | Oil climbs as OPEC+ signals deeper output cuts | bullish (0.67) | no | — |
| 6 | Company says it does not expect the recall to affect guidance | neutral (0.95) | no | — |

Two headlines weren't flagged, but I'd still question them:
- **#4:** Laya labelled it neutral (0.76), but I'd call it **bearish**. A regulatory probe into anti-money-laundering controls is a real risk for the bank (fines, remediation costs, reputational damage). It wasn't flagged only because its certainty (0.39) is above the 0.3 cutoff.
- **#5:** "Bullish" fits oil producers but doesn't name a specific stock, so the label depends on which stock you mean.

Laya's other labels look reasonable. This is for research only and isn't trading advice.

## 过程

- 对话轮数：2（一次工具调用，一次回答）
- 总耗时约 51 秒，其中约 30 秒是第一次加载 Laya 权重；之后 Laya 判断 6 条新闻不到 1 秒
- Claude 的费用按标价折算约 $0.076

## 看点

1. **分工符合预期**：Laya 负责批量初筛；"cuts to Hold"是分析师下调评级，Laya 判成中性，但确定度低，于是被标记出来，由 Claude 改判为利空。
2. **Claude 也会复查没被标记的条目**：第 4 条（监管调查）Laya 判中性、确定度 0.39，刚好高于 0.3 的复核线，Claude 仍然指出它更可能是利空。这说明复核线可以按业务风险调高。
3. **边界清楚**：Laya 不下单，Claude 也不给交易建议，输出只用于研究。

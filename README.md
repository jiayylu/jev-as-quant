# jev-as-quant

**把 System-1 决策模型（Laya / Jev）当作量化系统里的"判断层"，并和 Claude（System 2）组合使用。**
从需求分析、任务判断、架构设计、代码实现到模拟实验的完整项目，结论好的坏的都报告。

> *English abstract.* Jev (TypeSafe AI) and its open-weight counterpart Laya (Convai Innovations) return
> typed, calibrated decisions (`Choice` / `Score` / `Noul`) instead of text. This repo wires them into a
> look-ahead-free quant research stack (features → verbalizer → typed judges → code-owned policy and
> reduce-only risk → next-open execution), adds Claude as a System-2 fallback (cascade + MCP tool), and
> tests the "naturally fit for trading" claim with five experiments on a laptop. Short version: Laya is a
> good **reader of financial text**, especially when its least-certain ~20% is escalated to Claude; it is
> a poor **reader of numbers** and adds nothing over plain rules on technical indicators.

> ⚠️ 仅供研究和教育用途。没有券商接口，不会下单，任何输出都不构成投资建议。

---

## 结论速览

| 原来的说法 | 实测 | 判断 |
|---|---|---|
| 毫秒级决策 | 本机 M3 Pro：新闻单问题 **34 ms**；一段行情描述每个问题 **55 ms**，而且随问题数**线性**增长（6 个问题 309 ms）。CPU 上慢 3 倍 | ✅ 日频、分钟级够用；❌ 高频不行（E1） |
| 能直接看技术指标做买卖 | 给原始数字时，Laya 连"RSI 是否大于 70"都答不准（平衡准确率 **0.52**，和瞎猜差不多）；数字转成文字后到 0.76，但一行 `if` 是 1.00 | ❌ 数字判断必须交给代码（E2） |
| 读新闻、判断利好利空 | 零样本 Laya macro-F1 **0.71**（校准后），好于关键词词典（0.62），不如用 9.5k 条标注训练的 TF-IDF（0.76） | ✅ 零样本就能用，**但必须先校准**（ECE 0.22 → 0.02）（E3） |
| 和 Claude 结合 | 只把 Laya 最没把握的 **18%** 交给 Claude：macro-F1 0.71 → **0.78**，超过 TF-IDF；Claude 成本约 **每千条 $0.36** | ✅ 效果最好的组合（E3） |
| 策略路由（市场状态） | 合成市场有真值：规则的平衡准确率 0.50，Laya 0.37，**危机召回率为 0**；按 Laya 的判断路由，收益和规则差不多（差异在噪声范围内） | ⚠️ 没有增量（E4） |
| 新闻驱动交易 | 合成市场里捕获的植入新闻冲击：词典 0.49 · Laya 0.56 · TF-IDF 0.58 · **Laya→Claude 0.68** | ✅ 级联最好（E4） |
| 风控判断 | Laya 的"避险"判断只在 6% 的危机时刻触发，规则是 40% | ❌ 风控用代码（E4） |
| 真实市场的买卖信号 | 8 只 ETF、2014–2026 的回测还在运行，结果会补在后续提交里 | （E5） |
| "零幻觉" | 厂商原话：只保证输出符合 schema，**答案本身可能是错的** | ❌ 类型安全 ≠ 判断正确 |

**一句话**：Laya 擅长**读文字**，不擅长**读数字**。它最适合当 Claude 的"快速预筛层"：大部分样本在本地几十毫秒内判完，只把没把握的那一小部分交给 Claude。

## 文档

| | |
|---|---|
| [01 需求分析](docs/01-需求分析.md) | 目标用户、功能与非功能需求、不做什么、验收标准 |
| [02 任务判断](docs/02-任务判断.md) | 调研事实、逐条核实原始说法、量化任务适配矩阵、实验前写下的假设、和 Claude 的分工 |
| [03 架构设计](docs/03-架构设计.md) | 设计原则、分层、一次决策的时间线、与 Claude 的三种结合方式、模块地图 |
| [04 实验报告](docs/04-实验报告.md) | E1–E5 的完整结果与结论 |
| [reports/RESULTS.md](reports/RESULTS.md) | 自动生成的全部数据表 |

## 架构

```mermaid
flowchart LR
    D[行情 / 新闻] --> F[特征：代码]
    F --> V[转写成文字<br/>只描述，不下结论]
    V --> L{Laya<br/>System 1<br/>约 30–50 ms/问}
    N[新闻文本] --> L
    L -->|确定度够| P[策略：代码<br/>阈值 → 目标仓位]
    L -->|确定度低，约两成| C[Claude<br/>System 2<br/>同一套问题和 schema]
    C --> P
    P --> R[风控：代码<br/>硬约束 → 模型否决只减不增 → 熔断]
    R --> X[次日开盘成交<br/>手续费 + 滑点]
```

设计原则：**模型负责判断，代码负责执行**；**代码算数，模型读字**；**模型的风控意见只能减仓**；**所有引擎共用同一套 Choice / Score / Noul 格式**（Laya、Jev、Claude、规则都能直接替换）。详见 [03 架构设计](docs/03-架构设计.md)。

## 实验结果

<picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/e3_news-dark.png"><img alt="E3" src="reports/figures/e3_news.png"></picture>

**E3：读新闻。** 零样本 Laya 好于词典、不如有监督的 TF-IDF；校准把概率修正准了；把最没把握的样本交给 Claude，效果就超过 TF-IDF。整个过程只用了 1,500 条标注做校准，TF-IDF 用了 9,543 条。

<picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/e4_synthetic-dark.png"><img alt="E4" src="reports/figures/e4_synthetic.png"></picture>

**E4：真值已知的合成市场。** 只看技术指标时，Laya 识别市场状态不如规则；新闻文本来自真实推文时，级联捕获的新闻 alpha 最多。

<picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/e2_numeracy-dark.png"><img alt="E2" src="reports/figures/e2_numeracy.png"></picture>

**E2：Laya 读不了数字。** 同样的市场状态，给原始 JSON 时判断接近瞎猜；写成带形容词的句子才好一些，但仍然比不上一行 `if`。

<picture><source media="(prefers-color-scheme: dark)" srcset="reports/figures/e1_latency-dark.png"><img alt="E1" src="reports/figures/e1_latency.png"></picture>

**E1：延迟。** 短文本和官方 T4 数据一致（约 33 ms），但每多问一个问题就多一份开销。

**E5：真实 ETF。** <!--E5-SUMMARY-->回测还在运行，结果会补在后续提交里。

## 和 Claude 怎么结合

| 方式 | 做什么 | 代码 |
|---|---|---|
| ① 级联 | Laya 判断全部样本，把确定度低的批量交给 Claude，用同一套问题和 JSON schema | `engines/cascade.py`、`engines/claude.py` |
| ② Laya 作为 Claude 的工具 | MCP server 提供 `screen_headlines`、`judge`、`market_state` 三个工具；Claude 一次筛几百条，只精读被标记的几条 | `mcp_server.py`、`.mcp.json`、`.claude/skills/laya-triage` |
| ③ Claude 当老师 | Claude 设计题目、提供标签，用来校准（将来可以蒸馏）Laya | `calibration.py` |

Claude 这边有两种调用方式：官方 `anthropic` SDK（结构化输出、拒答处理、服务端 fallback）；没有 API key 时，可以走已登录的 Claude Code CLI（`claude -p --json-schema`）。本项目的实验用的是后一种。

## 快速开始

```bash
git clone https://github.com/jiayylu/jev-as-quant && cd jev-as-quant
uv venv --python 3.11 && uv pip install -e ".[all]"   # Python >= 3.10
uv run pytest                                          # 离线：合成数据 + 假引擎，约 10 秒
```

第一次用到 Laya 时会从 Hugging Face 下载权重（`convaiinnovations/laya`，每个 checkpoint 约 1.7 GB）。

```bash
# 类型化问题 → Laya（问题文件就是 Jev/Laya 的原生 JSON 格式）
uv run jevquant judge --state "Shares plunge 12% after the company slashes guidance" --questions examples/questions.json
# 批量筛新闻：输出校准后的概率和 needs_review 标记
uv run jevquant screen examples/headlines.txt
# 某个标的的技术面快照：Laya 和规则基线的判断并排给出（仅供研究）
uv run jevquant market SPY
# 复现全部实验（Laya 调用有缓存，重跑不会重复计算）
cd experiments && for e in e1_latency e2_numeracy e3_news e4_synthetic e5_real; do uv run python $e.py; done && uv run python make_report.py
```

### 在 Claude Code 里使用（Laya 成为 Claude 的一个工具）

```bash
claude mcp add laya -- uv run --directory "$(pwd)" --extra all jevquant-mcp
```

仓库里已经带了 `.mcp.json` 和 `.claude/skills/laya-triage`，在仓库目录里打开 Claude Code 会自动识别。然后可以直接说"帮我筛一下这 200 条新闻"：Claude 先调用 `screen_headlines` 一次拿到全部结果，再只精读 `needs_review` 的那几条。

### 在代码里使用

```python
from jevquant import Choice, Noul, Score
from jevquant.engines import CascadeEngine, load_engine

laya = load_engine("laya:typed-decisions")       # 本地、开源权重
jev = load_engine("jev")                         # TypeSafe 托管 API（需要 TYPESAFE_API_KEY）
claude = load_engine("claude-api")               # 官方 anthropic SDK；没有 key 可以用 "claude-code"
engine = CascadeEngine(fast=laya, slow=claude, threshold=0.3)

d = engine.system_one("Apple beats estimates and raises guidance", {
    "sentiment": Choice("How will this move the stock?", {"bullish": "up", "bearish": "down", "neutral": "no effect"}),
    "surprise": Score("How surprising is it?", ["expected", "somewhat", "very"]),
    "guidance": Noul("The headline is about guidance."),
})
print(d["sentiment"].choice, d["sentiment"].probabilities, d.escalated, d.latency_ms)
```

## 目录结构

```
src/jevquant/
  typed.py            Choice / Score / Noul 及其答案类型（与 Jev/Laya 的 wire format 一致）
  engines/            laya.py  jev.py  claude.py  cascade.py  cache.py
  judges.py           4 个 Judge（信号 / 风控 / 路由 / 新闻）+ Reader
  verbalize.py        特征 → 文字（只描述，不下结论）
  features.py  strategies.py  risk.py  backtest.py  calibration.py  metrics.py
  service.py  mcp_server.py  cli.py
data_cache/           运行时下载的行情和推文（不提交）
experiments/          e1 … e5 实验脚本 + make_report.py
reports/              实验输出的 JSON、图表、RESULTS.md
docs/                 01 需求分析 · 02 任务判断 · 03 架构设计 · 04 实验报告
tests/                离线单元测试 + 可选的真实 Laya 一致性测试
```

## 局限与下一步

- **合成市场是我设计的**：状态参数、新闻冲击的大小和时序都是假设。它的作用是"真值已知的测试台"，不能代表真实市场的收益。
- **真实市场只测了技术指标**：没有带时间戳的历史新闻源，所以 Laya 最擅长的"读新闻"在真实回测里没有测到。接入新闻源后，`NewsStrategy` 可以直接用。
- **Jev 没有实测**：目前需要排队申请。客户端是按公开文档写的，只和本地模拟服务器对过协议。
- **只做了校准，没有微调**：官方的微调笔记本需要 2×T4 跑 4～5 小时。用 Claude 的标签蒸馏 Laya 是最值得做的下一步。
- **对抗文本**：新闻是外部写的文字，可能被刻意操纵。风控层保证模型突破不了限额，但判断本身仍然可能被带偏。

## 参考资料

- Laya：[官网](https://laya.convaiinnovations.com/) · [GitHub](https://github.com/NandhaKishorM/laya) · [Hugging Face](https://huggingface.co/convaiinnovations/laya-typed-decisions) · [PyPI `laya`](https://pypi.org/project/laya/)
- Jev：[MarkTechPost 发布报道](https://www.marktechpost.com/2026/09/19/typesafe-ai-releases-jev/) · [实践指南（dev.to）](https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e) · [Cloudflare 模型文档](https://developers.cloudflare.com/ai/models/typesafe/jev/)
- 社区：[Jev 金融与交易项目汇总（gist）](https://gist.github.com/drillan/6916b16e8ea31a8ec36c8f59d6483150) · [buberlo/jev-trader](https://github.com/buberlo/jev-trader)
- 数据：[zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)（MIT）· Yahoo Finance（通过 `yfinance`，运行时下载，不再分发）

## 许可

Apache-2.0。Laya 权重同为 Apache-2.0；数据集按各自的许可在运行时下载。

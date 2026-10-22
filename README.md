# FinAgent

**Multi-agent equity research on a small NVIDIA Nemotron model — no GPU required.**

FinAgent is a team of AI agents that researches public companies (as financial analyst):
- a planner assigns work,
- four specialists (market, fundamentals, risk, filings) call real data tools in parallel,
- a fact-checker traces every number they write back to the tool output it came from and checks that each claim keeps its source's company, period and metric, and
- a lead analyst writes the note, which gets the same check.

It is built to make one claim testable: **a 4-billion-parameter model on a CPU can do useful, auditable agentic work when the system around it is designed for
small models.** A pilot study on 12 companies ([Evaluation](#evaluation)) measures how well that holds and what each design choice contributes. It is evidence
for this setup, not a general result. The same code moves to NVIDIA NIM or Amazon Bedrock endpoints by changing configuration.

**Who it is for.** Self-learners who want to see how an agentic system is built end to end — tools, orchestration, verification, evaluation — and educators
who want a lab that runs on students' laptops. Everything works on a CPU, and [`workshop/LAB.md`](workshop/LAB.md) turns the project into a two-hour workshop.

## Results

| Machine | Inference server | Question | End to end | Model calls | Tool calls (chosen by the model) | Numbers traced | Claims matching source |
|---|---|---|---|---|---|---|---|
| Apple M1 Pro laptop, CPU only (`num_gpu 0`) | Ollama 0.17 | NVDA: is the valuation justified, what are the risks? | **135 s** | 11 | 8 (8) | 42 / 42 | 27 / 27 |
| Apple M1 Pro laptop, CPU only (`num_gpu 0`) | Ollama 0.17 | NVDA vs AMD: growth, valuation, downside risk | **154 s** | 12 | 16 (10) | 51 / 51 | 29 / 29 |
| Intel Xeon E5-2690 v4 (2016), 8-vCPU VM, 4 threads, shared with other jobs | Ollama 0.40 | NVDA: is the valuation justified, what are the risks? | **446 s** | 10 | 8 (8) | 52 / 52 | 33 / 33 |

Model: [NVIDIA Nemotron 3 Nano 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF), Q4_K_M, 2.8 GB. The notes and full run records (every
prompt's token count, every tool call, its result and the data sources behind it) are in [`examples/`](examples/); the web UI replays them without a model.
Notes from version 0.1, written before claims were checked, are kept in [`examples/v0.1/`](examples/v0.1/) for the lab's fact-checking exercise.
Letting Ollama use the laptop's Apple GPU (Metal) cut a run only from 105 s to 87 s in testing: at 4B parameters the CPU is already enough.
The risk engine's filtered historical simulation runs a million 10-day paths for a five-asset portfolio in about 0.3 s on the
same laptop CPU (`finagent bench-risk`).

What a run looks like (laptop CPU; each line is a model call, a tool call with its evidence ID, or a fact-check):

```
19:47:24  planner      → fundamentals · market · risk · filings  (model)
19:47:32  fundamentals   model   7.6s  662→65 tok  → financial_statements, valuation_snapshot
19:47:32  fundamentals   F1  financial_statements(ticker=NVDA, years=3) 0.15s
19:47:32  fundamentals   F2  valuation_snapshot(ticker=NVDA) 0.19s
19:47:48  risk           model   7.7s  674→83 tok  → value_at_risk, stress_test
19:47:48  risk           R1  value_at_risk(tickers=['NVDA'], confidence=0.95, horizon_days=10) 0.06s
19:47:48  risk           R2  stress_test(tickers=['NVDA']) 0.01s
...
19:48:54  fact-check   fundamentals: 9/9 numbers traced · 3/5 claims match their sources
                       NVIDIA revenue $130.5bn (growth 114.2%) -> $130.5bn: fiscal year missing (source: FY ending 2025, not the latest) ...
19:48:54  fact-check   risk: 11/11 numbers traced · 5/5 claims match their sources
19:48:54  fundamentals ↺ revising (2 problem(s))
19:49:11  fundamentals   revision  17.4s  1131→239 tok  → text
19:49:11  fact-check   fundamentals after revision: 11/11 numbers traced · 5/5 claims match their sources
19:49:30  writer         model  18.5s  954→329 tok  → text
19:49:30  fact-check   writer: 15/15 numbers traced · 8/8 claims match their sources
```

![Agent timeline in the web UI](docs/ui-timeline.jpg)

## Why

- **Access.** Most agentic-AI teaching assumes a GPU or a paid API. FinAgent runs on a student laptop or a shared university CPU server with
  [NVIDIA Nemotron 3 Nano 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF) served by Ollama or llama.cpp.
- **Small models are enough for most agent steps.** NVIDIA Research argues that small language models are sufficient and far more economical for the narrow,
  repetitive sub-tasks that make up agentic systems ([Belcak et al., 2025](https://arxiv.org/abs/2506.02153)). FinAgent is a concrete
  case study with the telemetry to check the claim.
- **Finance needs auditability.** A research note is only useful if its numbers are right and say what they claim to say. Every figure here is traceable to a
  tool call, a public source and, for fundamentals, the XBRL concept and the SEC filing it came from.

## How it works

```mermaid
flowchart LR
    Q([question + tickers]) --> G[Guardrail<br/>rules · NemoGuard]
    G -->|allowed| P[Planner<br/>JSON-schema output]
    G -->|refused| X([refusal])
    P -->|Send| M[Market analyst]
    P -->|Send| F[Fundamentals analyst]
    P -->|Send| R[Risk analyst]
    P -->|Send| D[Filings analyst]
    M & F & R & D --> V{Fact-check<br/>numbers + claims}
    V -->|problems| RV[Targeted revision]
    V -->|clean| W[Lead analyst]
    RV --> W
    W --> WC{Fact-check}
    WC -->|problems: revise once| W
    WC --> N([research note + evidence ledger])

    subgraph Tools [deterministic tools]
      T1[price_performance<br/>benchmark_comparison]
      T2[financial_statements<br/>valuation_snapshot]
      T3[value_at_risk<br/>stress_test]
      T4[search_10k<br/>BM25 / NeMo Retriever]
    end
    M -.-> T1
    F -.-> T2
    R -.-> T3
    D -.-> T4
```

The workflow is a [LangGraph](https://langchain-ai.github.io/langgraph/) state machine (`finagent/graph.py`). Specialists fan out with LangGraph's `Send`
API; progress streams to the CLI and web UI as it happens.

| Agent | Tools | Data |
|---|---|---|
| Market analyst | `price_performance`, `benchmark_comparison` | daily prices; SPY benchmark; Sharpe ratios against the daily 13-week T-bill rate (US Treasury) |
| Fundamentals analyst | `financial_statements`, `valuation_snapshot`, `calculator` | SEC EDGAR XBRL company facts (10-K/10-Q), with the concept and filing behind each figure |
| Risk analyst | `value_at_risk`, `stress_test` | historical VaR and filtered historical simulation (EWMA-scaled, 100,000 paths) with current vs average volatility; 2018, 2020 and 2022 crash windows |
| Filings analyst | `search_10k` | latest 10-K from EDGAR, split into Items 1, 1A, 7 and 7A, retrieved with BM25 (plus NeMo Retriever embeddings on NIM) |

Two risk views are deliberate. Historical VaR weighs every day of the last five years equally; filtered historical simulation rescales that history to
today's volatility. When markets are calmer than average the second is lower, when they are stormier it is higher — and the note says which regime applies.

![Risk charts in the web UI: the filtered historical simulation's distribution of 10-day returns, with its VaR and the historical VaR](docs/ui-charts.jpg)

## Design principles for small-model agents

1. **Narrow agents.** Each specialist has two or three tools and a short prompt. Narrow scope is what makes a 4B model's tool calls reliable.
2. **Tools compute, models narrate.** The model never does arithmetic. Tools return finished, pre-labelled facts in the units an analyst quotes
   (`NVDA FY ending 2026-01-25: revenue $215.94bn, up 65.47% year on year`); derived figures go through a `calculator` tool so they are traceable too.
3. **Constrain structure, don't request it.** The planner's output is bound to a JSON schema, which Ollama and llama.cpp compile into a decoding grammar.
4. **Verify deterministically.** Instead of a large LLM-as-judge, two rule-based checks run on everything a model writes. `provenance.py` traces every
   number to a tool result the agent actually received, allowing for rounding and unit changes (0.2282 → 22.82%, 215,938,000,000 → $215.94B).
   `attribution.py` checks that each cited claim *says what its source says*: the same company, period (1y/3y/5y, n-day horizon, fiscal year, trailing
   twelve months, growth-rate span), method and metric, a direction ("fell", "rose") that agrees with the source's sign, and — for 10-K points — the
   passage's wording. Agents with problems get one revision with the exact problems in the prompt; the lead analyst's summary gets the same; anything
   left is flagged ⚠ in the note.
5. **No evidence, no findings.** If a model skips a tool call the answer needs, the system runs it and records that it did. If every tool call of an agent
   fails, its finding says what failed — the model is never asked to write without data.
6. **Report honestly.** Every run records which tool calls the models chose and which the system added, every model call's latency and tokens, the
   provenance and attribution scores, and for every piece of evidence the URLs it came from, when they were retrieved, and whether a provider outage meant a
   stale cached copy.
7. **One protocol, any hardware.** Ollama, llama.cpp, self-hosted NIM, the NVIDIA API catalog and Amazon Bedrock all speak the OpenAI-compatible API, so
   moving from a laptop CPU to NVIDIA GPUs is configuration, not code. The risk simulation takes its array module as a parameter: NumPy on CPU, CuPy on an
   NVIDIA GPU.
8. **Measure what each choice is worth.** Every mechanism above can be switched off (`FINAGENT_ABLATE=facts,phases,coverage,revision,own_evidence,attribution`),
   and `finagent eval` scores full and ablated runs with the same yardstick.

The web UI's *Fact-check* tab shows each check and, for every agent that was sent back, its first draft next to its revision. Here the fundamentals agent
listed older fiscal years' revenue without their year; the revision copies the labelled facts instead:

![Fact-check tab in the web UI: first draft and revision side by side](docs/ui-factcheck.jpg)

## Quickstart

### 1. Local CPU with Ollama (no GPU, no API key)

```bash
ollama pull nemotron-3-nano:4b                 # 2.8 GB

git clone https://github.com/maria-pro/finagent && cd finagent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ui,dev]"
cp .env.example .env                           # set SEC_USER_AGENT to your name + email

finagent doctor                                # checks endpoint, tool calling, JSON output, data sources
finagent analyze NVDA -q "Is NVIDIA's valuation justified by its fundamentals, and what are the main risks?"
streamlit run app/streamlit_app.py             # web UI: live agent trace, charts, evidence ledger, fact-check
```

Runs are saved to `reports/` as a Markdown note plus a JSON run record. The web UI can replay any saved run — and the examples shipped in `examples/` —
without calling a model, which helps when a live run would take minutes.

### 2. llama.cpp instead of Ollama

```bash
scripts/serve_llamacpp.sh                      # CPU only (-ngl 0), port 8191
FINAGENT_BASE_URL=http://127.0.0.1:8191/v1 finagent doctor
```

### 3. NVIDIA API catalog (NIM)

```bash
export NVIDIA_API_KEY=nvapi-...                # free key from build.nvidia.com
finagent doctor --profile nim
finagent analyze NVDA AMD -q "Compare growth and downside risk" --profile nim
```

The `nim` profile routes specialists to `nvidia/nemotron-3.5-lightning-30b-a3b`, the planner and writer to `nvidia/nemotron-3-super-120b-a12b`, adds NeMo
Retriever embeddings (`nvidia/nemotron-3-embed-1b`) to the 10-K search, enables the NemoGuard topic-control rail and runs the specialists in parallel.
Point `FINAGENT_BASE_URL` at a self-hosted NIM container to keep everything on-premises.

### 4. Amazon Bedrock

```bash
export BEDROCK_API_KEY=...                     # Bedrock long-term API key
export AWS_REGION=us-east-1
finagent doctor --profile bedrock
```

Bedrock serves Nemotron through an OpenAI-compatible endpoint, so this is configuration only: specialists on `nvidia.nemotron-nano-3-30b`, orchestration
on `nvidia.nemotron-super-3-120b`. The `local` profile also runs unchanged on any CPU cloud instance with Ollama.

`finagent doctor` is the first thing to run on any endpoint: it reports how reasoning is switched off, sends a real tool-calling request and a
JSON-schema request, checks embeddings and the topic rail when configured, and lists any request parameters the server rejects (runs retry without them).

### Shared servers behind a proxy

If the Ollama service on a shared machine cannot reach the internet and you cannot change it, `scripts/ollama_import_gguf.sh` downloads NVIDIA's official
GGUF through your own shell and registers it with Ollama, using the built-in Nemotron renderer and tool-call parser and a thread cap for polite sharing.

## What small models get wrong, and how the system copes

Building with a 4B model surfaces failure modes that larger models mostly hide. Each of these was seen in real runs; each is now handled by the system
rather than by hoping the model behaves, and each is covered by a regression test or, for the writer's answer, by an evaluation metric.

| Failure seen in real runs | How FinAgent handles it |
|---|---|
| With the answer format in the system prompt, agents skipped their tools and wrote a fluent answer with invented numbers and invented evidence IDs. | **Phase-separated prompting:** the system prompt only says "you have no data until you call your tools"; the format arrives after the first tool results. An answer written with no data is discarded and the agent is nudged once. |
| An agent copied the example bullet from its prompt (a real-looking figure) into its findings. | Prompts use placeholders, never sample numbers. Each agent is checked only against **its own** evidence, so borrowing another agent's figure is caught. |
| Correct numbers with wrong labels: an older year's growth quoted as the latest, a 25-day VaR called "1-year". | Tools return **pre-labelled facts**, latest first. The model sees only these, which also cuts its prompt tokens. |
| Even when copying labelled facts, agents and the writer dropped the label: an older fiscal year's growth written as plain "revenue growth", five-year returns without "5-year", expected shortfall called VaR. A number check passes all of these. | **Claim attribution** compares each claim with the fact it cites and sends the agent — or the writer — back with the exact mismatch. |
| When a data source failed (ExxonMobil's ticker now points to a new holding company with no 10-K yet), the filings agent invented figures and cited the failed call. | **No evidence, no findings:** an agent whose tool calls all failed reports what failed; claims citing a failed call are flagged. |
| The summary skipped part of the question — no valuation figures in an answer to "is the valuation justified?". | The writer must address every part of the question with a cited finding; the evaluation's facet-coverage score measures it. |
| Agents stopped after one tool call, or merged two tickers into one portfolio when asked to compare them. | Agents are told which tools they have not used yet; a **coverage check** runs any required call the model still skipped, and the trace records that the system, not the model, made it. |
| The fact-checker flagged "S&P 500" as an untraced figure, and an early version accepted a fabricated number within 0.6% of a real one. | Index and form names are masked; matching allows display rounding only, not "close enough". |

## Evaluation

`finagent eval` runs the agent team over many companies, questions and ablations, and scores every note with the same yardstick, whatever was switched on
while it was written:

- **Numbers traced** (`number_provenance`) — share of numbers in the note that trace to a tool result the writing agent received.
- **Claims matching source** (`draft_claim_attribution`, `claim_attribution`) — share of cited claims that keep their source's company, period,
  metric and direction (10-K points: by wording); shown for the first drafts and for the final note, so the effect of the revision pass is visible.
- **Fully clean notes** — notes with every number traced and every claim matching.
- **Facet coverage** (`facet_coverage`) — whether the lead analyst's summary addresses each part of the question (valuation, fundamentals, risk,
  performance, filings, comparison) by citing evidence of the right kind: a valuation figure for "is the valuation justified?", a risk measure or a
  10-K risk passage for "what are the risks?", every company named in the answer for a comparison. A transparent proxy for "does it answer the
  question", not a judgement of the reasoning.
- **Autonomy and cost** — tool calls chosen by the model rather than added by the coverage check (`tool_calls_by_model` of `tool_calls`), nudges,
  revisions, tokens and time.

### Pilot study

**Setup.** Nemotron 3 Nano 4B (Q4_K_M) through Ollama, CPU only, on an Apple M1 Pro laptop; temperature 0. Twelve companies chosen to stress the data
layer as much as the model: six large technology companies (NVDA, AMD, AAPL, MSFT, GOOGL with its multiple share classes, META), two banks (JPM, BAC) and a
broker-dealer (GS) whose statements lack a gross margin, Berkshire Hathaway (BRK-B, share counts in custom units), ExxonMobil (XOM, mid-way through a
holding-company reorganisation) and Coca-Cola (KO). Two question types: *"Is X's valuation justified by its fundamentals, and what are the main risks?"*
and *"How risky is X for a long-term investor, and what does its 10-K flag?"*.

**Full system.** All 24 runs, produced with version 0.2.2 and scored with the 0.2.3 checks:

| Question | Runs | Median time | Numbers traced | Claims matching source (draft → final) | Fully clean notes | Facet coverage | Tool calls chosen by the model | Revisions / run | Tokens / run |
|---|---|---|---|---|---|---|---|---|---|
| Valuation and risks | 12 | 132 s | 100% | 98% → 99% | 9 / 12 | 89% | 98% | 0.7 | 11,400 |
| Risk and 10-K flags | 12 | 120 s | 100% | 96% → 97% | 7 / 12 | 88% | 87% | 0.8 | 10,500 |
| **All runs** | 24 | 126 s | 100% | 97% → 98% | 16 / 24 | 88% | 93% | 0.7 | 11,000 |

- **Every number traced, and nearly every claim kept its label.** All numbers in all 24 notes trace to a tool result. 97% of
  cited claims matched their sources in the first drafts and 98% after the revision pass; 16 of the 24 notes came out fully
  clean. What remains is marked ⚠ in the note: a period dropped in the summary ("a high beta of 2.31" for a 3-year beta),
  older fiscal years listed without their year, figures in an interpretive sentence without a citation, and 10-K points
  paraphrased too loosely to match their passage. One filings agent quoted figures from a different 10-K passage than the
  one it cited, and misread a two-year table; the claim check caught it, where the number check alone would not have.
- **The summaries addressed 53 of the 60 question parts with cited evidence (88%).** Three misses are data gaps that the
  notes report rather than hide: Berkshire Hathaway has no usable share count in XBRL, so the valuation tool declines
  rather than guesses, and ExxonMobil's new holding company has no annual statements or 10-K yet. The other four are
  summaries that skipped part of the question: two valuation answers argued from growth without quoting a valuation
  multiple (NVDA, AAPL), and two risk answers cited no 10-K point (JPM, BAC). That is the next thing to fix.
- **The model ran its own tools.** It chose 155 of the 167 tool calls; the coverage check added the rest, mostly a stress
  test the risk agent skipped on the risk question. No agent tried to answer without data, so none needed a nudge.
- **About two minutes per note on a laptop CPU:** median 126 s (100–159 s), 11,000 tokens on average.
- **Re-scoring found two false alarms in the checker.** In the AMD and GOOGL valuation notes, two correct sentences were
  flagged: one for a benchmark named in another clause, one for reading "32.81% net" as unlabelled. Version 0.2.3 fixes
  both. Re-scoring the saved runs (`finagent eval --rescore`) removed exactly those two flags and changed nothing else.

**Ablations.** Four of the design choices switched off one at a time on six of the companies (AAPL, GOOGL, JPM, KO, NVDA, XOM; valuation question,
version 0.2.0), compared with the full system on the same six:

| Switched off | Runs | Median time | Numbers traced | Claims matching source (draft → final) | Fully clean notes | Facet coverage | Revisions / run | Tokens / run |
|---|---|---|---|---|---|---|---|---|
| nothing (full system) | 6 | 130 s | 99% | 94% → 97% | 3 / 6 | 89% | 1.2 | 11,900 |
| labelled facts | 6 | 164 s | 98% | 70% → 84% | 2 / 6 | 83% | 2.8 | 14,900 |
| phase-separated prompting | 6 | 163 s | 99% | 92% → 96% | 3 / 6 | 94% | 1.3 | 14,900 |
| coverage check | 6 | 128 s | 99% | 94% → 97% | 3 / 6 | 89% | 1.2 | 11,900 |
| claim attribution | 6 | 111 s | 99% | 92% → 92% | 0 / 6 | 89% | 0.2 | 10,100 |

- **Labelled facts carry the most weight.** Without them, first drafts matched their sources 70% of the time instead of
  94%. Revisions more than doubled and still left more mismatches (84% against 97%), and runs took a quarter longer.
- **Claim attribution turns good drafts into clean notes.** Drafts were just as good without it, but nothing caught their
  mismatches: no note came out fully clean, against half of them with the check. It costs about 20 seconds per run.
- **Phase separation and the coverage check are safety nets.** The failure phase separation guards against — agents
  writing an invented answer instead of calling their tools — did not occur in these runs (it did with the same model
  under llama.cpp during development), so no agent needed a nudge either way. Without it, runs still used a quarter more
  tokens and took half a minute longer, and the coverage check had to add three skipped tool calls instead of one. The
  coverage check itself fired once in the full runs (a skipped call for XOM) and did not change the scores.
- **Number tracing barely moved (98–99%).** When it has the data, a small model rarely invents numbers; it mislabels them.
  That is why claims, not numbers, are the measure to watch.

These runs were scored with version 0.2.0 of the claim checker, which also flagged some correct sentences that combine
several facts (fixed in 0.2.2). Every variant was scored the same way, so the comparison holds; the absolute claim scores
are slightly conservative.

**What the pilot does not show.** It is one model at one setting (temperature 0, so one sample per question), on 12 companies — enough to see each
mechanism working and to find bugs, not to estimate small differences precisely. The two tables come from different versions (0.2.2 and 0.2.0), so
compare rows within a table, not across them. Facet coverage is a proxy for answering the question; it does not judge whether the reasoning is sound.
The attribution rules were calibrated on this project's own runs, so they may miss phrasings they do not anticipate and can still raise the odd false
alarm. And there is no larger-model baseline yet: the same harness runs one with `--profile nim` or `--profile bedrock`.

**Reproduce.** The results, with every run's full record, are in [`eval/results/`](eval/results/). Market data changes daily, so new runs will
not match these numbers exactly; the saved runs can be re-scored exactly, without a model.

```bash
finagent eval --set pilot --questions valuation,risk --variants full     # the full-system study
finagent eval --tickers NVDA AAPL JPM GOOGL KO XOM --variants full,-facts,-phases,-coverage,-attribution
finagent eval --rescore eval/results/pilot-0.2.2-runs                     # re-score the saved runs with the current checks
finagent eval --summarize eval/results/pilot-0.2.2.jsonl
finagent eval --set broad --variants full                                 # 40 companies
finagent eval --set pilot --profile nim                                   # the same study on NVIDIA-hosted models
```

## Learning with FinAgent

**If you are learning on your own**, a path through the project:

1. Run `finagent doctor` and one `finagent analyze`, and read the live trace: every line is a model call, a tool call or a fact-check.
2. Open the run in the web UI. The *Fact-check* tab shows what was caught and revised; the *Agent timeline* shows the agents taking turns on one CPU.
3. Read the code in this order: `tools/` (what the agents can do), `agents/specialists.py` (the tool loop), `graph.py` (orchestration), `provenance.py`
   and `attribution.py` (verification), `evaluate.py` (measurement).
4. Switch one design choice off with `FINAGENT_ABLATE` and see what changes in the note and the scores.
5. Work through [`workshop/LAB.md`](workshop/LAB.md).

**If you teach**, [`workshop/LAB.md`](workshop/LAB.md) is a two-hour lab in the style of an NVIDIA Deep Learning Institute workshop: learning objectives,
six modules with exercises, an optional ablation-study module, an assessment rubric and instructor notes. It needs no GPU, and saved runs replay instantly
in the web UI when class time is short. It follows on from the DLI courses
[*Building Agentic AI Applications with LLMs*](https://learn.nvidia.com/courses/course-detail?course_id=course-v1%3ADLI+C-FX-25+V1) (LangGraph,
tool-using ReAct agents) and *Building RAG Agents with LLMs* (retrieval).

## Repository layout

```
finagent/
  config.py          profiles (local / nim / bedrock), ablation switches, FINAGENT_* overrides
  llm.py             OpenAI-compatible client: server probe, reasoning switch, parameter fallback, tool-call recovery
  graph.py           LangGraph workflow and event streaming
  agents/            planner, specialists (tool loop, coverage check, revision), writer, guardrails
  tools/             market, fundamentals, risk, filings, calculator
  data/              cached HTTP with fetch records, SEC EDGAR (XBRL facts, 10-K text), prices, Treasury rates
  analytics.py       returns, drawdowns, beta, dated Sharpe, historical VaR, filtered historical simulation, stress tests
  retrieval.py       chunking, BM25, optional dense embeddings with reciprocal rank fusion
  provenance.py      number extraction and tracing
  attribution.py     claim-to-source checking: company, period, metric, direction, passage wording
  checks.py          the fact-check shared by the workflow, the report and the evaluation
  evaluate.py        evaluation harness: companies x questions x ablations
  report.py          note assembly: key figures, evidence ledger and run statistics come from tools, not models
  cli.py             analyze · doctor · eval · bench-risk · tools
app/streamlit_app.py web UI
tests/               offline test suite (synthetic data + scripted model)
examples/            real run records and notes from the runs in the results table
eval/results/        pilot evaluation results
workshop/LAB.md      two-hour hands-on lab
scripts/             llama.cpp server, offline Ollama import
docs/                screenshots
```

## Testing

```bash
pytest -q
```

The suite (121 tests, 93% line coverage, under ten seconds) runs offline against synthetic prices, synthetic SEC facts and a scripted model. It covers
the finance maths (filtered historical simulation against volatility regimes, dated Sharpe), XBRL parsing (fiscal years, TTM, restatements, tag changes,
share classes, debt composition, net-income attribution), the LLM client through the real OpenAI SDK over a mocked HTTP transport (server probing,
reasoning switches, the parameter-fallback loop, concurrency), the provenance and attribution checkers, data freshness, the guardrails, the CLI, the
evaluation harness including re-scoring of saved runs, and the full workflow — including a planted hallucination and a writer that drops periods, both of
which must be caught and repaired.

## Limitations

- Daily prices come from Yahoo Finance's public chart endpoint, which is unofficial and may change. SEC EDGAR and the US Treasury are official sources.
- XBRL tagging varies by filer. Some insurers tag revenue differently; companies that report shares in custom units (for example Berkshire Hathaway)
  get no valuation rather than a guessed one; a newly registered successor company has no annual history until its first 10-K.
- A 4B model writes plainer prose than a frontier model, and its interpretation is not expert. The checks keep the *numbers and labels* honest; they do not
  make the *judgement* sound.
- The attribution checker is a set of explicit rules. It checks labels and direction, not reasoning, and it can miss phrasings its rules do not
  anticipate — treat its score as a floor on the mislabelling caught, not a proof of correctness.
- The `nim` and `bedrock` profiles are implemented and fall back gracefully when an endpoint rejects an optional parameter, but every measured result here
  is a CPU run. The `nim` model IDs are listed in NVIDIA's public catalog; run `finagent doctor --profile nim` (or `bedrock`) with your key before relying
  on either.
- The note is educational analysis, not investment advice.

## License

MIT for the code. Nemotron models are covered by the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/).

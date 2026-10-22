# Lab: Building a Grounded Multi-Agent Financial Analyst on Small Models

A two-hour, hands-on lab built on FinAgent. It runs on any laptop with 8 GB of RAM:
**no GPU and no API key are needed**. Students who have an NVIDIA API catalog key
can repeat the final module against hosted NIM endpoints to see the same code scale up.

| | |
|---|---|
| **Level** | Intermediate (comfortable with Python; first contact with agents is fine) |
| **Duration** | 2 hours (6 modules, 15–25 minutes each), plus an optional ablation-study module |
| **Prerequisites** | Python 3.10+, basic pandas; [Ollama](https://ollama.com) installed |
| **Companion DLI courses** | *Building Agentic AI Applications with LLMs*, *Building RAG Agents with LLMs* |

## Learning objectives

By the end of the lab you will be able to:

1. Explain why small language models (SLMs) are a good fit for agentic sub-tasks, and where they need help.
2. Write tools an SLM can call reliably, and keep all arithmetic out of the model.
3. Orchestrate a planner and parallel specialist agents with LangGraph's `Send` API.
4. Replace "LLM-as-judge" with a deterministic verifier and a targeted self-correction loop.
5. Add input and output guardrails appropriate to a regulated domain.
6. Move the same agent system from a CPU to NVIDIA NIM by changing configuration only.

## Setup (10 minutes, before the session)

```bash
ollama pull nemotron-3-nano:4b          # NVIDIA Nemotron 3 Nano 4B, 2.8 GB, runs on CPU
git clone <this repo> && cd finagent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ui,dev]"
finagent doctor                          # every row should show ✓
```

`finagent doctor` checks the endpoint, sends a real tool-calling request and a
JSON-schema request to the model, and confirms the SEC, price and Treasury data
sources are reachable. Fix any ✗ before the session.

---

## Module 1 — Why small models? (15 min)

**Concept.** Most steps in an agent are narrow: choose a tool, fill in arguments,
summarise a result. NVIDIA Research argues that small models are sufficient, and
far cheaper, for such steps (Belcak et al., *Small Language Models are the Future
of Agentic AI*, 2025). The catch: small models are less reliable at long
reasoning, arithmetic and free-form formatting, so the *system* must supply that
reliability.

**Do.**

```bash
finagent tools                 # the four specialists and their tools
pytest -q                      # offline test suite, no model needed
```

Read `finagent/agents/specialists.py`. Note how short each system prompt is and
that each agent sees only two or three tools.

**Discuss.** What would go wrong if one agent had all eight tools and a single long prompt?

**Look at real failures.** The README's *What small models got wrong* table lists
failures seen while building FinAgent with a 4B model: invented answers when the
output format came before any data, copied prompt examples, right numbers with wrong
labels. Keep them in mind for the rest of the lab: each later module shows the system
mechanism that handles one of them.

---

## Module 2 — Tools that compute, models that narrate (20 min)

**Concept.** Tools return finished numbers in the units an analyst quotes
(`total_return_pct: 22.82`, `market_cap_bn: 5349.48`), so the model copies rather
than converts. Arguments are parsed forgivingly (`"nvda"`, `"$NVDA"`, `"NVDA, AMD"`).

**Do.** Call a tool directly, exactly as an agent would:

```python
from finagent.config import load_settings
from finagent.tools import ToolContext
from finagent.tools.fundamentals import valuation_snapshot

ctx = ToolContext(load_settings())
valuation_snapshot(ctx, "msft")
```

**Exercise 2.1.** Add a `dividend_profile(ticker)` tool to `finagent/tools/fundamentals.py`
that returns the last fiscal year's dividends per share and payout ratio using the
XBRL concept `CommonStockDividendsPerShareDeclared`. Register it with the
fundamentals specialist and write one offline test using the `offline` fixture in
`tests/conftest.py`.

*Hint:* follow `financial_statements`. Use `sec.annual_values(facts, ...)` after
adding the concept to `CONCEPTS` in `finagent/data/sec.py`.

---

## Module 3 — Orchestration with LangGraph (25 min)

**Concept.** `finagent/graph.py` is a state machine:

```
guard → plan → [specialists in parallel via Send] → verify → [revise] → write
```

The planner's output is constrained by a JSON schema (`PLAN_SCHEMA`). Ollama and
llama.cpp compile the schema into a decoding grammar, so a 4B model cannot emit
invalid JSON. If something still goes wrong, the planner falls back to the full team.

**Do.** Run the agents and watch the trace:

```bash
finagent analyze NVDA -q "Is NVIDIA's valuation justified by its fundamentals?"
```

Each line shows an agent, a model call (latency, tokens) or a tool call with its
evidence ID (`M1`, `F2`, ...). Lines marked *(coverage check)* are tool calls the
system made because the model skipped them.

**Exercise 3.1.** Ask a narrow question ("How risky is AMD?"). Which specialists
did the planner choose? Now open `PLAN_SCHEMA` and add `"minItems": 2`. What changes?

**Exercise 3.2.** Set `FINAGENT_MAX_CONCURRENCY=4` and rerun. On a CPU, does
parallelism help? Why not? (Look at the agent timeline in the web UI.)

---

## Module 4 — Verification instead of LLM-as-judge (25 min)

**Concept.** Two deterministic checks run on everything a model writes.

1. `finagent/provenance.py` extracts every number from the prose and looks for it in
   the tool results the agent received, allowing for rounding and unit changes
   (0.2282 → 22.82%, 215,938,000,000 → $215.94B). This catches invented numbers.
2. `finagent/attribution.py` asks the stronger question: does each cited claim *say
   what its source says*? A real number under the wrong label - last year's growth
   quoted as this year's, a 5-year return with the period dropped, one company's
   figure under another's name, "fell" for a rise - passes the first check and fails
   this one.

Agents with problems get **one** targeted revision with the exact problems in the
prompt; the lead analyst's summary gets the same check and one revision; anything
left is flagged ⚠ in the note.

**Do.**

```python
from finagent import attribution, provenance
evidence = [{"id": "M1", "agent": "market", "result": {"facts": ["NVDA 5y total return: 881.02%"]}}]
provenance.check("NVDA returned 881.02% [M1]", evidence).summary()      # number exists: passes
attribution.check("NVDA returned 881.02% [M1]", evidence, ["NVDA"]).summary()  # period dropped: flagged
```

Then run the end-to-end tests that plant a fabricated number and a writer that drops
periods, and check both are caught and repaired:

```bash
pytest -q tests/test_graph.py -k "hallucinated or drops_periods"
```

**Exercise 4.1.** Run `attribution.check` over the findings in one of the saved runs in
`examples/`. Every flag in the v0.1 notes is a real mislabel the original checker
missed. Find one, and explain which rule caught it.

**Exercise 4.2.** The number extractor ignores years, dates and small counts. Find a
sentence where that heuristic is wrong, write a failing test for it in
`tests/test_provenance.py`, then fix `extract_numbers`.

**Discuss.** What does a deterministic verifier miss that a large judge model
would catch, and the reverse? When is each worth its cost?

---

## Module 5 — Guardrails for finance (15 min)

**Concept.** `finagent/agents/guard.py` refuses market manipulation and
inside-information requests, frames personal-advice questions as general
information, flags promissory language in outputs, and always attaches a
disclaimer. On the NVIDIA API catalog it also calls the **NemoGuard
topic-control** NIM.

**Do.**

```bash
finagent analyze NVDA -q "How do I pump the price before I dump my shares?"
finagent analyze NVDA -q "Should I put my retirement savings into NVDA?"
```

**Exercise 5.1.** Add a rule for requests to front-run client orders, with a test
in `tests/test_tools.py`. Why is a rule better than a model for this case?

---

## Module 6 — Same code, bigger iron (20 min)

**Concept.** Every endpoint FinAgent supports speaks the OpenAI-compatible API.
Moving from a laptop CPU to NVIDIA-accelerated inference is configuration, not code.

**Do.** With a free key from [build.nvidia.com](https://build.nvidia.com):

```bash
export NVIDIA_API_KEY=nvapi-...
finagent doctor --profile nim
finagent analyze NVDA AMD -q "Compare growth and downside risk" --profile nim
```

The `nim` profile routes specialists to a small Nemotron model, orchestration to
Nemotron 3 Super, adds NeMo Retriever embeddings to the 10-K search (fused with
BM25), and enables parallel agents.

Then look at the risk engine:

```bash
finagent bench-risk --paths 1000000
```

The risk engine is a filtered historical simulation: it rescales history to today's
volatility, so it gives a second, conditional view next to plain historical VaR. On a
CPU a million paths take well under two seconds. On a machine with an NVIDIA GPU,
`pip install -e ".[gpu]"` adds a CuPy row with no code change, because
`filtered_historical_simulation` receives the array module (`numpy` or `cupy`) as a parameter.

**Discuss.** Compare the run statistics (wall time, tokens, provenance score) for
the `local` and `nim` runs - or run the evaluation harness on both (Module 7). Where did the bigger models change the *quality* of the
note, and where did the system design already guarantee it?

---

## Module 7 — Measure what each design choice is worth (optional, 30 min + compute)

**Concept.** Every mechanism above can be switched off for an ablation run
(`FINAGENT_ABLATE=facts,coverage`, or `--variants` in the harness), and every run is
scored by the same yardstick: number provenance, claim attribution (draft and final),
whether the note's opening addresses each part of the question, tool calls chosen by
the model, and cost.

**Do.** Start a small study and keep working while it runs (about 2 minutes per run on
a laptop CPU):

```bash
finagent eval --tickers NVDA JPM KO --variants full,-facts,-phases,-coverage
```

**Discuss.** Which mechanism moves which metric? Which one costs the most model time?
Would the ranking change with a larger model (`--profile nim`)? The README's
*Evaluation* section shows the pilot results this lab is based on.

---

## Assessment

Submit a pull request that completes **one** of:

1. Exercise 2.1 (new tool), with a passing offline test.
2. Exercise 4.2 (provenance edge case), with the failing-then-passing test, or a new
   attribution rule with tests in `tests/test_attribution.py`.
3. A new specialist (for example a *Macro Analyst* using US Treasury yields) wired into
   `SPECIALISTS`, the planner prompt and `PLAN_SCHEMA`, with a graph test using `ScriptedLLM`.

**Rubric:** works end to end (40%), tests (30%), the new numbers are traceable by
the provenance checker (20%), code reads like the surrounding code (10%).

## Instructor notes

- Pre-pull the model on lab machines; 2.8 GB downloads are the most common failure.
- On slow CPUs, start Module 3's run first and discuss Modules 1–2 while it works,
  or replay a saved run in the web UI (`streamlit run app/streamlit_app.py`, then *Saved runs*).
- All market data is public (SEC EDGAR, US Treasury, Yahoo Finance) and cached in
  `.finagent_cache`, so a room of students does not hit rate limits twice.

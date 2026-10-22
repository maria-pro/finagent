# Changelog

## 0.3.1 — 2026-10-23

Security, for anyone serving the web UI to other people.

- The web UI listens on this computer only (`.streamlit/config.toml`); it has no login.
- An API key held by the server is no longer pre-filled in the web UI, so it never reaches
  the browser, and it is used only with the endpoint the server is configured for: a visitor
  who changes the endpoint must bring their own key.
- `FINAGENT_UI_SHARED=1` for shared deployments: fixed endpoints and models, and each
  visitor sees only the shipped examples and their own runs, kept in their browser session
  instead of the shared `reports/` folder.
- `FINAGENT_TOPIC_RAIL_REQUIRED=1` makes the topic-control rail fail closed.
- The filings tool withholds 10-K sentences that address an AI reader (prompt injection).
  The screen matched nothing in the 10-Ks of the measured companies, so the 0.3.0
  measurements and example runs stand for this version.
- README: sharing the web UI, and limitations on guardrails and untrusted documents;
  SECURITY.md; lab Exercise 5.2 on planted instructions.
- 143 offline tests.

## 0.3.0 — 2026-10-23

The tools, checked against what they measure; exercises on what the checks miss; and the
project described as what it is, a teaching tool.

The fact-checks compare what a model writes with what the tools return, so they cannot
catch a tool that measures the wrong thing. Fixes in the tools:

- Banks, brokers and insurers (SEC industry codes 6000-6499, from EDGAR's submissions
  data) get price-to-book and return on equity instead of enterprise value, EV-to-sales,
  price-to-sales and free cash flow, which do not mean the same for them.
- A missing debt tag gives no enterprise value, not a debt of zero.
- Total revenues win over contract revenue reported for the same year (Berkshire
  Hathaway: $371bn, not $247bn).
- Year-end balances are found when a later 10-Q repeats them.
- Stress tests compound the weighted daily returns, rebalanced daily, instead of
  following a buy-and-hold index started ten years earlier, whose weights had drifted far
  from the stated ones by any crash window.
- Alpha and beta come from returns in excess of the T-bill rate.
- Price levels (52-week high, 200-day average, market cap) use actual closes; returns use
  adjusted closes, and a missing adjusted series is recorded in the evidence.

Lab and documentation:

- Exercises 4.3-4.5: a null baseline for number tracing, negated 10-K claims that pass the
  wording check, and a note that is fully traced but wrong. The weaknesses stay in `main`
  so the exercises work; worked solutions and fixes are on the `solutions` branch.
- README: the measurements are worked examples of how to measure an agentic system, and
  the limitations list what the checks cannot see.
- The 24-run measurement was re-run on 0.3.0 and the example runs refreshed. The 0.2.2
  runs stay in `eval/results/`; one of them is Exercise 4.5.

Packaging:

- CI installs the exact versions in `uv.lock`; upper bounds at the next major version for
  openai, langgraph, numpy and pandas.
- CITATION.cff; package description and authors.
- 138 offline tests.

## 0.2.3 — 2026-10-22

Two false alarms in the claim checker, found in the pilot's 0.2.2 runs, and the pilot
results.

- A benchmark named in another clause no longer counts against a figure: in "a 14.55%
  10-day VaR and a 20% SPY fall implying -46.13%", the VaR is not a claim about SPY.
- "32.81% net" reads as a net margin, so "margins of 32.03% operating and 32.81% net"
  is no longer flagged.
- `finagent eval --rescore` records which version scored each run (`scored_with`) next
  to the version that produced it (`version`).
- Pilot evaluation results in `eval/results/` and the README: 24 full-system runs
  (12 companies, two questions) produced with 0.2.2 and re-scored with 0.2.3, with every
  run's full record, and the ablation study.
- Example runs refreshed on 0.2.3. The 0.1 notes are kept in `examples/v0.1/` for the
  lab's claim-checking exercise.
- 121 offline tests.

## 0.2.2 — 2026-10-22

Claim checking, sharpened on the pilot's risk-question runs, where summaries combine
several facts in one sentence.

- Each number is checked against its own clause of the claim: in "a 14.27% 10-day VaR
  and revenue of $215.94bn", the "10-day" no longer counts against the revenue. Labels
  are compared clause by clause too, which also catches swapped labels ("expected
  shortfall 14.27% and VaR 13.75%").
- In a fact with sub-facts ("3-year beta: 1.15; a 20% SPY fall implies -23%"), the lead
  label stops at the semicolon.
- SEC identifiers (CIK, accession numbers) are not treated as figures; filing metadata
  counts as evidence for 10-K points; the wording threshold for 10-K points is 40%.
- Raw tool results without labelled facts are checked with their nested fields and context.
- `finagent eval` saves every run's full record and `--rescore` recomputes all metrics
  from saved records with the current checks.

## 0.2.1 — 2026-10-22

Fixes found by the first pilot evaluation (`eval/results/`).

- **No evidence, no findings:** when every tool call of a specialist fails, its finding
  states what failed instead of asking the model to write - in the pilot, the filings
  agent invented figures when ExxonMobil's 10-K search failed (the checker caught them).
- A tool that returns "no data" (`{"error": ...}`) is recorded as a failed call in the
  ledger, and attempted calls are not re-run by the coverage check.
- Claims citing a failed tool call are reported as such.
- Writer: the Answer must address every part of the question with a cited finding (the
  pilot's summaries sometimes skipped the valuation or fundamentals figures).
- Broker-dealer net revenue (`RevenuesNetOfInterestExpense`, e.g. Goldman Sachs) is
  recognised; clearer errors for registrants without annual filings (ExxonMobil's new
  holding company, CIK 2115436, has filed only a 10-Q).
- Negative growth is written "down X%" rather than "up -X%".
- Evaluation rows record the code version.

## 0.2.0 — 2026-10-22

Stronger verification, a better risk model, auditable data, and an evaluation harness.

### Verification
- **Claim attribution** (`finagent/attribution.py`): every cited claim is now checked
  against the sources it cites, not just for the existence of its numbers. A claim must
  keep its source's company, period (1y/3y/5y, n-day horizon, fiscal year, trailing
  twelve months, growth-rate span), method and metric labels; direction words ("fell",
  "rose") must agree with the source's sign; 10-K points must share most of their
  wording with the cited passage. Run over the v0.1 example notes, it finds real
  mislabels the number check missed (older fiscal years' growth presented as current,
  five-year returns without the period).
- The lead analyst's summary now gets the same fact-check and one revision pass, and
  is told to copy finding bullets rather than paraphrase them.
- Specialists are revised with the exact problems (claim, issue, source text).
- `checks.note_quality` scores a finished note the same way whatever produced it, so
  ablated and full runs are comparable.

### Evaluation
- `finagent eval`: tickers × questions × ablation variants, scored on number
  provenance, claim attribution (draft and final), facet coverage (does the summary
  address each part of the question with cited evidence), autonomy and cost. Results
  stream to JSON lines; `--summarize` renders a table.
- Every design choice can be switched off (`FINAGENT_ABLATE=facts,phases,coverage,
  revision,own_evidence,attribution`).

### Finance
- Risk: the iid bootstrap "Monte Carlo" VaR, which only resampled the same history as
  the historical VaR, is replaced by **filtered historical simulation** (EWMA
  volatility, λ = 0.94): residuals are bootstrapped and rescaled to today's volatility,
  with volatility updated along each path. Facts label both views and report current
  vs average volatility.
- Sharpe ratio uses the dated daily 13-week T-bill series (US Treasury CSVs per year,
  ^IRX fallback) instead of today's rate for every day.
- Enterprise value: debt is assembled explicitly from the tags each filer uses
  (`LongTermDebtNoncurrent` + `DebtCurrent`, or + `LongTermDebtCurrent` + short-term
  borrowings, or `LongTermDebt` + short-term borrowings) on the same balance-sheet date
  as cash; the method string records which.
- Net income prefers `NetIncomeLoss` (attributable to the parent); `ProfitLoss`
  (includes non-controlling interests) is a last resort and is flagged when used.
- Tool outputs record the XBRL concept and filing accession behind each figure.

### Data integrity
- Cache writes are atomic (temp file + `os.replace`).
- Every evidence record lists the URLs it was computed from, when each was retrieved,
  and whether it was served from a stale cache because a provider was down; the report
  shows retrieval times and flags stale data.

### Robustness
- Ollama is detected by probing `/api/version` instead of matching the port, and the
  reasoning style can be set explicitly (`FINAGENT_REASONING_STYLE`).
- Rejected optional parameters (structured output, reasoning switches) are dropped one
  at a time, even when the server's error does not name them; the run reports which.
- 10-K excerpts are the window of sentences that best matches the query, so the text
  that made a chunk rank is never cut off.
- Retrieval: chunks with no query terms no longer receive BM25 credit in reciprocal
  rank fusion (which could cancel the dense ranking).
- Calculator: "5%" and "50% of 200" are percentages; `%` between operands stays modulo.
- Promissory language is flagged inline in the note, as documented.

### Platforms
- `bedrock` profile: Nemotron on Amazon Bedrock through its OpenAI-compatible endpoint
  (`nvidia.nemotron-nano-3-30b` specialists, `nvidia.nemotron-super-3-120b`
  orchestration). Not yet measured.
- `finagent doctor` reports the reasoning control in use, embedding and topic-rail
  checks when configured, and parameters the endpoint rejected.

### Tests
- 108 offline tests, 93% line coverage. New: the LLM client through the real OpenAI SDK
  over a mocked HTTP transport (probing, reasoning toggles, the fallback loop, tool-call
  parsing, concurrency limit), attribution rules, CLI commands, run records, data
  freshness, filtered historical simulation, dated Sharpe, debt composition.

## 0.1.0 — 2026-10-22

First release.

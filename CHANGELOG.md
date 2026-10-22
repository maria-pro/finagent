# Changelog

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
  provenance, claim attribution (draft and final), facet coverage (does the opening
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
- 108 offline tests (113 in 0.2.1), 93% line coverage. New: the LLM client through the real OpenAI SDK
  over a mocked HTTP transport (probing, reasoning toggles, the fallback loop, tool-call
  parsing, concurrency limit), attribution rules, CLI commands, run records, data
  freshness, filtered historical simulation, dated Sharpe, debt composition.

## 0.1.0 — 2026-10-22

First release.

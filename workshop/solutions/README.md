# Lab solutions

For instructors, and for checking your own work after you have tried the exercises. The code for
these answers lives on this branch (`solutions`); `git diff main solutions` shows every change.
The branch follows `main`, so everything else in the repository is the same.

## Exercise 4.3 — Null baseline

**Script:** [`scripts/null_baseline.py`](../../scripts/null_baseline.py). It writes random numbers into
sentences, checks them against the evidence ledgers of saved runs, and counts how many pass as
"traced" — once with no citation, once citing a random piece of evidence from the same run.

```bash
python scripts/null_baseline.py eval/results/pilot-0.2.2-runs
```

On the pilot's 24 saved runs (median 142 numbers per ledger), with the checker on `main`:

| Fabricated number | No citation | Random citation |
|---|---|---|
| `$123.45bn` | 0.5% | 0.6% |
| `23.47%` | 2.6% | 2.6% |
| `27.31` | 2.4% | 2.3% |
| `23.5%` | 7.3% | 7.1% |
| `27.3` | 6.0% | 6.7% |
| `23%` | **39.0%** | **37.6%** |

The 0.3.0 runs (`eval/results/pilot-0.3.0-runs`) give the same picture: 38.5% for a whole-number percentage.

**Why.** A number counts as traced if *any* value in the run's evidence matches it within display
rounding, in *any* unit: 23% is matched by 22.6%, by 0.2251, and by a P/E of 23.1. A two-decimal
number has a narrow target; a whole-number percentage has a wide one, and a ledger of 140 numbers
covers a lot of the range from 10 to 99. The 100% "numbers traced" in the README is strong evidence
because the agents copy two-decimal figures; any rounding a model does is barely checked.

**The fix on this branch** (`finagent/provenance.py`):

1. *Units must agree* (`_COMPATIBLE`): a percentage traces to a percentage or to a fraction
   (0.2282 → 22.82%), never to a ratio; amounts convert between bn, million, trillion and dollars.
   Tool fields carry their unit in their name (`_pct`, `_bn`).
2. *Rounded numbers count only from the evidence their sentence cites*: a number with fewer than
   two decimals must appear in the evidence cited in its own sentence or bullet.

After the fix:

| Fabricated number | No citation | Random citation |
|---|---|---|
| `$123.45bn` | 0.4% | 0.4% |
| `23.47%` | 1.6% | 1.7% |
| `27.31` | 0.3% | 0.4% |
| `23.5%` | 0.0% | 0.8% |
| `27.3` | 0.0% | 0.2% |
| `23%` | 0.0% | 5.0% |

**Check the fix on real notes, not just on random numbers.** Re-scored with the stricter check, the
24 pilot notes lose 4 of their 944 numbers, and all 4 deserve it: two whole-number
percentages in interpretive sentences with no citation, and two figures a filings agent cited to
the wrong 10-K passage. A stricter rule that flags correct notes would be a worse rule.

## Exercise 4.4 — Break the 10-K check

**Claims that pass on `main`**, against a passage of Apple's 10-K that says the company *is* exposed
to credit risk and that its hedging *may not* be effective:

- "Apple is not exposed to credit risk from its counterparties" — passes, 60% word overlap
- "Apple's hedging activities are fully effective in offsetting foreign exchange risk" — passes, 67%

`not` is a stopword, so word overlap cannot see it.

**The fix on this branch** (`finagent/attribution.py`, `_negations`): find the passage sentence closest
to the claim; a negation governs the first content word after it ("not **exposed**", "may not be
**effective**", "unable to **reduce**"); flag the claim when a word it shares with that sentence is
negated in one and not in the other. Legal boilerplate ("not limited to", "without limitation",
"whether or not") negates nothing. Tests: `test_negated_10k_claims_are_caught`.

**What the first attempt taught.** The obvious rule — flag when exactly one of claim and sentence
contains a negation anywhere — caught both examples and also flagged 6 of the 115 real 10-K claims in
the pilot, at least five of them wrongly: boilerplate, a negation in a clause the claim did not cover ("…or that we
believe are *not material*…"), and a sentence split at a semicolon. The word-level rule flags none of
the 115 and still catches both. Write the test for the case you are fixing, then run the new rule
over real outputs before trusting it.

**Where the rule is approximate.** "The first content word after the negation" is a guess at its scope: "may not be
effective to offset" and "the inability to offset" negate different words by that rule, though they mean the same. On the
pilot's notes the guess holds — no false flags on the 115 real 10-K claims of version 0.2.2 or the 119 of 0.3.0, once
"inability" was added as a negation word after it produced the one false flag on 0.3.0 — but a claim written differently
could trip it.

**What it still misses:** antonyms without a negation word ("fully effective" vs "ineffective"),
quantifiers ("all" vs "some"), and conditions dropped from a claim. A small natural-language-inference
cross-encoder catches more of these and runs on a CPU, at the cost of another model and its own
error rate.

## Exercise 4.5 — Traced but wrong

**The case:** `eval/results/pilot-0.2.2-runs/gs-valuation-full-v0.2.2-*.md`, a Goldman Sachs note with
100% of numbers traced and 96% of claims matching their sources. Two errors pass every check.

1. **The tool was wrong.** Version 0.2.2's `valuation_snapshot` treated a broker-dealer like an
   industrial company. Goldman reports no debt under the tags the tool looked for, so debt became $0:
   enterprise value came out at a quarter of market cap, EV-to-sales 1.06 against a price-to-sales of
   3.89, and the free-cash-flow yield −18.33%, because a broker-dealer's operating cash flow swings
   with its trading assets and customer funds. The note read that as "underlying cash flow weakness".
   The checks compare what the model wrote with what the tool returned; they cannot see a tool that
   is wrong. Version 0.3.0 gives banks, brokers and insurers (by SEC industry code) price-to-book and
   return on equity instead, and treats a missing debt tag as unknown, not zero.
2. **The judgement was wrong.** The Answer says the valuation is "not fully justified … as the stock
   trades at a low P/E of 12.28" — a low P/E argues the other way. Every figure is right and
   correctly labelled; the reasoning is not. A deterministic check cannot see this. The options are
   an expert rating against a rubric, an LLM judge with a rubric (which brings its own error rate),
   or structure: give the writer a benchmark (the stock's own history, its sector) so that "cheap"
   and "expensive" are tool outputs too.

**What "traced" means:** the numbers came from the tools and keep their labels. It does not mean the
tools measured the right thing, or that the conclusion follows. That is what a reader still has to
check, and why the README calls the notes *faithful to their sources*, not *correct*.

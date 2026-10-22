# Measurement results

The data behind the README's [Measuring the system](../../README.md#measuring-the-system) section: worked examples of how to
measure an agentic system, not evidence for claims beyond this setup.

**Setup.** NVIDIA Nemotron 3 Nano 4B (Q4_K_M) served by Ollama 0.17 with `num_gpu 0` (CPU only) on an Apple M1 Pro
laptop, temperature 0, run on 2026-10-22. Market data, SEC filings and Treasury rates as retrieved on that day.

| File | What it holds |
|---|---|
| `pilot-0.3.0.jsonl` | 24 full-system runs on version 0.3.0: 12 companies × 2 questions. One JSON object per run: scores, the claims still flagged, tool calls (chosen by the model or added by the coverage check), nudges, revisions, tokens and time. |
| `pilot-0.3.0.md` | Summary table of those runs (`finagent eval --summarize`). |
| `pilot-0.3.0-runs/` | The full record and the note of every run: each model call's tokens and latency, each tool call with its result and data sources, the first drafts and the fact-check. |
| `pilot-0.2.2.jsonl`, `pilot-0.2.2.md`, `pilot-0.2.2-runs/` | The same 24 questions on version 0.2.2, scored with the 0.2.3 checks (`version`, `scored_with`). Kept as a record, and for the lab: the Goldman Sachs valuation note is Exercise 4.5, a note that passes every check and is wrong anyway. The ⚠ marks in these notes are the ones made while each run was written. |
| `ablation-0.2.0.jsonl` | The ablation study on version 0.2.0, valuation question: the full system on all 12 companies, and four design choices (`-facts`, `-phases`, `-coverage`, `-attribution`), each switched off on six of them. |
| `ablation-0.2.0.md` | Summary table of the ablation study. |

The two questions:

- `valuation`: *"Is X's valuation justified by its fundamentals, and what are the main risks?"*
- `risk`: *"How risky is X for a long-term investor, and what does its 10-K flag?"*

## Working with the results

```bash
finagent eval --summarize eval/results/pilot-0.3.0.jsonl        # summary table
finagent eval --rescore eval/results/pilot-0.3.0-runs           # re-score every saved run with the current checks
```

Re-scoring is how a change to the fact-checker can be tested against real notes without running the model again. Two false
alarms fixed in 0.2.3 were found that way: re-scoring the 0.2.2 runs removed exactly those two flags (in the AMD and GOOGL
valuation notes) and changed nothing else. The lab's solutions for Exercises 4.3 and 4.4 were checked the same way: a stricter
rule that flags correct claims in these notes would be a worse rule. To open a saved run in the web UI, copy its `.json` file
into `reports/`.

Version 0.2.0 did not save full run records, so the ablation study can be summarised but not re-scored. Its claim scores
come from the 0.2.0 checker, which also flagged some correct sentences that combine several facts; every variant was
scored the same way.

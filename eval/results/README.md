# Pilot evaluation results

The data behind the README's [Evaluation](../../README.md#evaluation) section.

**Setup.** NVIDIA Nemotron 3 Nano 4B (Q4_K_M) served by Ollama 0.17 with `num_gpu 0` (CPU only) on an Apple M1 Pro
laptop, temperature 0, run on 2026-10-22. Market data, SEC filings and Treasury rates as retrieved on that day.

| File | What it holds |
|---|---|
| `pilot-0.2.2.jsonl` | 24 full-system runs produced with version 0.2.2 (`version`) and scored with the 0.2.3 checks (`scored_with`): 12 companies × 2 questions. One JSON object per run: scores, the claims still flagged, tool calls (chosen by the model or added by the coverage check), nudges, revisions, tokens and time. |
| `pilot-0.2.2.md` | Summary table of those runs (`finagent eval --summarize`). |
| `pilot-0.2.2-runs/` | The full record and the note of every run: each model call's tokens and latency, each tool call with its result and data sources, the first drafts and the fact-check. The ⚠ marks in the notes are the ones made while the run was written, by the 0.2.2 checks. |
| `ablation-0.2.0.jsonl` | The ablation study on version 0.2.0, valuation question: the full system on all 12 companies, and four design choices (`-facts`, `-phases`, `-coverage`, `-attribution`), each switched off on six of them. |
| `ablation-0.2.0.md` | Summary table of the ablation study. |

The two questions:

- `valuation`: *"Is X's valuation justified by its fundamentals, and what are the main risks?"*
- `risk`: *"How risky is X for a long-term investor, and what does its 10-K flag?"*

## Working with the results

```bash
finagent eval --summarize eval/results/pilot-0.2.2.jsonl        # summary table
finagent eval --rescore eval/results/pilot-0.2.2-runs           # re-score every saved run with the current checks
```

Re-scoring is how a change to the fact-checker can be tested against real notes without running the model again. That is
how the two false alarms fixed in 0.2.3 were found and checked: re-scoring these 24 runs removed exactly those two flags
(in the AMD and GOOGL valuation notes) and changed nothing else. To open a saved run in the web UI, copy its `.json` file
into `reports/`.

Version 0.2.0 did not save full run records, so the ablation study can be summarised but not re-scored. Its claim scores
come from the 0.2.0 checker, which also flagged some correct sentences that combine several facts; every variant was
scored the same way.

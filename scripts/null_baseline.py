"""Exercise 4.3 (solution): how often does a fabricated number pass as "traced"?

A score of "100% of numbers traced" only means something if a made-up number would usually
fail. This script injects random numbers of different precision into sentences checked
against the evidence ledgers of saved runs, and counts how many the number check accepts:
once with no citation, once citing a random piece of evidence from the same run.

    python scripts/null_baseline.py eval/results/pilot-0.3.0-runs
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from finagent import provenance

FORMATS = {
    "$123.45bn": lambda rng: f"${rng.uniform(1, 500):.2f}bn",
    "23.47%": lambda rng: f"{rng.uniform(1, 99.99):.2f}%",
    "27.31": lambda rng: f"{rng.uniform(10, 99.99):.2f}",
    "23.5%": lambda rng: f"{rng.uniform(1, 99.9):.1f}%",
    "27.3": lambda rng: f"{rng.uniform(10, 99.9):.1f}",
    "23%": lambda rng: f"{rng.randint(10, 99)}%",
}


def load_ledgers(runs_dir: str | Path) -> list[list[dict]]:
    paths = sorted(p for p in Path(runs_dir).glob("*.json") if not p.name.startswith("._"))
    return [json.loads(p.read_text(encoding="utf-8"))["evidence"] for p in paths]


def false_pass_rates(ledgers: list[list[dict]], draws: int = 200, seed: int = 7, cite: bool = False) -> dict[str, float]:
    """Share of fabricated numbers, per format, that the number check accepts."""
    rng = random.Random(seed)
    rates = {}
    for name, make in FORMATS.items():
        passed = total = 0
        for evidence in ledgers:
            ids = [e["id"] for e in evidence if not e.get("error")]
            for _ in range(draws):
                citation = f" [{rng.choice(ids)}]" if cite and ids else ""
                report = provenance.check(f"The figure was {make(rng)}{citation}.", evidence)
                passed += report.verified
                total += report.checked
        rates[name] = passed / total if total else 0.0
    return rates


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_dir")
    parser.add_argument("--draws", type=int, default=200, help="fabricated numbers per format and ledger")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    ledgers = load_ledgers(args.runs_dir)
    sizes = [len(provenance.evidence_numbers(e)) for e in ledgers]
    print(f"{len(ledgers)} ledgers, median {statistics.median(sizes):.0f} numbers each\n")
    uncited = false_pass_rates(ledgers, args.draws, args.seed)
    cited = false_pass_rates(ledgers, args.draws, args.seed, cite=True)
    print(f"{'fabricated number':18} {'no citation':>12} {'random citation':>16}")
    for name in FORMATS:
        print(f"{name:18} {uncited[name]:>12.1%} {cited[name]:>16.1%}")


if __name__ == "__main__":
    main()

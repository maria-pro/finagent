"""Save and load run records (report + evidence + trace) shared by the CLI and the web UI.

A saved run can be replayed in the UI without re-running any model - useful for
presenting a run that took several minutes on a CPU.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

RECORD_KEYS = (
    "tickers", "question", "guard", "plan", "plan_source", "findings", "evidence", "llm_calls",
    "narrative", "narrative_draft", "writer_check", "revisions", "report", "stats", "eval",
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def build_record(final: dict, events: list[dict], settings) -> dict:
    """The run record: everything needed to replay and audit a run, without the API key."""
    record = {k: final.get(k) for k in RECORD_KEYS}
    record["events"] = events
    record["settings"] = {
        "profile": settings.profile,
        "base_url": settings.base_url,
        "orchestrator_model": settings.orchestrator_model,
        "specialist_model": settings.specialist_model,
        "temperature": settings.temperature,
        "reasoning": settings.reasoning,
        "ablate": sorted(settings.ablate),
    }
    return record


def save_run(final: dict, events: list[dict], settings, out_dir: str | Path = "reports", stem: str | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tickers = final.get("tickers", [])
    stem = stem or f"{datetime.now():%Y%m%d-%H%M}-{'-'.join(tickers).lower()}-{_slug(final.get('question', ''))}"
    record = build_record(final, events, settings)
    (out / f"{stem}.md").write_text(final.get("report", ""), encoding="utf-8")
    path = out / f"{stem}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def list_runs(dirs: tuple[str, ...] = ("reports", "examples")) -> list[Path]:
    """Your own runs (newest first), then the example runs shipped with the repository."""
    out: list[Path] = []
    for d in dirs:
        # skip macOS "._" metadata files that appear on exFAT/network drives
        out += sorted((p for p in Path(d).glob("*.json") if not p.name.startswith("._")), reverse=True)
    return out


def load_run(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

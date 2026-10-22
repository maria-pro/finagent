"""Runtime settings: which model serves which agent role, and where data is cached.

Two profiles ship by default:

* ``local`` - one small NVIDIA Nemotron model (Nemotron 3 Nano 4B) served on CPU
  by Ollama or llama.cpp. No GPU, no API key, and nothing leaves the machine
  except the public market-data requests.
* ``nim``   - the NVIDIA API catalog (build.nvidia.com): small models for the
  specialist sub-agents, a larger model for orchestration, NeMo Retriever
  embeddings and a NemoGuard topic-control rail.

Every field can be overridden with a ``FINAGENT_*`` environment variable (or a
``.env`` file), so the same code runs against self-hosted NIM, vLLM or any other
OpenAI-compatible endpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROFILES: dict[str, dict] = {
    "local": {
        "base_url": "http://127.0.0.1:11434/v1",  # Ollama; llama-server works too
        "api_key": "not-needed",
        "orchestrator_model": "nemotron-3-nano:4b",
        "specialist_model": "nemotron-3-nano:4b",
        "embed_model": "",
        "guard_model": "",
        # Nemotron Nano: greedy decoding when reasoning is off.
        "temperature": 0.0,
        "top_p": 1.0,
        "reasoning": False,
        # A CPU inference server works best on one request at a time; the LLM
        # client queues the agents' requests.
        "max_concurrency": 1,
        "timeout_s": 900.0,
    },
    "nim": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "",
        "orchestrator_model": "nvidia/nemotron-3-super-120b-a12b",
        "specialist_model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "embed_model": "nvidia/nemotron-3-embed-1b",
        "guard_model": "nvidia/llama-3.1-nemoguard-8b-topic-control",
        # NVIDIA's recommended sampling for Nemotron 3.
        "temperature": 1.0,
        "top_p": 0.95,
        "reasoning": False,
        "max_concurrency": 4,
        "timeout_s": 180.0,
    },
}


@dataclass(frozen=True)
class Settings:
    profile: str
    base_url: str
    api_key: str
    orchestrator_model: str
    specialist_model: str
    embed_model: str
    guard_model: str
    temperature: float
    top_p: float
    reasoning: bool
    max_concurrency: int
    timeout_s: float
    sec_user_agent: str
    cache_dir: Path

    @property
    def is_local(self) -> bool:
        return self.base_url.startswith(("http://127.0.0.1", "http://localhost", "http://0.0.0.0"))

    def model_for(self, role: str) -> str:
        return self.specialist_model if role == "specialist" else self.orchestrator_model

    def with_overrides(self, **overrides) -> "Settings":
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


def _coerce(value: str, current):
    if isinstance(current, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def load_settings(profile: str | None = None) -> Settings:
    name = (profile or os.getenv("FINAGENT_PROFILE") or "local").strip().lower()
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}; choose one of {sorted(PROFILES)}")
    values = dict(PROFILES[name])
    if name == "nim":
        values["api_key"] = os.getenv("NVIDIA_API_KEY", "")

    for key, current in list(values.items()):
        env = os.getenv(f"FINAGENT_{key.upper()}")
        if env is not None and env != "":
            values[key] = _coerce(env, current)

    return Settings(
        profile=name,
        sec_user_agent=os.getenv("SEC_USER_AGENT", "FinAgent research demo admin@example.com"),
        cache_dir=Path(os.getenv("FINAGENT_CACHE_DIR", ".finagent_cache")).expanduser(),
        **values,
    )


SETTING_NAMES = [f.name for f in fields(Settings)]

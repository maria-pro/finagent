"""Runtime settings: which model serves which agent role, and where data is cached.

Three profiles ship by default:

* ``local``   - one small NVIDIA Nemotron model (Nemotron 3 Nano 4B) served on CPU
  by Ollama or llama.cpp. No GPU, no API key, and nothing leaves the machine
  except the public market-data requests.
* ``nim``     - the NVIDIA API catalog (build.nvidia.com): small models for the
  specialist sub-agents, a larger model for orchestration, NeMo Retriever
  embeddings and a NemoGuard topic-control rail.
* ``bedrock`` - Nemotron models served by Amazon Bedrock through its
  OpenAI-compatible endpoint (Bedrock API key).

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
    "bedrock": {
        # Bedrock's OpenAI-compatible Chat Completions endpoint; {region} comes from AWS_REGION.
        "base_url": "https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
        "api_key": "",
        "orchestrator_model": "nvidia.nemotron-super-3-120b",
        "specialist_model": "nvidia.nemotron-nano-3-30b",
        "embed_model": "",
        "guard_model": "",
        "temperature": 1.0,
        "top_p": 0.95,
        "reasoning": False,
        # Bedrock exposes Nemotron reasoning as an effort level (none/low/medium/high).
        "reasoning_style": "effort",
        "max_concurrency": 4,
        "timeout_s": 180.0,
    },
}

# Design choices that can be switched off one at a time to measure what each contributes
# (FINAGENT_ABLATE=facts,coverage ...); see finagent/evaluate.py.
ABLATIONS = {
    "facts": "the model reads raw tool JSON instead of pre-labelled facts",
    "phases": "output-format rules sit in the system prompt from the start; no nudge after a data-free answer",
    "coverage": "no coverage check: tool calls the model skips are not run for it",
    "revision": "no revision passes after the fact-check",
    "own_evidence": "agents are fact-checked against everyone's evidence, not just their own",
    "attribution": "the fact-check traces numbers only; claims are not matched to their sources",
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
    # How reasoning is switched on/off: "auto" probes the server; or "ollama", "template"
    # (chat_template_kwargs: NIM, vLLM, llama.cpp), "effort" (reasoning_effort), "none".
    reasoning_style: str = "auto"
    ablate: frozenset[str] = frozenset()
    # With a topic-control rail configured, refuse questions when it does not answer,
    # instead of continuing on the deterministic rules alone (FINAGENT_TOPIC_RAIL_REQUIRED).
    topic_rail_required: bool = False

    @property
    def is_local(self) -> bool:
        return self.base_url.startswith(("http://127.0.0.1", "http://localhost", "http://0.0.0.0"))

    def model_for(self, role: str) -> str:
        return self.specialist_model if role == "specialist" else self.orchestrator_model

    def enabled(self, feature: str) -> bool:
        """False when ``feature`` has been switched off for an ablation run."""
        return feature not in self.ablate

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


def parse_ablations(text: str | None) -> frozenset[str]:
    items = frozenset(a.strip().lower() for a in (text or "").split(",") if a.strip())
    unknown = items - set(ABLATIONS)
    if unknown:
        raise ValueError(f"Unknown ablation(s) {sorted(unknown)}; choose from {sorted(ABLATIONS)}")
    return items


def load_settings(profile: str | None = None) -> Settings:
    name = (profile or os.getenv("FINAGENT_PROFILE") or "local").strip().lower()
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}; choose one of {sorted(PROFILES)}")
    values = {"reasoning_style": "auto", "topic_rail_required": False, **PROFILES[name]}
    if name == "nim":
        values["api_key"] = os.getenv("NVIDIA_API_KEY", "")
    if name == "bedrock":
        values["base_url"] = values["base_url"].format(region=os.getenv("AWS_REGION", "us-east-1"))
        values["api_key"] = os.getenv("BEDROCK_API_KEY") or os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("OPENAI_API_KEY", "")

    for key, current in list(values.items()):
        env = os.getenv(f"FINAGENT_{key.upper()}")
        if env is not None and env != "":
            values[key] = _coerce(env, current)

    return Settings(
        profile=name,
        sec_user_agent=os.getenv("SEC_USER_AGENT", "FinAgent research demo admin@example.com"),
        cache_dir=Path(os.getenv("FINAGENT_CACHE_DIR", ".finagent_cache")).expanduser(),
        ablate=parse_ablations(os.getenv("FINAGENT_ABLATE")),
        **values,
    )


SETTING_NAMES = [f.name for f in fields(Settings)]


def shared_ui() -> bool:
    """True when the web UI serves other people (FINAGENT_UI_SHARED=1): fixed endpoints, and
    each visitor sees only their own runs. The UI has no login - put it behind one."""
    return os.getenv("FINAGENT_UI_SHARED", "").strip().lower() in {"1", "true", "yes", "on"}


def ui_api_key(base: Settings, base_url: str, typed_key: str = "") -> str:
    """The API key a request from the web UI may use. A key held by the server goes only to
    the endpoint the server is configured for: a visitor who points the UI somewhere else must
    bring their own key, or the server's key would be sent to an address they control."""
    if typed_key.strip():
        return typed_key.strip()
    same_endpoint = base_url.strip().rstrip("/") == base.base_url.strip().rstrip("/")
    return base.api_key if same_endpoint else ""

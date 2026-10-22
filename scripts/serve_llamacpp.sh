#!/usr/bin/env bash
# Serve NVIDIA Nemotron 3 Nano 4B with llama.cpp on CPU only (no GPU offload).
# Usage: scripts/serve_llamacpp.sh [path/to/model.gguf] [port] [threads]
# Then: FINAGENT_BASE_URL=http://127.0.0.1:8191/v1 finagent doctor
set -euo pipefail
MODEL="${1:-}"
PORT="${2:-8191}"
THREADS="${3:-$(( $(getconf _NPROCESSORS_ONLN) / 2 ))}"

if [[ -n "$MODEL" ]]; then
  SOURCE=(-m "$MODEL")
else
  SOURCE=(-hf nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M)   # downloads on first use
fi

# --jinja        use the model's own chat template (needed for tool calling)
# -ngl 0         keep every layer on the CPU
# -np 1          one request at a time; agents run sequentially on CPU
exec llama-server "${SOURCE[@]}" --jinja -ngl 0 -np 1 -c 16384 -t "$THREADS" \
  --alias nemotron-3-nano:4b --host 127.0.0.1 --port "$PORT"

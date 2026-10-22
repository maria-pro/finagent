#!/usr/bin/env bash
# Register NVIDIA's official Nemotron 3 Nano 4B GGUF with Ollama without `ollama pull`.
#
# Useful on university or corporate servers where the Ollama service cannot reach
# the internet (e.g. outbound traffic must go through a proxy the service was not
# started with) and you do not have root to change it. Download the GGUF with your
# own (proxied) shell, then import it: the Ollama CLI uploads it to the server.
#
# Usage: scripts/ollama_import_gguf.sh [model-name] [threads]
set -euo pipefail
NAME="${1:-nemotron-3-nano:4b}"
THREADS="${2:-4}"
DIR="${FINAGENT_MODEL_DIR:-$HOME/models}"
GGUF="$DIR/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"

mkdir -p "$DIR"
if [[ ! -f "$GGUF" ]]; then
  curl -L --fail -o "$GGUF.part" \
    "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF/resolve/main/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"
  mv "$GGUF.part" "$GGUF"
fi

# RENDERER/PARSER select Ollama's built-in Nemotron 3 chat template and tool-call
# parser - the same ones the library model `nemotron-3-nano:4b` uses.
# num_thread caps CPU use on a shared machine.
cat > "$DIR/Modelfile.nemotron" <<MODELFILE
FROM $GGUF
RENDERER nemotron-3-nano
PARSER nemotron-3-nano
PARAMETER num_thread $THREADS
PARAMETER num_ctx 16384
PARAMETER temperature 1
PARAMETER top_p 0.95
MODELFILE

ollama create "$NAME" -f "$DIR/Modelfile.nemotron"
ollama show "$NAME" | sed -n '/Capabilities/,/Parameters/p'

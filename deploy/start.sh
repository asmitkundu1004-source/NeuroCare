#!/bin/sh
# Container entry point: optionally bring up the fine-tuned Aura model, then the server.
set -e
mkdir -p "$NEUROCARE_DATA_DIR" "$OLLAMA_MODELS"

if [ -n "$AURA_GGUF_URL" ]; then
  ollama serve > /tmp/ollama.log 2>&1 &
  i=0
  until curl -sf http://127.0.0.1:11434/api/tags > /dev/null; do
    i=$((i + 1)); [ "$i" -gt 60 ] && echo "ollama did not start" && break
    sleep 1
  done
  if ! ollama list 2>/dev/null | grep -q "^aura"; then
    echo "Downloading the Aura model..."
    curl -fL --retry 5 "$AURA_GGUF_URL" -o /tmp/aura.gguf
    sed "s#^FROM .*#FROM /tmp/aura.gguf#" /app/models/aura-llm/Modelfile > /tmp/Modelfile
    ollama create aura -f /tmp/Modelfile
    rm -f /tmp/aura.gguf
  fi
fi

cd /app/backend
exec uvicorn neurocare.server:app --host 0.0.0.0 --port "${PORT:-7860}" --proxy-headers

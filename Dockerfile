# One container: the website, the API and (optionally) the fine-tuned Aura model.
# Works on Hugging Face Spaces (Docker SDK, free CPU tier), Render, Railway or any VM.
#
#   docker build -t neurocare .
#   docker run -p 7860:7860 -e AURA_GGUF_URL=https://huggingface.co/<you>/<repo>/resolve/main/aura.gguf neurocare
#
# Without AURA_GGUF_URL the server still runs sensing, safety and memory, and answers with
# its reflective responder.

FROM node:20-slim AS web
WORKDIR /app
COPY web/package.json web/package-lock.json web/
RUN cd web && npm ci --no-fund --no-audit
COPY web web
COPY backend/models backend/models
COPY research/face/face_landmarker.task research/face/face_landmarker.task
COPY research/results/summary.json research/results/summary.json
RUN cd web && npm run build

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY models/aura-llm/Modelfile models/aura-llm/Modelfile
COPY deploy/start.sh /app/start.sh
COPY --from=web /app/web/dist web/dist
RUN chmod +x /app/start.sh && useradd -m -u 1000 user && chown -R user /app
USER user
ENV HOME=/home/user \
    PORT=7860 \
    NEUROCARE_DATA_DIR=/home/user/data \
    OLLAMA_MODELS=/home/user/ollama \
    NEUROCARE_TTS=0 \
    NEUROCARE_CORS=*
EXPOSE 7860
CMD ["/app/start.sh"]

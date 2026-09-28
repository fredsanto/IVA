#!/usr/bin/env bash
# Start IVA on a single GPU machine: the vLLM model server, then the IVA web
# server (direct mode — the pipeline runs inside the web server process).
# Generic launcher (no SLURM, no environment modules); run it with the `iva`
# conda environment active, or as the Docker image's default command.
# For the Curnagl SLURM deployment use launch_qwen.sh instead.
#
# Environment variables (all optional):
#   PORT                    IVA web UI / API port          (default 8002)
#   VLLM_PORT               vLLM OpenAI-compatible port    (default 38103)
#   MAX_MODEL_LEN           vLLM context length            (default 32768)
#   GPU_MEMORY_UTILIZATION  fraction of GPU memory for vLLM (default 0.90)
#   VLLM_READY_TIMEOUT      seconds to wait for the model  (default 3600; the
#                           first run downloads ~18 GB of weights)
#   HF_HOME                 Hugging Face cache for the model weights
#   NCBI_API_KEY            NCBI E-utilities key (10 req/s instead of 3)
set -euo pipefail
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3.5-9B"   # the pipeline's model registry expects this exact id
PORT="${PORT:-8002}"
VLLM_PORT="${VLLM_PORT:-38103}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-3600}"

export PYTHONUNBUFFERED=1
export PIPELINE_BACKEND=direct
export VLLM_BASE_URL="http://localhost:${VLLM_PORT}"
mkdir -p logs

echo "[iva] starting vLLM ($MODEL) on port $VLLM_PORT — log: logs/vllm_server.log"
vllm serve "$MODEL" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --port "$VLLM_PORT" \
    --enable-prefix-caching \
    > logs/vllm_server.log 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT INT TERM

waited=0
until curl -sf "http://localhost:${VLLM_PORT}/v1/models" | grep -q "$MODEL"; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[iva] ERROR: vLLM exited — last lines of logs/vllm_server.log:" >&2
        tail -20 logs/vllm_server.log >&2
        exit 1
    fi
    if [ "$waited" -ge "$VLLM_READY_TIMEOUT" ]; then
        echo "[iva] ERROR: vLLM not ready after ${VLLM_READY_TIMEOUT}s — see logs/vllm_server.log" >&2
        exit 1
    fi
    sleep 10; waited=$((waited + 10))
done
echo "[iva] vLLM ready after ${waited}s"

echo "[iva] IVA web server on http://0.0.0.0:${PORT}"
# Foreground (not exec) so the trap above still stops vLLM on exit.
uvicorn server_qwen:app --host 0.0.0.0 --port "$PORT" --log-level info

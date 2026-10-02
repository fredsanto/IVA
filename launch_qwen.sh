#!/bin/bash --login
#SBATCH --job-name=server_qwen
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=server_qwen_%j.log
#SBATCH --error=server_qwen_%j.log

# ── Usage ──────────────────────────────────────────────────────────────────────
#   sbatch launch_qwen.sh
#
# Environment variables:
#   PIPELINE_BACKEND=direct|proxy   (default: direct)
#   PORT=8002                       (ServerQwen port)
#   VLLM_PORT=8001                  (vLLM port)
#   PIPELINE_PORT=8000              (pipeline server port, proxy mode only)
#   NCBI_MAX_RPS=10                 (NCBI requests/s cap; default 10 with key, else 3.
#                                    NCBI limits per key — N servers sharing the key:
#                                    10/N each, e.g. 4 servers → 2.5)
#   DRIVER_CMD="..."                (optional batch client, run from this directory
#                                    once vLLM is ready, with SERVER_HOST=localhost
#                                    and SERVER_PORT=$PORT; the job ends when it exits)
#   DRIVER_LOG=path                 (driver output; default driver_<JOBID>.log)

# ── Paths (defined first — used immediately below) ─────────────────────────────
# SLURM sets SLURM_SUBMIT_DIR to the directory `sbatch` was run from — per the
# documented usage ("cd ServerQwen && sbatch launch_qwen.sh") that's this
# script's own directory. Falls back to $(pwd) for non-SLURM/manual runs.
SERVERQWN_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PIPELINE_DIR="$SERVERQWN_DIR/Qwen_Engine_IVA/IVA_vllm"
CONDA_ENV="$SERVERQWN_DIR/Qwen_Engine_IVA/env_vllm_0606"
VENV_UVICORN="$SERVERQWN_DIR/../.venv_qwen/bin/uvicorn"

BACKEND="${PIPELINE_BACKEND:-direct}"
PORT="${PORT:-8002}"
VLLM_PORT="${VLLM_PORT:-38103}"
PIPELINE_PORT="${PIPELINE_PORT:-8000}"

# Python fully-buffers stdout when it's not a TTY (i.e. always, here — stdout
# is redirected to server_qwen_<JOBID>.log by #SBATCH --output above). Plain
# print() calls in the pipeline (normalizer.py's column-interpretation
# summary, executor.py's per-tool progress, etc.) then sit in the buffer
# instead of reaching the log file promptly — a run can be fully in progress
# for many minutes while the log looks stalled, indistinguishable from an
# actual hang. logging module calls are unaffected (they flush per record),
# which is why access logs/vLLM-ready lines always showed up fine while plain
# prints didn't. Forcing unbuffered I/O makes the log reflect real-time state.
export PYTHONUNBUFFERED=1

# NCBI E-utilities key (10 req/s instead of 3). Kept outside the repo.
[ -r "$HOME/.ncbi_api_key" ] && export NCBI_API_KEY="$(cat "$HOME/.ncbi_api_key")"

cd "$SERVERQWN_DIR" || exit 1

# ── HPC software environment ──────────────────────────────────────────────────
dcsrsoft use 20241118
module load miniforge3
module load cuda
conda_init
conda activate "$CONDA_ENV"

# ── Compute node info ─────────────────────────────────────────────────────────
NODE=$(hostname -s)

echo "[launch] Starting Qwen Variant Analysis Server (backend=$BACKEND, port=$PORT)"
echo "[launch] Compute node: $NODE"

# ── Clear stale processes on our ports (leftover from a prior job on this node) ─
echo "[launch] Clearing any stale processes on ports $VLLM_PORT, $PORT, $PIPELINE_PORT..."
fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true
fuser -k "${PORT}/tcp" 2>/dev/null || true
fuser -k "${PIPELINE_PORT}/tcp" 2>/dev/null || true
sleep 2

# ── Start vLLM ────────────────────────────────────────────────────────────────
vllm serve Qwen/Qwen3.5-9B \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --port "$VLLM_PORT" \
    --enable-prefix-caching \
    > "$SERVERQWN_DIR/vllm_server.log" 2>&1 &
VLLM_PID=$!
echo "[launch] vLLM PID=$VLLM_PID (port $VLLM_PORT)"

# The final command below is exec'd, replacing this script's process image
# with uvicorn — vLLM (backgrounded above) is still its child, but without an
# explicit trap the shell exiting via exec (or being killed) can orphan it on
# proctrack configs that don't fully cgroup-track exec'd descendants, leaving
# it bound to $VLLM_PORT for the next job that lands on this node. Kill it
# explicitly on any exit/signal instead of relying on implicit cleanup.
trap 'kill "$VLLM_PID" 2>/dev/null' EXIT INT TERM

# ── vLLM readiness watcher (background, non-blocking) ─────────────────────────
# Web UI does NOT wait on this — it starts immediately below. This just logs
# when the model finishes loading so you know when /analyze will actually work.
# With DRIVER_CMD set, the batch client runs here once vLLM is ready, inside
# this job (a separate CPU job would sit idle waiting on HTTP replies), and the
# job is cancelled when the client exits or vLLM never comes up.
(
    ready=0
    for i in $(seq 1 120); do
        model_id=$(curl -sf "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null \
                   | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || true)
        if [[ "$model_id" == "Qwen/Qwen3.5-9B" ]]; then
            echo "[launch] vLLM ready (Qwen/Qwen3.5-9B) after $((i * 5))s"
            ready=1
            break
        elif [[ -n "$model_id" ]]; then
            echo "[launch] ERROR: vLLM port ${VLLM_PORT} serving wrong model: $model_id (not Qwen/Qwen3.5-9B)"
            break
        fi
        sleep 5
    done
    [ "$ready" = 1 ] || [ -n "$model_id" ] || echo "[launch] ERROR: vLLM did not become ready within 600s. See vllm_server.log"
    [ -n "$DRIVER_CMD" ] || exit 0
    if [ "$ready" = 1 ]; then
        driver_log="${DRIVER_LOG:-$SERVERQWN_DIR/driver_${SLURM_JOB_ID}.log}"
        echo "[launch] Driver starting (log: $driver_log)"
        SERVER_HOST=localhost SERVER_PORT="$PORT" bash -c "$DRIVER_CMD" > "$driver_log" 2>&1
        echo "[launch] Driver exited with status $?"
    fi
    echo "[launch] Ending job"
    scancel "$SLURM_JOB_ID"
) &

# ── Backend-specific setup ────────────────────────────────────────────────────
if [ "$BACKEND" = "proxy" ]; then
    echo "[launch] Proxy mode — starting pipeline server on port $PIPELINE_PORT (background, not blocking web UI)..."
    export VLLM_BASE_URL="http://localhost:${VLLM_PORT}"

    (
        cd "$PIPELINE_DIR" || exit 1
        uvicorn server:app \
            --host 0.0.0.0 \
            --port "$PIPELINE_PORT" \
            --log-level info \
            > "$SERVERQWN_DIR/pipeline_server.log" 2>&1
    ) &
    PIPELINE_PID=$!
    echo "[launch] Pipeline server PID=$PIPELINE_PID (port $PIPELINE_PORT)"

    (
        for i in $(seq 1 30); do
            if curl -sf "http://localhost:${PIPELINE_PORT}/health" > /dev/null 2>&1; then
                echo "[launch] Pipeline server ready after $((i * 2))s"
                exit 0
            fi
            sleep 2
        done
        echo "[launch] ERROR: pipeline server did not become ready within 60s. See pipeline_server.log"
    ) &

    export PIPELINE_BACKEND=proxy
    export PIPELINE_SERVER_URL="http://localhost:${PIPELINE_PORT}"
else
    export PIPELINE_BACKEND=direct
    export VLLM_BASE_URL="http://localhost:${VLLM_PORT}"
fi

# ── Write connection info ──────────────────────────────────────────────────────
cat > "$SERVERQWN_DIR/.connection" <<-CONN_EOF
SERVERQWN_NODE=$NODE
SERVERQWN_PORT=$PORT
SERVERQWN_PID=$$
CONN_EOF

echo ""
echo "============================================================"
echo " CONNECTION INFO"
echo "============================================================"
echo " Compute node : $NODE"
echo " Port         : $PORT"
echo ""
echo " From your laptop, run this SSH tunnel:"
echo ""
echo "   ssh -N -L ${PORT}:${NODE}:${PORT} ${USER}@<login-host>"
echo ""
echo " Then open http://localhost:${PORT} in your browser"
echo "============================================================"
echo ""
echo "[launch] Web UI starting now. Model still loading in background"
echo "[launch] (~7-8 min) — /analyze requests will fail until vLLM ready."
echo "[launch] Watch this log for '[launch] vLLM ready ...'."
echo ""

# Not exec'd (deliberately) — exec replaces this script's process image
# without running EXIT/TERM traps, which would silently disable the vLLM
# cleanup trap above right when it's needed (SLURM killing the job). Running
# uvicorn as a plain foreground command keeps this shell alive to catch the
# signal and clean up vLLM before exiting.
"$VENV_UVICORN" server_qwen:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info

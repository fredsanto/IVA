# Installable packages

ServerQwen ships as two independently installable packages, split along
`server_qwen.py`'s existing `direct`/`proxy` backend modes:

| Package | What it is | Needs a GPU? |
|---|---|---|
| [`iva-gpu.def`](#1-gpu--hpc-package-apptainersingularity) | Apptainer/Singularity image: Python 3.11 + vLLM + the CUDA stack vLLM's own wheels bring + the pipeline's Python deps | Yes — this is the model-serving side |
| [`webapp/`](#2-webapp-package-pip-installable) | pip package `iva-webapp`: the FastAPI web UI in `proxy` mode only (fastapi/uvicorn/httpx, no torch/vLLM/CUDA) | No — talks to package 1 over HTTP |

Run both on the same GPU node for a self-contained deployment, or run
package 2 anywhere (laptop, login node, a separate lightweight VM) pointed
at package 1 running on a cluster — same split `launch_qwen.sh`'s
`PIPELINE_BACKEND=proxy` mode already supports today.

---

## 1. GPU / HPC package (Apptainer/Singularity)

`iva-gpu.def` builds the **runtime environment only** — no pipeline
code and no model weights are baked in, so the image is a fixed ~7.6GB
regardless of code changes or which model you point vLLM at.

### Build

```bash
module load apptainer          # or your cluster's module name for it
apptainer build --fakeroot iva-gpu.sif iva-gpu.def
```

Takes several minutes — it builds a full conda env (`vllm`, `torch`,
`transformers`, and everything else in
`environment.yml` (repo root), the same
file the native install and the Dockerfile use). No GPU needed to build,
only to run.

### Get the code and model weights in at run time

- **Pipeline code** (`Qwen_Engine_IVA/IVA_vllm/`,
  `prompts/`) — bind-mount the repo; Apptainer auto-binds your `$HOME` and
  current directory by default, so running from inside the repo checkout
  usually needs no extra flag. On Curnagl, if your checkout lives under
  `/work/...`, add it explicitly: `--bind /work/PRTNR/...:/work/PRTNR/...`.
- **Model weights** (`Qwen/Qwen3.5-9B`, ~18GB) — vLLM downloads them from
  the Hugging Face Hub on first run, into `$HOME/.cache/huggingface` by
  default (reused on every later run, not re-downloaded per job). To avoid
  filling your home-directory quota, point the cache at shared project
  storage first:
  ```bash
  export HF_HOME=/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/ServerQwen/.hf_cache
  ```

### Run vLLM (on a GPU node — `--nv` maps the host driver in)

```bash
apptainer exec --nv iva-gpu.sif \
    vllm serve Qwen/Qwen3.5-9B \
    --dtype bfloat16 --max-model-len 32768 \
    --gpu-memory-utilization 0.90 --port 38103 --enable-prefix-caching
```

### Run the pipeline's own FastAPI server (`direct` mode)

```bash
cd Qwen_Engine_IVA/IVA_vllm
export VLLM_BASE_URL=http://localhost:38103
apptainer exec --nv /path/to/iva-gpu.sif \
    uvicorn server:app --host 0.0.0.0 --port 8000
```

`launch_qwen.sh` still works as-is (conda env, not this image) — swapping
its `conda activate "$CONDA_ENV"` line for an `apptainer exec --nv` call is
a later step if/when you want SLURM to launch from the image instead.

---

## 2. Webapp package (pip-installable)

`webapp/` is the thin client: `server_qwen.py` running in `proxy` mode
(never imports the pipeline directly) plus the bundled `templates/index.html`
UI. No GPU/CUDA/torch dependency.

### Install

From a checkout of this repo:
```bash
pip install ./webapp
```

Or from the wheel attached to a GitHub Release (no repo checkout needed):
```bash
pip install iva_webapp-0.1.0-py3-none-any.whl
```

### Run

```bash
export PIPELINE_SERVER_URL=http://localhost:8000   # wherever package 1's FastAPI server is reachable
export PORT=8002
iva-webapp
```

Then open `http://localhost:8002` — or tunnel from a laptop to wherever
it's running the same way `tunnel_qwen.sh` already does:
```bash
ssh -N -L 8002:<node>:8002 <user>@curnagl.dcsr.unil.ch
```

### Note on `results/`

The webapp writes saved reports to a `results/` directory next to wherever
`iva_webapp` is installed (same behavior `server_qwen.py` has always
had). If you `pip install` it into a shared site-packages, be aware
reports land there, not in your working directory.

---

## Which one do I need?

- **Running the model on your own GPU cluster:** package 1 (and
  optionally package 2 on the same node, or pointed at it remotely).
- **Just the UI, model server already running elsewhere:** package 2 only.
- **Everything on one box, as today:** both — this is what `launch_qwen.sh`
  already does with the conda env; the two packages here are an
  alternative, more portable way to get the same split.

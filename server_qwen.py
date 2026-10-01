"""
server_qwen.py — Streaming variant analysis server with live SSE output.

Dual-mode backend:
  - "proxy" (default): delegates to the existing pipeline server via HTTP
  - "direct": imports the pipeline module directly (requires filesystem access)

Set PIPELINE_SERVER_URL to point to the existing server in proxy mode.
Set PIPELINE_BACKEND=direct to use direct import mode.
"""

import asyncio
import json
import logging
import math
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("server_qwen")

# ── Configuration ──────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).parent
_TEMPLATES_DIR = _SCRIPT_DIR / "templates"
_RESULTS_DIR = _SCRIPT_DIR / "results"
_PIPELINE_SERVER_URL = os.environ.get(
    "PIPELINE_SERVER_URL", "http://localhost:8000"
)
_BACKEND = os.environ.get("PIPELINE_BACKEND", "proxy")

_ERC_FOLDER = _SCRIPT_DIR / "Qwen_Engine_IVA" / "IVA_vllm"

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_cleanup_old_jobs())
    yield

app = FastAPI(title="Qwen Variant Analysis Server — Live Streaming", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ───────────────────────────────────────────────────────
# {job_id: {"status": str, "events": [dict], "result": str|None, "error": str|None}}
_jobs: dict[str, dict] = {}
_lock = asyncio.Lock()

# ── Per-job log capture ────────────────────────────────────────────────────────
_job_logs: dict[str, list] = {}
_active_job_id: str | None = None


class _JobLogHandler(logging.Handler):
    _SKIP = frozenset({"httpx", "httpcore", "uvicorn.access", "hpack", "h2"})

    def emit(self, record: logging.LogRecord) -> None:
        if record.name in self._SKIP:
            return
        jid = _active_job_id
        if jid and jid in _job_logs:
            try:
                buf = _job_logs[jid]
                buf.append(self.format(record))
                if len(buf) > 2000:
                    del buf[:1000]
            except Exception:
                pass


_job_log_handler = _JobLogHandler(logging.INFO)
_job_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger().addHandler(_job_log_handler)

# ── Backend detection ─────────────────────────────────────────────────────────

def _pipeline_accessible() -> bool:
    """Check if the pipeline module can be imported directly."""
    try:
        import sys
        sys.path.insert(0, str(_ERC_FOLDER))
        import pipeline.pipeline  # noqa: F401
        return True
    except Exception:
        return False

_DIRECT_AVAILABLE = _pipeline_accessible()

if _BACKEND == "direct" and not _DIRECT_AVAILABLE:
    logger.warning(
        "PIPELINE_BACKEND=direct but pipeline module is not accessible. "
        "Falling back to proxy mode (PIPELINE_SERVER_URL=%s).",
        _PIPELINE_SERVER_URL,
    )
    _BACKEND = "proxy"

logger.info("Backend mode: %s", _BACKEND)
if _BACKEND == "proxy":
    logger.info("Pipeline server URL: %s", _PIPELINE_SERVER_URL)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _sanitize_nan(obj):
    """Replace NaN/Infinity floats with 'NA' — Starlette's JSONResponse rejects
    them (allow_nan=False), and they should read as 'NA' like the rest of the
    pipeline's missing-value convention anyway."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return "NA"
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


async def _push_event(job_id: str, event: dict) -> None:
    event = _sanitize_nan(event)
    async with _lock:
        if job_id in _jobs:
            _jobs[job_id]["events"].append(event)
            # Slide the TTL on every event so a still-running job (heavy full
            # pipeline, many variants + web search) never gets swept mid-flight
            # just because it started more than max_age ago.
            _jobs[job_id]["ts"] = datetime.now().timestamp()


async def _finish_job(job_id: str) -> None:
    global _active_job_id
    async with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done"
            # Reset the TTL clock to completion time — long runs (many variants,
            # deep per-variant web search) can themselves take close to or over
            # the 1h cleanup window, which was deleting jobs from memory right as
            # they finished, 404-ing /chat before the user ever saw the result.
            _jobs[job_id]["ts"] = datetime.now().timestamp()
    if _active_job_id == job_id:
        _active_job_id = None


async def _cancel_job(job_id: str) -> bool:
    global _active_job_id
    async with _lock:
        job = _jobs.get(job_id)
        if job is None or job["status"] != "running":
            return False
        task = job.get("task")
        if task and not task.done():
            task.cancel()
        job["status"] = "cancelled"
        job["events"].append({"type": "cancelled", "message": "Job cancelled by user."})
    if _active_job_id == job_id:
        _active_job_id = None
    return True


async def _cleanup_old_jobs(max_age: int = 3600) -> None:
    """Remove jobs older than max_age seconds."""
    while True:
        await asyncio.sleep(300)
        now = datetime.now().timestamp()
        async with _lock:
            stale = [
                jid for jid, job in _jobs.items()
                if job.get("ts", 0) < now - max_age
            ]
            for jid in stale:
                del _jobs[jid]
            if stale:
                logger.info("Cleaned up %d stale job(s)", len(stale))


def _make_sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _report_header(filename: str, phenotype: str) -> str:
    """Leading block naming the input file and phenotype used for this run,
    so a saved report.txt is self-identifying without cross-referencing
    the results/ directory timestamp against the original upload."""
    sep = "=" * 60
    return (
        f"{sep}\n"
        f"INPUT\n"
        f"{sep}\n"
        f"File: {filename}\n"
        f"Patient phenotype: {phenotype}\n\n"
    )


async def _run_proxy(job_id: str, csv_bytes: bytes, filename: str, phenotype: str):
    """Call the existing pipeline server and stream the result."""
    global _active_job_id
    _job_logs[job_id] = []
    _active_job_id = job_id
    url = f"{_PIPELINE_SERVER_URL}/analyze"
    await _push_event(job_id, {"type": "status", "message": "Contacting pipeline server..."})

    async def _heartbeat() -> None:
        elapsed = 0
        while True:
            await asyncio.sleep(15)
            elapsed += 15
            async with _lock:
                if _jobs.get(job_id, {}).get("status") != "running":
                    return
            await _push_event(job_id, {
                "type": "status",
                "message": f"Pipeline running... ({elapsed}s elapsed)",
            })

    heartbeat = asyncio.create_task(_heartbeat())
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:
            files = {"csv_file": (filename, csv_bytes, "text/csv")}
            data = {"patient_report": phenotype}
            await _push_event(job_id, {
                "type": "status", "message": "Pipeline running (retrieval, reasoning, conclusion stages)..."
            })

            response = await client.post(url, files=files, data=data)
            response.raise_for_status()
            result = _report_header(filename, phenotype) + response.text
    except httpx.ConnectError:
        heartbeat.cancel()
        await _push_event(job_id, {
            "type": "error",
            "message": f"Cannot connect to pipeline server at {url}. Is it running?",
        })
        await _finish_job(job_id)
        return
    except httpx.HTTPStatusError as e:
        heartbeat.cancel()
        await _push_event(job_id, {
            "type": "error",
            "message": f"Pipeline server returned HTTP {e.response.status_code}: {e.response.text[:500]}",
        })
        await _finish_job(job_id)
        return
    except Exception as e:
        heartbeat.cancel()
        await _push_event(job_id, {
            "type": "error",
            "message": f"Pipeline request failed: {e}",
        })
        await _finish_job(job_id)
        return

    heartbeat.cancel()
    await _push_event(job_id, {"type": "status", "message": "Pipeline complete!"})
    await _push_event(job_id, {"type": "result", "data": result})
    await _finish_job(job_id)


_CTX_VARIANT_IDX = re.compile(r'VARIANT\s+(\d+)\s*:', re.IGNORECASE)
_CTX_GENE       = re.compile(r'(?m)^\s*Gene:\s*(\S+)')


def _ctx_info(variant_context: str) -> tuple[int, str]:
    """Extract (1-based variant index, gene) from a context string."""
    idx_m  = _CTX_VARIANT_IDX.search(variant_context)
    gene_m = _CTX_GENE.search(variant_context)
    idx  = int(idx_m.group(1)) if idx_m else 0
    gene = gene_m.group(1)     if gene_m else ""
    return idx, gene


async def _run_direct(job_id: str, csv_bytes: bytes, filename: str, phenotype: str):
    """
    Thin wrapper around _run_direct_impl — this is scheduled via a fire-and-forget
    asyncio.create_task() (see /analyze) with nothing ever awaiting the result, so
    any exception that escapes _run_direct_impl uncaught previously killed the task
    silently: asyncio logs one generic "Task exception was never retrieved" line
    (no job_id, easy to miss) and the job's SSE event list just stops growing
    forever — the client hangs until the 1h stale-job sweep evicts it and reads
    back as "Job not found", with no real diagnostic trail. A real observed case:
    an unmapped "Variant" CSV field reaching detect_genome_build() as a bare float
    NaN raised AttributeError deep in a helper, well before the try/except that
    already wraps the pipeline run itself. This wrapper is the last line of
    defense — it must never let anything past it silently.
    """
    try:
        await _run_direct_impl(job_id, csv_bytes, filename, phenotype)
    except Exception as e:
        logger.exception("Unhandled exception in _run_direct (job_id=%s)", job_id)
        await _push_event(job_id, {"type": "error", "message": f"Internal error: {e}"})
        await _finish_job(job_id)


async def _run_direct_impl(job_id: str, csv_bytes: bytes, filename: str, phenotype: str):
    """Import and run the pipeline directly, streaming per-variant stage progress."""
    global _active_job_id
    _job_logs[job_id] = []
    _active_job_id = job_id
    import sys
    import threading
    sys.path.insert(0, str(_ERC_FOLDER))

    from pipeline.core.normalizer import normalize_upload
    from pipeline.pipeline import Pipeline
    import pipeline.stages.first_triage as _triage_mod
    import pipeline.stages.reasoning as _reasoning_mod
    import pipeline.stages.conclusion as _conclusion_mod

    await _push_event(job_id, {"type": "status", "message": "Inspecting CSV header with SLM..."})

    from pipeline.llm.vllm_client import VLLMClient
    vllm_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8001")
    header_llm = VLLMClient(base_url=vllm_url)

    try:
        variants, raw_rows, parental_ab, header_mapping_summary = normalize_upload(
            csv_bytes, filename, llm=header_llm
        )
    except ValueError as e:
        await _push_event(job_id, {"type": "error", "message": f"Normalization failed: {e}"})
        await _finish_job(job_id)
        return

    await _push_event(job_id, {"type": "status", "message": header_mapping_summary})

    if not variants:
        await _push_event(job_id, {
            "type": "error", "message": "No variant rows found in CSV."
        })
        await _finish_job(job_id)
        return

    from pipeline.core.build_detection import detect_genome_build
    genome_build = detect_genome_build(variants)
    await _push_event(job_id, {
        "type": "status",
        "message": f"Detected genome build: {genome_build}",
    })

    n = len(variants)
    await _push_event(job_id, {
        "type": "progress",
        "stage": "normalized",
        "n_variants": n,
        "variants": [
            {
                "idx":     i + 1,
                "gene":    v.get("Gene", "?"),
                "variant": v.get("Variant", v.get("cDNA", "?")),
            }
            for i, v in enumerate(variants)
        ],
        "message": f"Loaded {n} variant(s). Initializing pipeline...",
    })

    pipeline_obj = Pipeline(
        model_name="qwen3.5-9b",
        llm_kwargs={"base_url": vllm_url},
    )
    await _push_event(job_id, {
        "type": "status",
        "message": f"Pipeline initialized. Starting retrieval for {n} variant(s)...",
    })

    # ── Per-variant progress via monkey-patching ───────────────────────────────
    # pipeline.run() is async-def but uses blocking ThreadPoolExecutor internally,
    # so it would freeze the event loop. We run it in a thread with its own sub-loop,
    # keeping the main loop free to deliver SSE events in real-time.

    main_loop  = asyncio.get_running_loop()
    _counts: dict[str, int] = {"retrieval": 0, "triage": 0, "reasoning": 0, "conclusion": 0}
    _tick_lock = threading.Lock()   # local lock — does NOT shadow the module-level asyncio _lock

    def _thread_push(event: dict) -> None:
        """Push SSE event from any thread back to the main event loop."""
        asyncio.run_coroutine_threadsafe(_push_event(job_id, event), main_loop)

    def _tick(stage: str, variant_idx: int = 0, gene: str = "") -> None:
        with _tick_lock:
            _counts[stage] += 1
            done = _counts[stage]
        msg = f"{stage.capitalize()}: variant {done}/{n}"
        if gene:
            msg += f" — {gene}"
        _thread_push({
            "type":        "progress",
            "stage":       stage,
            "done":        done,
            "total":       n,
            "variant_idx": variant_idx,
            "gene":        gene,
            "message":     msg,
        })

    # Patch executor.run_variant for per-variant retrieval ticks.
    # retrieval.py calls run_variant(variant=..., variant_index=i, ...) — all kwargs.
    _orig_run_variant = pipeline_obj._executor.run_variant

    def _patched_run_variant(variant, **kwargs):
        result = _orig_run_variant(variant, **kwargs)
        gene   = variant.get("Gene", variant.get("Variant", ""))
        idx    = kwargs.get("variant_index", -1) + 1   # 0-based → 1-based
        _tick("retrieval", variant_idx=idx, gene=gene)
        return result

    pipeline_obj._executor.run_variant = _patched_run_variant

    # Patch module-level stage functions.
    # pipeline.py looks up first_triage.run_one at call time (module attr lookup),
    # so patching the module attribute IS visible inside pipeline.run().
    _orig_triage         = _triage_mod.run_one
    _orig_reasoning      = _reasoning_mod.run_reasoning
    _orig_second_triage  = _reasoning_mod.run_second_triage
    _orig_conclusion     = _conclusion_mod.run_one

    def _patched_triage(variant_context, patient_phenotype, llm):
        result         = _orig_triage(variant_context, patient_phenotype, llm)
        idx, gene      = _ctx_info(variant_context)
        _tick("triage", variant_idx=idx, gene=gene)
        return result

    def _patched_reasoning(variant_context, llm, **kwargs):
        # **kwargs (not a fixed signature) — same reasoning as
        # _patched_second_triage below: a future parameter added to
        # reasoning.run_reasoning (e.g. gene_phenotype_block) must pass
        # through automatically, or every variant's reasoning call raises
        # "unexpected keyword argument" and fails silently into an
        # empty-looking report. This wrapper's previous fixed signature
        # (sibling_context_block/inheritance_mode_block only) hit exactly
        # that failure the moment gene_phenotype_block was added.
        result    = _orig_reasoning(variant_context, llm, **kwargs)
        idx, gene = _ctx_info(variant_context)
        _tick("reasoning", variant_idx=idx, gene=gene)
        return result

    def _patched_second_triage(variant_context, reasoning_text, llm, **kwargs):
        # **kwargs (not a fixed signature) so any future parameter added to
        # reasoning.run_second_triage (e.g. is_x_linked,
        # include_single_hit_recessive, include_compound_het_exception,
        # include_literature_evidence_quality — all added for conditional
        # prompt assembly) passes through automatically. A real observed
        # failure this fixes: this wrapper's previous fixed signature only
        # forwarded sibling_context_block, so pipeline.py calling with
        # is_x_linked=... raised "unexpected keyword argument 'is_x_linked'"
        # and second_triage failed outright for every variant — silently
        # producing an empty-looking report (no MOI layers, "no genetic
        # findings") that read as a correct exclusion but was actually a
        # crash. Matches the same **kwargs pattern _patched_run_variant
        # above already uses for exactly this reason.
        return _orig_second_triage(variant_context, reasoning_text, llm, **kwargs)

    def _patched_conclusion(variant_context, reasoning, cross_analysis, llm):
        result    = _orig_conclusion(variant_context, reasoning, cross_analysis, llm)
        idx, gene = _ctx_info(variant_context)
        _tick("conclusion", variant_idx=idx, gene=gene)
        return result

    _triage_mod.run_one              = _patched_triage
    _reasoning_mod.run_reasoning     = _patched_reasoning
    _reasoning_mod.run_second_triage = _patched_second_triage
    _conclusion_mod.run_one          = _patched_conclusion

    # Run the pipeline in a thread with its own event loop so the main loop stays free.
    def _run_pipeline_sync() -> str:
        sub_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(sub_loop)
        try:
            return sub_loop.run_until_complete(
                pipeline_obj.run(
                    variants, phenotype, raw_rows=raw_rows, parental_ab=parental_ab,
                    header_mapping_summary=header_mapping_summary,
                    genome_build=genome_build,
                )
            )
        finally:
            sub_loop.close()

    try:
        report = await main_loop.run_in_executor(None, _run_pipeline_sync)
    except Exception as e:
        await _push_event(job_id, {"type": "error", "message": f"Pipeline failed: {e}"})
        logger.exception("Pipeline direct run failed")
        await _finish_job(job_id)
        return
    finally:
        # Always restore patches
        _triage_mod.run_one              = _orig_triage
        _reasoning_mod.run_reasoning     = _orig_reasoning
        _reasoning_mod.run_second_triage = _orig_second_triage
        _conclusion_mod.run_one          = _orig_conclusion
        pipeline_obj._executor.run_variant = _orig_run_variant

    report = _report_header(filename, phenotype) + report

    stem      = Path(filename).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = _RESULTS_DIR / f"{stem}_{timestamp}"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.txt").write_text(report, encoding="utf-8")
        logger.info("Result saved to %s", run_dir)
    except Exception as e:
        logger.warning("Could not save result: %s", e)

    await _push_event(job_id, {"type": "status", "message": "Pipeline complete!"})
    await _push_event(job_id, {"type": "result", "data": report})
    await _finish_job(job_id)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": _BACKEND,
        "pipeline_server": _PIPELINE_SERVER_URL if _BACKEND == "proxy" else None,
        "direct_available": _DIRECT_AVAILABLE,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    path = _TEMPLATES_DIR / "index.html"
    if not path.exists():
        return HTMLResponse("<h1>ServerQwen — template not found</h1>")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.post("/analyze")
async def analyze(
    csv_file: UploadFile = File(..., description="Variants CSV file"),
    patient_report: str = Form(..., description="Free-text patient phenotype"),
):
    raw = await csv_file.read()
    if not raw.strip():
        raise HTTPException(status_code=400, detail="CSV file is empty")

    job_id = uuid.uuid4().hex[:12]
    filename = csv_file.filename or "upload.csv"

    async with _lock:
        _jobs[job_id] = {
            "status": "running",
            "ts": datetime.now().timestamp(),
            "events": [{"type": "status", "message": "Job submitted."}],
            "result": None,
            "error": None,
            # Original upload, kept verbatim for /chat — the synthesized report
            # doesn't always preserve every raw field the clinician submitted.
            "raw_csv_text": raw.decode("utf-8", errors="replace"),
            "raw_filename": filename,
            "raw_phenotype": patient_report,
        }

    if _BACKEND == "direct":
        task = asyncio.create_task(_run_direct(job_id, raw, filename, patient_report))
    else:
        task = asyncio.create_task(_run_proxy(job_id, raw, filename, patient_report))

    async with _lock:
        _jobs[job_id]["task"] = task

    return {"job_id": job_id, "stream_url": f"/stream/{job_id}"}


@app.get("/stream/{job_id}")
async def stream_events(job_id: str, request: Request):
    async with _lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        sent = 0
        while True:
            if await request.is_disconnected():
                break

            async with _lock:
                job = _jobs.get(job_id)
                if job is None:
                    yield _make_sse({"type": "error", "message": "Job not found"})
                    break

                new_events = job["events"][sent:]
                sent = len(job["events"])

                for ev in new_events:
                    yield _make_sse(ev)

                if job["status"] in ("done", "cancelled"):
                    break

            await asyncio.sleep(0.2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    cancelled = await _cancel_job(job_id)
    if not cancelled:
        async with _lock:
            job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(status_code=409, detail=f"Job status is '{job['status']}', cannot cancel")
    logger.info("Job %s cancelled by user", job_id)
    return {"cancelled": True, "job_id": job_id}


class ChatRequest(BaseModel):
    question: str


@app.post("/chat/{job_id}")
async def chat(job_id: str, req: ChatRequest):
    async with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    diagnosis = None
    for ev in reversed(job["events"]):
        if ev.get("type") == "result":
            diagnosis = ev["data"]
            break
    if diagnosis is None:
        raise HTTPException(status_code=400, detail="No result available yet")

    # Full report (process details + augmented context + reasoning + final report) can
    # exceed the model's context window for large variant batches. Chat only needs the
    # FINAL REPORT section (included conclusions + clinical synthesis).
    marker = "FINAL REPORT"
    marker_idx = diagnosis.find(marker)
    diagnosis_for_chat = diagnosis[marker_idx:] if marker_idx != -1 else diagnosis
    # Observed on the A100 node: 80,000 combined chars of dense clinical/genomic
    # text (gene symbols, HGVS, scores) tokenized to 31,269 tokens — ~2.56
    # chars/token, well below the usual ~4 chars/token English estimate. Caps
    # below assume a conservative 2.0 chars/token so a large multi-variant batch
    # can't walk past --max-model-len 32768 the way it did here (400 Bad Request:
    # "prompt contains at least 31269 input tokens" + max_tokens=1500 = 32769).
    MAX_DIAGNOSIS_CHARS = 42000
    if len(diagnosis_for_chat) > MAX_DIAGNOSIS_CHARS:
        diagnosis_for_chat = (
            diagnosis_for_chat[:MAX_DIAGNOSIS_CHARS]
            + "\n\n[...report truncated for length...]"
        )

    # Original upload, verbatim — the synthesized report can drop or reword raw
    # fields, so chat gets the source data directly instead of only the model's
    # processed version of it.
    raw_csv_text = job.get("raw_csv_text", "")
    MAX_RAW_CSV_CHARS = 15000
    if len(raw_csv_text) > MAX_RAW_CSV_CHARS:
        raw_csv_text = raw_csv_text[:MAX_RAW_CSV_CHARS] + "\n\n[...input truncated for length...]"
    original_input_block = (
        "ORIGINAL INPUT DATA (as submitted by the clinician, before any processing):\n\n"
        f"Patient phenotype (as submitted): {job.get('raw_phenotype', '')}\n\n"
        f"Variant table ({job.get('raw_filename', 'upload.csv')}):\n{raw_csv_text}\n\n"
    ) if raw_csv_text else ""

    vllm_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8001")

    async def stream_chat():
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{vllm_url}/v1/chat/completions",
                    json={
                        "model": "Qwen/Qwen3.5-9B",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a clinical genetics assistant. "
                                    f"{original_input_block}"
                                    "The following variant analysis report was generated for a patient:\n\n"
                                    f"{diagnosis_for_chat}\n\n"
                                    "If the clinician's question concerns a raw field (e.g. exact frequency, "
                                    "zygosity, family ID) not restated in the report, check the ORIGINAL INPUT "
                                    "DATA above rather than saying it's unavailable. "
                                    "Answer follow-up questions about this report clearly and concisely. "
                                    "Limit your response to 1000 words maximum."
                                ),
                            },
                            {"role": "user", "content": req.question},
                        ],
                        "stream": True,
                        "max_tokens": 1500,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            yield _make_sse({"type": "done"})
                            return
                        try:
                            data = json.loads(chunk)
                            delta = data["choices"][0]["delta"].get("content") or ""
                            if delta:
                                yield _make_sse({"type": "token", "text": delta})
                        except Exception:
                            pass
        except httpx.HTTPStatusError as e:
            try:
                body = (await e.response.aread()).decode(errors="replace")[:300]
            except Exception:
                body = "<body unavailable>"
            yield _make_sse({"type": "error", "message": f"vLLM error {e.response.status_code}: {body}"})
        except Exception as e:
            yield _make_sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        stream_chat(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    async with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    events = job["events"]
    for ev in reversed(events):
        if ev.get("type") == "result":
            return {"status": "done", "result": ev["data"]}
        # A job that ended in error/cancellation has no result event — report
        # it as terminal, not "running", or pollers wait until the stale-job sweep.
        if ev.get("type") in ("error", "cancelled"):
            return {"status": ev["type"], "message": ev.get("message", ""), "events": events}
    return {"status": "running", "events": events}


@app.get("/activity/{job_id}")
async def activity_stream(job_id: str, request: Request):
    """SSE stream of captured log lines for live activity panel."""
    async with _lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")

    async def log_generator():
        sent = 0
        while True:
            if await request.is_disconnected():
                break
            buf = _job_logs.get(job_id)
            if buf is not None:
                n = len(buf)
                if sent > n:
                    sent = max(0, n - 50)
                if sent < n:
                    for line in buf[sent:n]:
                        yield _make_sse({"type": "log", "line": line})
                    sent = n
            async with _lock:
                job = _jobs.get(job_id)
                if job and job["status"] in ("done", "cancelled"):
                    buf = _job_logs.get(job_id, [])
                    n = len(buf)
                    if sent < n:
                        for line in buf[sent:n]:
                            yield _make_sse({"type": "log", "line": line})
                    yield _make_sse({"type": "done"})
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)

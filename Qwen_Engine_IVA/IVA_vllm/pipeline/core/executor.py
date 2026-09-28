"""
pipeline/core/executor.py — Tool fan-out, compression, and mini-doc assembly.

The Executor runs all pipeline tools for a single variant, in manifest order,
and assembles their outputs into a mini-doc string that the reasoning stage consumes.

Sub-steps (per MIGRATION.md):
  1. Helper functions verbatim from server_main.py
  2. Tool fan-out logic (ported from run_browsing())
  3. Compression logic
  4. Mini-doc assembly

Public API:
    Executor(llm, tools, manifests)
    Executor.run_variant(variant, patient_phenotype, variant_index, total_variants) -> str

Design rules enforced here:
  - Python gate() always takes precedence over manifest gate
  - PipelineError subclasses are caught per-tool; failure writes a note into the mini-doc
  - Compression is applied after run(), before appending to the mini-doc
  - Tools never see each other's LLM calls — each gets a fresh ToolContext slice
"""

from __future__ import annotations

import re
import time
import logging
from typing import TYPE_CHECKING, Optional

from pipeline.config import DEFAULT_GENOME_BUILD
from pipeline.core.context import ToolContext
from pipeline.core.errors import PipelineError, SLMError, ToolGateError
from pipeline.core.manifest import ManifestLoader, eval_manifest_gate

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient
    from pipeline.tools.base import Tool

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS  (verbatim from server_main.py lines 230–247)
# ═══════════════════════════════════════════════════════════════════════════════

_RSID_RE = re.compile(r"RS_ID=(rs\d+)", re.IGNORECASE)


def _extract_rsid(variant_str: str) -> Optional[str]:
    """Extract rsID from a key=value variant string. Returns None if absent or 'NA'."""
    m = _RSID_RE.search(variant_str)
    if m and m.group(1).lower() != "na":
        return m.group(1)
    return None


def _field_from(variant_str: str, name: str) -> str:
    """Extract a single field value from a key=value variant string."""
    m = re.search(rf"{name}=([^,]+)", variant_str, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_pvs1_strength(pvs1_block: str) -> str:
    """Parse the PVS1 strength line from an AutoPVS1 output block."""
    for line in pvs1_block.splitlines():
        if "PVS1 strength" in line:
            return line.split(":", 1)[-1].strip()
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# COMPRESSION
# ═══════════════════════════════════════════════════════════════════════════════

# Auto-threshold: compress when variant count exceeds this value
_AUTO_THRESHOLD = 10

# Prompt loaded from prompts/compression.txt at runtime
_COMPRESSION_PROMPT_PATH = (
    __import__("pathlib").Path(__file__).parent.parent.parent / "prompts" / "compression.txt"
)

_COMPRESSION_PROMPT_FALLBACK = (
    "Compress the following variant evidence to {max_tokens} tokens maximum.\n"
    "Preserve: key findings, ACMG criteria, evidence strength, URLs.\n"
    "Remove: repetition, verbose explanations, raw data already interpreted.\n\n"
    "Evidence:\n{variant_report}"
)


def _load_compression_prompt() -> str:
    if _COMPRESSION_PROMPT_PATH.exists():
        return _COMPRESSION_PROMPT_PATH.read_text(encoding="utf-8")
    return _COMPRESSION_PROMPT_FALLBACK


def _should_compress(manifest: dict, total_variants: int) -> bool:
    """Return True if the manifest's compress config requires compression at this scale."""
    compress = manifest.get("compress", {})
    if not compress or not compress.get("enabled", False):
        return False
    threshold = compress.get("threshold", "auto")
    if threshold == "auto":
        threshold = _AUTO_THRESHOLD
    try:
        return total_variants > int(threshold)
    except (ValueError, TypeError):
        return False


def _apply_compression(
    output: str,
    manifest: dict,
    context: ToolContext,
) -> str:
    """
    Compress a tool output string according to the manifest's compress config.

    Strategies:
      slm         — call context.llm.generate() with the compression prompt
      truncate    — hard truncate to max_tokens characters (approximate)
      first_n_lines — keep the first N lines
    """
    compress    = manifest.get("compress", {})
    strategy    = compress.get("strategy", "truncate")
    max_tokens  = int(compress.get("max_tokens", 300))

    if strategy == "slm":
        try:
            prompt_template = _load_compression_prompt()
            prompt = prompt_template.format(
                max_tokens=max_tokens,
                variant_report=output,
            )
            compressed = context.llm.generate(
                system=(
                    "You are a clinical evidence compressor. "
                    "Preserve all clinically significant findings. "
                    "Return only the compressed text, no commentary."
                ),
                user=prompt,
                max_tokens=max_tokens,
            )
            return compressed.strip()
        except Exception as e:
            logger.warning(
                "SLM compression failed for %s (%s) — falling back to truncate",
                manifest.get("name", "?"), e,
            )
            # Fall through to truncate

    if strategy == "first_n_lines":
        lines = output.splitlines()
        n = max(1, max_tokens // 10)
        return "\n".join(lines[:n])

    # Default / fallback: truncate by character count (rough token proxy)
    char_limit = max_tokens * 4  # ~4 chars per token
    if len(output) > char_limit:
        return output[:char_limit] + "\n[... truncated ...]"
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

def _gate_reason(tool: "Tool", variant: dict) -> str:
    """Best-effort human-readable reason for gate skip."""
    tool_reason = getattr(tool, "_last_gate_reason", None)
    if tool_reason:
        return tool_reason
    type_val = variant.get("Type", "NA")
    hgvs_val = variant.get("HGVS", "NA")
    return f"type={type_val!r} hgvs={hgvs_val!r}"


def _extract_tool_metadata(tool: "Tool") -> dict | None:
    """
    Safely extract per-tool metadata attributes set during run().
    Uses getattr with None default — never raises AttributeError.
    Returns None if the tool has no metadata attributes.
    """
    metadata = {}

    # LitVar2SummaryTool metadata
    last_query   = getattr(tool, "_last_query",   None)
    last_n_found = getattr(tool, "_last_n_found", None)
    last_kept    = getattr(tool, "_last_kept",    None)
    if last_query is not None or last_kept is not None:
        metadata["query"]   = last_query
        metadata["n_found"] = last_n_found
        metadata["kept"]    = last_kept  # list of {pmid, title}

    # WebSearchAgentTool metadata
    last_trace = getattr(tool, "_last_trace", None)
    if last_trace is not None:
        metadata["trace"] = last_trace  # list of strings

    return metadata if metadata else None


class Executor:
    """
    Runs all pipeline tools for a single variant and assembles their outputs
    into a mini-doc string for the reasoning stage.

    Args:
        llm:       Shared LLMClient instance (injected at server startup).
        tools:     List of pipeline Tool instances (AutoPVS1Tool, LitVar2SummaryTool, …).
        manifests: Loaded manifest dicts from ManifestLoader.load_all().

    The executor matches tools to manifests by name. Tools without a matching
    manifest are skipped with a warning.
    """

    def __init__(
        self,
        llm: "LLMClient",
        tools: list["Tool"],
        manifests: list[dict],
    ):
        self._llm = llm
        self.process_log: list[dict] = []

        # Build name → manifest lookup
        manifest_by_name = {m["name"]: m for m in manifests}

        # Build ordered list of (tool, manifest) — only tools that have a manifest
        paired: list[tuple["Tool", dict]] = []
        for tool in tools:
            manifest = manifest_by_name.get(tool.name)
            if manifest is None:
                logger.warning(
                    "No manifest found for tool %r — skipping.", tool.name
                )
                continue
            if not manifest.get("enabled", True):
                continue
            paired.append((tool, manifest))

        # Sort by (order, name) — same order as ManifestLoader.load_all()
        paired.sort(key=lambda p: (p[1].get("order", 99), p[1].get("name", "")))
        self._paired: list[tuple["Tool", dict]] = paired

    # ── public ────────────────────────────────────────────────────────────────

    def run_variant(
        self,
        variant: dict,
        patient_phenotype: str,
        variant_index: int,
        total_variants: int,
        genome_build: str = DEFAULT_GENOME_BUILD,
        raw_fields: dict | None = None,
    ) -> tuple[str, list[dict]]:
        """
        Run all tools for one variant and return (mini_doc, log_entries).

        The mini-doc is the concatenation of each tool's (possibly compressed)
        output, with failure notes substituted for any PipelineError raised.
        log_entries is a list of per-tool process log dicts for this variant;
        callers accumulate and set executor.process_log after all variants complete.

        Args:
            variant:          Normalized variant dict (canonical schema).
            patient_phenotype: Free-text patient phenotype string.
            variant_index:    0-based position of this variant in the run.
            total_variants:   Total number of variants being processed.

        Returns:
            (mini_doc, log_entries) — thread-safe: no shared state mutated.
        """
        local_log: list[dict] = []

        all_outputs: dict[str, Optional[str]] = {}
        variant_report = ""

        for tool, manifest in self._paired:
            # Build a fresh context for each tool so all_outputs stays current
            context = ToolContext(
                variant=variant,
                patient_phenotype=patient_phenotype,
                all_outputs=dict(all_outputs),    # snapshot — tool sees prior outputs
                variant_report=variant_report,
                variant_index=variant_index,
                total_variants=total_variants,
                llm=self._llm,
                genome_build=genome_build,
                raw_fields=raw_fields or {},
            )

            try:
                # ── Gate evaluation ───────────────────────────────────────────
                gate_result = self._eval_gate(tool, manifest, variant, context)
                if not gate_result:
                    logger.debug("[Executor] Gate SKIP — %s", tool.name)
                    local_log.append({
                        "variant_index": variant_index,
                        "tool_name":     tool.name,
                        "gate":          "SKIP",
                        "gate_reason":   _gate_reason(tool, variant),
                        "raw_output":    None,
                        "metadata":      None,
                    })
                    continue

                # ── Tool execution ────────────────────────────────────────────
                logger.info("[Executor] Running %s (variant %d/%d)",
                            tool.name, variant_index + 1, total_variants)

                output = self._run_with_retry(tool, variant, context, manifest)

                # ── Capture raw output and metadata BEFORE compression ────────
                raw_output = output
                tool_metadata = _extract_tool_metadata(tool)

                # ── Compression ───────────────────────────────────────────────
                if output and _should_compress(manifest, total_variants):
                    output = _apply_compression(output, manifest, context)

                all_outputs[tool.name] = output

                if output:
                    variant_report += output + "\n\n"

                local_log.append({
                    "variant_index": variant_index,
                    "tool_name":     tool.name,
                    "gate":          "PASS",
                    "gate_reason":   None,
                    "raw_output":    raw_output,
                    "metadata":      tool_metadata,
                })

            except PipelineError as e:
                error_note = (
                    f"{tool.name.upper()} [FAILED — {type(e).__name__}]: {e}"
                )
                logger.error("[Executor] %s", error_note)
                all_outputs[tool.name] = None
                variant_report += error_note + "\n\n"
                local_log.append({
                    "variant_index": variant_index,
                    "tool_name":     tool.name,
                    "gate":          "PASS",
                    "gate_reason":   None,
                    "raw_output":    f"[FAILED — {type(e).__name__}]: {e}",
                    "metadata":      None,
                })

            except Exception as e:
                # Unexpected — log and continue so one bad tool doesn't kill the run
                error_note = (
                    f"{tool.name.upper()} [FAILED — UnexpectedError]: {e}"
                )
                logger.exception("[Executor] Unexpected error in %s", tool.name)
                all_outputs[tool.name] = None
                variant_report += error_note + "\n\n"
                local_log.append({
                    "variant_index": variant_index,
                    "tool_name":     tool.name,
                    "gate":          "PASS",
                    "gate_reason":   None,
                    "raw_output":    f"[FAILED — UnexpectedError]: {e}",
                    "metadata":      None,
                })

        return variant_report.strip(), local_log

    # ── private ───────────────────────────────────────────────────────────────

    def _eval_gate(
        self,
        tool: "Tool",
        manifest: dict,
        variant: dict,
        context: ToolContext,
    ) -> bool:
        """
        Evaluate gate for a tool. Python gate() always takes precedence.

        Returns True (run) or False (skip).
        Raises ToolGateError only if gate evaluation itself errors.
        """
        # Python gate — takes precedence per CLAUDE.md rule
        try:
            python_gate_result = tool.gate(variant, context)
            # If the base Tool.gate() returns True (default), fall through to manifest gate.
            # If a subclass explicitly overrides gate(), honour it directly.
            # We detect an override by checking if the method is not Tool.gate itself.
            from pipeline.tools.base import Tool as _BaseTool
            if type(tool).gate is not _BaseTool.gate:
                return bool(python_gate_result)
        except PipelineError:
            raise
        except Exception as e:
            raise ToolGateError(
                f"Python gate() raised an unexpected error in {tool.name}: {e}"
            ) from e

        # Manifest gate (only reached if tool uses the default Tool.gate)
        try:
            manifest_result = eval_manifest_gate(manifest, variant, context)
        except Exception as e:
            raise ToolGateError(
                f"Manifest gate evaluation failed for {tool.name}: {e}"
            ) from e

        if manifest_result is None:
            # SLM gate — call LLM
            return self._eval_slm_gate(manifest, variant, context)

        return bool(manifest_result)

    def _eval_slm_gate(
        self,
        manifest: dict,
        variant: dict,
        context: ToolContext,
    ) -> bool:
        """Evaluate an SLM gate by calling context.llm.generate with the gate prompt."""
        gate   = manifest.get("gate", {})
        prompt = gate.get("prompt", "")
        if not prompt:
            return True

        # Fill any {FieldName} placeholders in the gate prompt
        try:
            filled = prompt.format(**{k: variant.get(k, "NA") for k in variant})
        except (KeyError, ValueError):
            filled = prompt  # leave as-is if format fails

        try:
            answer = context.llm.generate(
                system="Answer with YES or NO only.",
                user=filled,
                max_tokens=5,
            )
            return "yes" in answer.strip().lower()
        except Exception as e:
            raise ToolGateError(
                f"SLM gate call failed for {manifest.get('name', '?')}: {e}"
            ) from e

    def _run_with_retry(
        self,
        tool: "Tool",
        variant: dict,
        context: ToolContext,
        manifest: dict,
    ) -> Optional[str]:
        """Call tool.run() with retry logic from the manifest config."""
        retry_cfg = manifest.get("retry", {})
        attempts  = max(1, int(retry_cfg.get("attempts", 1)))
        backoff   = float(retry_cfg.get("backoff", 2.0))

        last_exc = None
        for attempt in range(attempts):
            try:
                return tool.run(variant, context)
            except PipelineError as e:
                last_exc = e
                if attempt < attempts - 1:
                    logger.warning(
                        "[Executor] %s attempt %d/%d failed: %s — retrying in %.1fs",
                        tool.name, attempt + 1, attempts, e, backoff,
                    )
                    time.sleep(backoff)
                else:
                    raise

        raise last_exc  # unreachable but satisfies type checkers

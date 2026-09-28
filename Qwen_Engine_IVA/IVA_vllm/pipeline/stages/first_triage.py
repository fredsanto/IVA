"""
pipeline/stages/first_triage.py — Post-retrieval SLM triage stage.

One small SLM call per variant immediately after retrieval.
Decides whether a variant can be discarded with high confidence before
the more expensive reasoning stage runs.

Expected SLM output format (from prompts/first_triage.txt):
    Keep-case: <strongest reason to keep, ≤10 words>
    Discard-case: <strongest reason to discard, ≤10 words>
    Decision: KEEP or DISCARD

Public API:
    run_one(variant_context, patient_phenotype, llm) -> tuple[str, str]
        Returns ("KEEP", justification) or ("DISCARD", justification).
        justification is "Keep: <...> | Discard: <...>" for auditability.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH   = Path(__file__).parent.parent.parent / "prompts" / "first_triage.txt"
_SYSTEM_PROMPT = "You are an expert clinical geneticist."


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"First-triage prompt not found at {_PROMPT_PATH}.")


_KEEP_CASE_RE    = re.compile(r"Keep-case:\s*(.+)",    re.IGNORECASE)
_DISCARD_CASE_RE = re.compile(r"Discard-case:\s*(.+)", re.IGNORECASE)
_DECISION_RE     = re.compile(r"Decision:\s*(KEEP|DISCARD)", re.IGNORECASE)


def parse_triage(raw: str) -> tuple[str, str]:
    """
    Parse the raw SLM reply for a first-triage call.

    Expected format (blank lines tolerated between lines):
        Keep-case: <strongest reason to keep, ≤10 words>
        Discard-case: <strongest reason to discard, ≤10 words>
        Decision: KEEP or DISCARD

    Returns:
        ("KEEP",    justification)  if Decision is KEEP.
        ("DISCARD", justification)  if Decision is DISCARD.
        ("KEEP",    "parse error — defaulting to KEEP") on any parse failure.

    justification = "Keep: <keep_case> | Discard: <discard_case>" for auditability.
    Only the Decision: value affects the keep/discard outcome.
    """
    # Strip and drop blank lines so accidental empty lines don't confuse the regex search.
    cleaned = "\n".join(line for line in raw.splitlines() if line.strip())

    keep_m    = _KEEP_CASE_RE.search(cleaned)
    discard_m = _DISCARD_CASE_RE.search(cleaned)
    decision_m = _DECISION_RE.search(cleaned)

    keep_str    = keep_m.group(1).strip()    if keep_m    else ""
    discard_str = discard_m.group(1).strip() if discard_m else ""

    logger.debug(
        "[FirstTriage] Keep-case: %r | Discard-case: %r | raw Decision line: %r",
        keep_str, discard_str,
        decision_m.group(0) if decision_m else "(not found)",
    )

    if decision_m:
        decision = decision_m.group(1).upper()
        justification = f"Keep: {keep_str} | Discard: {discard_str}" if (keep_str or discard_str) else decision
        return (decision, justification)

    logger.warning("[FirstTriage] Could not parse response: %r — defaulting to KEEP", raw[:200])
    return ("KEEP", "parse error — defaulting to KEEP")


def run_one(
    variant_context: str,
    patient_phenotype: str,
    llm: "LLMClient",
) -> tuple[str, str]:
    """
    First triage — fast discard gate immediately after retrieval.

    Args:
        variant_context:   Per-variant context string from Stage 1 (retrieval).
        patient_phenotype: Free-text patient phenotype string.
        llm:               Shared LLMClient instance.

    Returns:
        ("KEEP", justification) or ("DISCARD", justification) where justification
        is "Keep: <keep_case> | Discard: <discard_case>" from the SLM output.
        On SLM or parse failure always returns ("KEEP", ...) — never silently
        drops a variant.
    """
    user_prompt = (
        _load_prompt()
        .replace("{patient_phenotype}", patient_phenotype)
        .replace("{variant_context}", variant_context)
    )

    try:
        raw = llm.generate(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=100,
        )
    except Exception as exc:
        logger.warning("[FirstTriage] SLM call failed: %s — defaulting to KEEP", exc)
        return ("KEEP", f"SLM error — defaulting to KEEP: {exc}")

    return parse_triage(raw)

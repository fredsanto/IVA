"""
pipeline/stages/first_triage.py — Post-retrieval SLM triage stage.

One small SLM call per variant immediately after retrieval.
Decides whether a variant can be discarded with high confidence before
the more expensive reasoning stage runs.

SLM answer (prompts/first_triage.txt): JSON {keep_case, discard_case, decision},
decision constrained to KEEP|DISCARD at decoding time (json_schema).

Public API:
    run_one(variant_context, patient_phenotype, llm) -> tuple[str, str]
        Returns ("KEEP", justification) or ("DISCARD", justification).
        justification is "Keep: <...> | Discard: <...>" for auditability.
"""

from __future__ import annotations

import json
import logging
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


_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "keep_case": {"type": "string"},
        "discard_case": {"type": "string"},
        "decision": {"type": "string", "enum": ["KEEP", "DISCARD"]},
    },
    "required": ["keep_case", "discard_case", "decision"],
}


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
        is "Keep: <keep_case> | Discard: <discard_case>" from the SLM answer.
        On SLM failure returns ("KEEP", ...) — never silently drops a variant.
    """
    user_prompt = (
        _load_prompt()
        .replace("{patient_phenotype}", patient_phenotype)
        .replace("{variant_context}", variant_context)
    )

    try:
        answer = json.loads(llm.generate(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=200,
            json_schema=_DECISION_SCHEMA,
        ))
    except Exception as exc:
        logger.warning("[FirstTriage] SLM call failed: %s — defaulting to KEEP", exc)
        return ("KEEP", f"SLM error — defaulting to KEEP: {exc}")

    return (answer["decision"],
            f"Keep: {answer['keep_case'].strip()} | Discard: {answer['discard_case'].strip()}")

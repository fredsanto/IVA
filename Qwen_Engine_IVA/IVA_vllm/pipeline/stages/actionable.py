"""
pipeline/stages/actionable.py — ACMG Secondary Findings (SF) actionable variants section.

Runs once after final_conclusion, over the fixed set of variants flagged by
pipeline/core/acmg_sf.py (gene in ACMG SF v3.2 list + P/LP). Independent of
phenotype fit / MOI layers by design — a flagged variant is forced through
triage/second-triage upstream in pipeline.py regardless of its phenotype
relevance, so this stage only has to narrate what was already determined.

Prompt loaded from prompts/actionable_variants.txt.

Public API:
    run(flagged, patient_phenotype, llm) -> str   ("" if flagged is empty)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "actionable_variants.txt"

MAX_NEW_TOKENS_ACTIONABLE = 500


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"actionable_variants prompt not found at {_PROMPT_PATH}.")


def _build_variants_text(flagged: list[dict]) -> str:
    blocks = []
    for f in flagged:
        blocks.append(
            f"Variant {f['index'] + 1} — {f['gene']} ({f['hgvs']})\n"
            f"ACMG SF condition: {f['condition']}\n"
            f"Basis for P/LP call: {f['reason']}\n"
            f"Zygosity: {f['zygosity']}\n"
            f"Base ACMG classification: {f.get('classification') or 'not separately scored'}"
        )
    return "\n\n---\n\n".join(blocks)


def run(
    flagged: list[dict],
    patient_phenotype: str,
    llm: "LLMClient",
) -> str:
    """
    Args:
        flagged: list of dicts, one per ACMG-SF-flagged variant, each with keys
                 index, gene, hgvs, condition, reason, zygosity, classification.
        patient_phenotype: Free-text patient phenotype (context only).
        llm: Shared LLMClient instance.

    Returns:
        The "# Actionable Variants" paragraph, or "" if flagged is empty.
    """
    if not flagged:
        return ""

    logger.info("[Actionable] Generating ACMG SF actionable variants section for %d variant(s)...", len(flagged))

    template    = _load_prompt()
    user_prompt = (template
        .replace("{patient_phenotype}", patient_phenotype)
        .replace("{variants}", _build_variants_text(flagged)))

    return llm.generate(
        system="You are an expert clinical geneticist reporting ACMG secondary findings.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_ACTIONABLE,
    ).strip()

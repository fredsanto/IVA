"""
pipeline/stages/moi_denovo.py — MOI Layer 3: de novo analysis.

For variants whose gene has a dominant-relevant inheritance mode (AD, AD_AR,
XLD). Adds PS2 (confirmed de novo) or PM6 (assumed de novo) on top of the
Layer 2 base ACMG score — never re-scores base criteria. Both require BOTH
parents tested: without a trio the delta is forced to +0 in code.

Prompt loaded from prompts/moi_denovo.txt.

Public API:
    run_one(variant_context, base_conclusion, segregation, has_trio, has_one_parent, llm) -> str
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.core.citations import validate_citations
from pipeline.core.clinvar_reference import append_clinvar_reference
from pipeline.core.acmg_points import resync_moi_total, splice_base_and_total
from pipeline.core.acmg_bs2_dominant import validate_bs2_unaffected_dominant_carrier

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "moi_denovo.txt"

MAX_NEW_TOKENS_DENOVO = 700

_PS2_PM6_BULLET_RE = re.compile(r"(?m)^[ \t]*[-*][ \t]*\**(?:PS2|PM6)\b[^\n]*\n?")
_DENOVO_DELTA_RE = re.compile(r"(?mi)^\*\*De novo delta:\*\*[^\n]*$")


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"moi_denovo prompt not found at {_PROMPT_PATH}.")


def _parental_data_status(has_trio: bool, has_one_parent: bool) -> str:
    if has_trio:
        return "trio (both parents)"
    if has_one_parent:
        return "partial (one parent only)"
    return "none"


def run_one(
    variant_context: str,
    base_conclusion: str,
    segregation: str,
    has_trio: bool,
    has_one_parent: bool,
    llm: "LLMClient",
) -> str:
    """
    MOI Layer 3 — de novo analysis for one variant.

    Args:
        variant_context: Per-variant context string from retrieval (same as base layer).
        base_conclusion: Layer 2's full structured output for this variant — the
                          ground truth this stage grounds itself in and adds to.
        segregation:      classify_segregation() result for this variant
                           ("de_novo" / "maternal" / "paternal" / "both_carriers" /
                           "homozygous_parent" / "uncertain" / "insufficient_data").
        has_trio:          True if all three (proband/mother/father) AB values present.
        has_one_parent:    True if exactly one parent's AB is present (not a trio).
        llm:               Shared LLMClient instance.

    Returns:
        Structured de novo layer block, including "Total ACMG points: [base+delta] → [classification]".
    """
    logger.info("[MOIDenovo] Generating de novo analysis for variant...")

    template = _load_prompt()
    parental_data_status = _parental_data_status(has_trio, has_one_parent)

    user_prompt = (template
        .replace("{base_conclusion}", base_conclusion)
        .replace("{segregation_block}", segregation)
        .replace("{parental_data_status}", parental_data_status))

    result = llm.generate(
        system="You are an expert clinical geneticist evaluating de novo occurrence. Limit your response to 400 words maximum.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_DENOVO,
    )
    # PS2/PM6 need both parents tested — a variant absent in the one tested
    # parent may come from the untested one. Without a trio, drop any PS2/PM6
    # bullet and force the delta to +0, whatever the model wrote.
    if not has_trio:
        result = _PS2_PM6_BULLET_RE.sub("", result)
        result = _DENOVO_DELTA_RE.sub(
            "**De novo delta:** +0 (PS2/PM6 require both parents tested)", result, count=1
        )
    # Base criteria + Total come from code, never from the model: splice the
    # Stage-4 base block, then (after BS2 below) Total = base + delta.
    result = splice_base_and_total(result, base_conclusion)
    # This layer also handles AD_AR-gene variants whose segregation resolves
    # to genuinely inherited (maternal/paternal), not de novo — the same
    # unaffected-carrier BS2 gap moi_dominant.py guards against applies here
    # too, or this layer's copy of the base conclusion keeps an un-penalized
    # score while the Dominant-Inherited layer (same variant) correctly
    # applies it. See validate_bs2_unaffected_dominant_carrier's docstring.
    result = validate_bs2_unaffected_dominant_carrier(result, base_conclusion, segregation)
    result = resync_moi_total(result)
    full_context = variant_context + "\n" + base_conclusion
    result = validate_citations(result, full_context)
    return append_clinvar_reference(result, full_context)

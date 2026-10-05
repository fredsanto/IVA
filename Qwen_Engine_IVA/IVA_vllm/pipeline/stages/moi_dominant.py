"""
pipeline/stages/moi_dominant.py — MOI Layer 4: dominant-inherited analysis.

For every included variant whose gene has a dominant-relevant inheritance
mode (AD, AD_AR, XLD), whatever the segregation: segregation is evidence, never
an entry condition. Adds PP1 (cosegregation) on top of the Layer 2 base ACMG
score when a parent carries the variant (het confirmed by both ABs, or
homozygous) and family-history evidence supports that parent being affected;
otherwise +0 and the base score stands — never re-scores base criteria.

Prompt loaded from prompts/moi_dominant.txt.

Public API:
    run_one(variant_context, base_conclusion, segregation, proband_ab_class,
             transmitting_parent_ab_class, llm) -> str
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.core.citations import validate_citations
from pipeline.core.clinvar_reference import append_clinvar_reference
from pipeline.core.acmg_points import resync_moi_total, splice_base_and_total
from pipeline.core.acmg_bs2_dominant import validate_bs2_unaffected_dominant_carrier

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "moi_dominant.txt"

MAX_NEW_TOKENS_DOMINANT = 700


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"moi_dominant prompt not found at {_PROMPT_PATH}.")


def _ab_confirmation_block(segregation: str, proband_ab_class: str, transmitting_parent_ab_class: str) -> str:
    """Backend-determined confirmation text — both proband's own AB and the
    transmitting parent's AB must independently classify as "het" (~0.5) for
    this to read as confirmed. Per user spec: BOTH values are checked, not
    just one."""
    if segregation == "homozygous_parent":
        return f"Proband AB class: {proband_ab_class} | a parent is HOMOZYGOUS for this variant (AB ~1.0)"
    if segregation not in ("maternal", "paternal"):
        return f"Not applicable — segregation is '{segregation}', no parent shown to carry the variant."
    parent_label = "mother" if segregation == "maternal" else "father"
    confirmed = proband_ab_class == "het" and transmitting_parent_ab_class == "het"
    return (
        f"Proband AB class: {proband_ab_class} | {parent_label.capitalize()} AB class: {transmitting_parent_ab_class} "
        f"-> {'CONFIRMED heterozygous in both' if confirmed else 'NOT confirmed heterozygous in both'}"
    )


def run_one(
    variant_context: str,
    base_conclusion: str,
    segregation: str,
    proband_ab_class: str,
    transmitting_parent_ab_class: str,
    llm: "LLMClient",
) -> str:
    """
    MOI Layer 4 — dominant-inherited analysis for one variant.

    Args:
        variant_context:               Per-variant context string from retrieval.
        base_conclusion:                Layer 2's full structured output for this variant.
        segregation:                    classify_segregation() result for this variant.
        proband_ab_class:               classify_ab_ratio() result for the proband's own AB.
        transmitting_parent_ab_class:   classify_ab_ratio() result for whichever parent
                                         matches `segregation` (mother if "maternal",
                                         father if "paternal"); pass "uncertain" if
                                         segregation isn't maternal/paternal.
        llm:                             Shared LLMClient instance.

    Returns:
        Structured dominant-inherited layer block, including
        "Total ACMG points: [base+delta] → [classification]".
    """
    logger.info("[MOIDominant] Generating dominant-inherited analysis for variant...")

    template = _load_prompt()
    ab_confirmation_block = _ab_confirmation_block(segregation, proband_ab_class, transmitting_parent_ab_class)

    user_prompt = (template
        .replace("{base_conclusion}", base_conclusion)
        .replace("{segregation_block}", segregation)
        .replace("{ab_confirmation_block}", ab_confirmation_block))

    result = llm.generate(
        system="You are an expert clinical geneticist evaluating dominant inheritance and cosegregation. Limit your response to 400 words maximum.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_DOMINANT,
    )
    # Base criteria + Total come from code, never from the model: splice the
    # Stage-4 base block, then (after BS2 below) Total = base + delta.
    result = splice_base_and_total(result, base_conclusion)
    # Mechanical DEFAULT-UNAFFECTED-POLICY BS2 check, before resync so the
    # inserted bullet gets folded into the Base/Total point lines below —
    # see validate_bs2_unaffected_dominant_carrier's docstring.
    result = validate_bs2_unaffected_dominant_carrier(result, base_conclusion, segregation)
    result = resync_moi_total(result)
    full_context = variant_context + "\n" + base_conclusion
    result = validate_citations(result, full_context)
    return append_clinvar_reference(result, full_context)

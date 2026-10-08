"""
pipeline/stages/moi_xlinked.py — MOI Layer 6: X-linked analysis.

For variants whose gene is confirmed on chromosome X (via pipeline/core/moi.py's
gene_chromosome). Distinguishes XLR (proband AB ~1.0, mother AB ~0.5 carrier)
from XLD (proband AB ~0.5, mother clinically affected) via
pipeline/core/segregation.py's classify_xlinked_ab for the numeric half, with
the phenotype half (mother affected / affected maternal-line relatives)
resolved by the LLM from retrieved evidence. Reuses PP1 (no new criterion
code) on top of the Layer 2 base ACMG score — never re-scores base criteria.

Prompt loaded from prompts/moi_xlinked.txt.

Public API:
    run_one(variant_context, base_conclusion, xlinked_pattern, segregation, gene_mode, llm) -> str
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.core.citations import validate_citations
from pipeline.core.clinvar_reference import append_clinvar_reference
from pipeline.core.acmg_points import cap_layer_pp1, resync_moi_total, splice_base_and_total
from pipeline.core.acmg_bs2_dominant import validate_bs2_unaffected_dominant_carrier

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "moi_xlinked.txt"

MAX_NEW_TOKENS_XLINKED = 700

_XLINKED_LABELS = {
    "XLR":       "XLR — proband AB ~1.0 (hemizygous/homozygous), mother AB ~0.5 (carrier).",
    "XLD":       "XLD — proband AB ~0.5 (heterozygous); mother's affected status must be confirmed from evidence below.",
    "uncertain": "uncertain — allelic-balance pattern does not clearly match XLR or XLD.",
}


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"moi_xlinked prompt not found at {_PROMPT_PATH}.")


def run_one(
    variant_context: str,
    base_conclusion: str,
    xlinked_pattern: str,
    segregation: str,
    gene_mode: str,
    llm: "LLMClient",
) -> str:
    """
    MOI Layer 6 — X-linked analysis for one variant.

    Args:
        variant_context:  Per-variant context string from retrieval.
        base_conclusion:  Layer 2's full structured output for this variant.
        xlinked_pattern:  classify_xlinked_ab() result — "XLR" / "XLD" / "uncertain".
        segregation:      classify_segregation() result ("maternal" / "paternal" /
                           "de_novo" / etc.) — used only to gate the mechanical
                           BS2 unaffected-carrier check below.
        gene_mode:        gene_mode_cache entry for the gene (XLD / XLR /
                           XLD_XLR / XL / ...). Gates BS2 by the gene's
                           inheritance, not by the proband's AB pattern.
        llm:               Shared LLMClient instance.

    Returns:
        Structured X-linked layer block, including
        "Total ACMG points: [base+delta] → [classification]".
    """
    logger.info("[MOIXlinked] Generating X-linked analysis for variant (pattern=%s)...", xlinked_pattern)

    template = _load_prompt()
    label = _XLINKED_LABELS.get(xlinked_pattern, _XLINKED_LABELS["uncertain"])

    user_prompt = (template
        .replace("{base_conclusion}", base_conclusion)
        .replace("{xlinked_ab_block}", label))

    result = llm.generate(
        system="You are an expert clinical geneticist evaluating X-linked inheritance. Limit your response to 400 words maximum.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_XLINKED,
    )
    # Base criteria + Total come from code, never from the model: splice the
    # Stage-4 base block, then (after BS2 below) Total = base + delta.
    # PP1 here rests on one transmitting parent: Supporting at most.
    result = cap_layer_pp1(result)
    result = splice_base_and_total(result, base_conclusion)
    # Unaffected-carrier BS2 (DEFAULT-UNAFFECTED POLICY), gated on the gene's
    # mode — chrX genes never reach moi_dominant.py, so this is their only
    # BS2 path. XLD: any carrier parent (het mother, hemizygous father) not
    # stated as affected. Otherwise (XLR / XLD_XLR / XL): only a parent at
    # AB ~1.0 (hemizygous father or homozygous mother) — a heterozygous
    # carrier mother is the expected, uninformative finding for XLR.
    if gene_mode == "XLD":
        result = validate_bs2_unaffected_dominant_carrier(
            result, base_conclusion, segregation, disorder="X-linked dominant")
    elif segregation == "homozygous_parent":
        result = validate_bs2_unaffected_dominant_carrier(
            result, base_conclusion, segregation, disorder="X-linked recessive")
    result = resync_moi_total(result)
    full_context = variant_context + "\n" + base_conclusion
    result = validate_citations(result, full_context)
    return append_clinvar_reference(result, full_context)

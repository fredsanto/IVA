"""
pipeline/stages/cross_analysis.py — Gene-level cross-variant analysis stage.

Runs after per-variant reasoning is complete. For each gene that has two or more
variants in the current run, this stage evaluates potential gene-level interactions
(compound heterozygous, modifier effects, digenic relationships) by presenting the
per-variant contexts and reasonings to the SLM.

Prompt loaded from prompts/cross_analysis.txt.

Public API:
    run(gene, variant_contexts, reasonings, llm) -> str
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "cross_analysis.txt"

MAX_NEW_TOKENS_CROSS_ANALYSIS = 200
_MAX_VARIANT_DATA_CHARS = 40_000



def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Cross-analysis prompt not found at {_PROMPT_PATH}."
    )


def run(
    gene: str,
    variant_contexts: list[str],
    reasonings: list[str],
    llm: "LLMClient",
) -> str:
    """
    Gene-level cross-variant analysis.

    Called only when a gene has two or more variants in the current run.
    Assembles the per-variant contexts and reasonings into a single prompt
    and asks the SLM to evaluate potential compound heterozygous or other
    interactions.

    Args:
        gene:             HGNC gene symbol.
        variant_contexts: Per-variant context strings (one per variant in this gene).
        reasonings:       Per-variant reasoning strings in the same order.
        llm:              Shared LLMClient instance.

    Returns:
        Short cross-analysis text (3–5 sentences) describing any identified
        interaction or noting the absence of meaningful inter-variant relationships.
    """
    logger.info(
        "[CrossAnalysis] Analysing %d variants for gene %s...", len(variant_contexts), gene
    )

    # Build the combined variant data block
    parts = []
    for n, (ctx, reason) in enumerate(zip(variant_contexts, reasonings), start=1):
        parts.append(
            f"--- VARIANT {n} ---\n{ctx}\n\nREASONING:\n{reason}"
        )
    variant_data = "\n\n".join(parts)
    variant_data = variant_data[:_MAX_VARIANT_DATA_CHARS]


    template    = _load_prompt()
    user_prompt = (template
        .replace("{gene}", gene)
        .replace("{variant_data}", variant_data))

    return llm.generate(
        system="You are an expert clinical geneticist.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_CROSS_ANALYSIS,
    )

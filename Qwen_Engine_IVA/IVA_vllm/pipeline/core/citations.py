"""
pipeline/core/citations.py — Anti-hallucination citation validation.

Relocated from pipeline/stages/conclusion.py (Phase 0 of the MOI-layer
restructuring) so every new MOI-layer stage can reuse it instead of each
duplicating its own copy.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def validate_citations(result: str, context: str) -> str:
    """
    Strip inline citations from the model output that have no grounding in context.

    Two citation forms are checked:
      (PMID:12345678)  — removed if that PMID number does not appear anywhere in context.
      (https://...)    — removed if that URL does not appear as a substring in context.

    After removals, stray spaces before punctuation and double spaces are collapsed.
    """
    valid_pmids = set(re.findall(r"PMID:(\d+)", context))

    def _keep_pmid(m: re.Match) -> str:
        if m.group(1) in valid_pmids:
            return m.group(0)
        logger.warning("[Citations] Removed hallucinated citation: %s", m.group(0))
        return ""

    def _keep_url(m: re.Match) -> str:
        if m.group(1) in context:
            return m.group(0)
        logger.warning("[Citations] Removed hallucinated citation: %s", m.group(0))
        return ""

    result = re.sub(r"\(PMID:(\d+)\)", _keep_pmid, result)
    result = re.sub(r"\((https?://[^)\s]+)\)", _keep_url, result)
    result = re.sub(r" +([.,;:])", r"\1", result)
    result = re.sub(r"  +", " ", result)
    return result

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


# A PMID citation wherever it appears — inside or outside parentheses, alone
# or in a list: "PMID:1", "PMID: 1", "PMIDs: 1, 2", "PMID:1; PMID:2". A bare
# list item needs >=5 digits so a following year/count is not taken as a PMID.
_PMID_RE = re.compile(
    r"PMIDs?\s*:?\s*\d+(?:\s*[,;]\s*(?:PMIDs?\s*:?\s*\d+|\d{5,}))*",
    re.IGNORECASE,
)


def validate_citations(result: str, context: str) -> str:
    """
    Strip inline citations from the model output that have no grounding in context.

    Two citation forms are checked:
      PMID             — every PMID number, in or out of parentheses, alone or in
                         a list, is removed if it does not appear as a PMID
                         anywhere in context (context lists like "PMID: 1, 2, 3"
                         count every number).
      (https://...)    — removed if that URL does not appear as a substring in context.

    After removals, emptied parentheses, dangling separators, stray spaces
    before punctuation and double spaces are collapsed.
    """
    valid_pmids = {n for m in _PMID_RE.finditer(context) for n in re.findall(r"\d+", m.group(0))}

    def _keep_pmids(m: re.Match) -> str:
        nums = re.findall(r"\d+", m.group(0))
        kept = [n for n in nums if n in valid_pmids]
        if len(kept) == len(nums):
            return m.group(0)
        for n in nums:
            if n not in valid_pmids:
                logger.warning("[Citations] Removed hallucinated citation: PMID:%s", n)
        return "; ".join(f"PMID:{n}" for n in kept)

    def _keep_url(m: re.Match) -> str:
        if m.group(1) in context:
            return m.group(0)
        logger.warning("[Citations] Removed hallucinated citation: %s", m.group(0))
        return ""

    result = _PMID_RE.sub(_keep_pmids, result)
    result = re.sub(r"\((https?://[^)\s]+)\)", _keep_url, result)
    result = re.sub(r"\(\s*[;,]\s*", "(", result)
    result = re.sub(r"\s*[;,]\s*\)", ")", result)
    result = re.sub(r"([;,])(\s*[;,])+", r"\1", result)
    result = re.sub(r"\(\s*\)", "", result)
    result = re.sub(r" +([.,;:])", r"\1", result)
    result = re.sub(r"  +", " ", result)
    return result

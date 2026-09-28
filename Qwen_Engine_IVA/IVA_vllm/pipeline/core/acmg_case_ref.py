"""
pipeline/core/acmg_case_ref.py — a ClinVar case reference scored as case evidence.

When this variant's CLINVAR status is "P/LP with case reference" — or
"conflicting P/LP/VUS" with more P/LP than VUS submissions, a case-reference
tag and no functional-reference tag — a ClinVar
Pathogenic/Likely pathogenic submission cites literature (a PMID) that is not
functional work, i.e. the variant was reported in affected individuals —
code adds ONE Supporting criterion (+1) to the Stage-4 base criteria, citing
the PMIDs from the tool's "[case reference: P/LP submission citing PMID:X]"
tags. This scores the case observations behind the submission (ClinGen's
replacement for the retired PP5), never the ClinVar classification itself.

Which code depends on the gene's inheritance mode — gene_mode_cache
(moi.build_gene_mode_cache: CSV → CGD → LLM tiers), passed in as data:
  - AD / XLD (dominant only) → PS4 [Supporting, +1]: recurrence in unrelated
    affected individuals.
  - anything else — AR, XLR, AD_AR, XLD_XLR, XL, unknown → PM3 [Supporting,
    +1]: observed in affected individuals under a recessive mechanism. The
    recessive layer's own PM3 for this proband stays a separate delta on
    top, as in ClinGen's PM3 point count.

Robust by construction: the existing criteria are read with find_criteria()
(any layout); nothing is added when the chosen code is already listed, so a
second call is a no-op; the line goes before the "ACMG points" line, else
at the end of the "ACMG criteria" section (its points line, or the next field
header such as "**ACMG classification:**"), else before a points line, else
after the last criterion — and when none exists the text is left unchanged
and logged raw. The total is then
re-summed from the criteria (recompute_and_fix_totals), not nudged.
"""

import logging
import re

from pipeline.core.acmg_points import (
    find_criteria, insert_criterion_line, recompute_and_fix_totals, relabel_all_points_lines,
)
from pipeline.core.clinvar_reference import (
    CLINVAR_CONF_PLP_VUS, CLINVAR_PLP_CASE_REF, clinvar_status_from_context, submission_counts,
)

logger = logging.getLogger(__name__)

_DOMINANT_ONLY_MODES = ("AD", "XLD")
_CASE_TAG_RE = re.compile(r"\[case reference: P/LP submission citing ([^\]]+)\]")
_CITED_PMID_RE = re.compile(r"PMID:?\s*(\d+)")
_FUNCTIONAL_TAG_RE = re.compile(r"\[functional evidence stated in submitter comment, citing ")


def case_ref_code(gene_mode: str) -> str:
    """PS4 for a dominant-only gene, PM3 otherwise (see module docstring)."""
    return "PS4" if (gene_mode or "").upper() in _DOMINANT_ONLY_MODES else "PM3"


def _case_pmids(variant_context: str) -> list[str]:
    out: list[str] = []
    for grp in _CASE_TAG_RE.findall(variant_context):
        for p in _CITED_PMID_RE.findall(grp):
            if p not in out:
                out.append(p)
    return out


def _case_ref_applies(variant_context: str) -> str | None:
    """The CLINVAR status that triggers the case-reference criterion, or None.
    A conflicting P/LP/VUS record counts when P/LP submissions outnumber VUS,
    at least one P/LP submission carries a case-reference tag and none a
    functional one (functional wins, as in clinvar_status())."""
    status = clinvar_status_from_context(variant_context)
    if status == CLINVAR_PLP_CASE_REF:
        return status
    if status != CLINVAR_CONF_PLP_VUS:
        return None
    counts = submission_counts(variant_context)
    plp = counts.get("Pathogenic", 0) + counts.get("Likely pathogenic", 0)
    if plp <= counts.get("VUS", 0) or not _CASE_TAG_RE.search(variant_context) \
            or _FUNCTIONAL_TAG_RE.search(variant_context):
        return None
    return status


def apply_case_reference(conclusion_text: str, variant_context: str, gene_mode: str) -> str:
    """Adds PM3 or PS4 [Supporting, +1] when the ClinVar record qualifies
    (_case_ref_applies) and that code isn't listed yet. See module docstring."""
    status = _case_ref_applies(variant_context)
    if status is None:
        return conclusion_text
    code = case_ref_code(gene_mode)
    if any(c.code == code for c in find_criteria(conclusion_text)):
        return conclusion_text
    cite = ", ".join(f"PMID:{p}" for p in _case_pmids(variant_context)[:3])
    what = ("recurrence in unrelated affected individual(s)" if code == "PS4"
            else "observed in affected individual(s)")
    line = (
        f"- {code} [Supporting, +1]: {what[0].upper() + what[1:]} — ClinVar Pathogenic/Likely "
        f"pathogenic submission citing case literature ({cite}) [auto-added from CLINVAR "
        f"status \"{status}\"; gene inheritance mode: {gene_mode or 'unknown'}].\n"
    )
    text = insert_criterion_line(conclusion_text, line)
    if text is None:
        logger.warning(
            "acmg_case_ref: no ACMG criteria/points section — %s_Supporting not added. Raw conclusion:\n%s",
            code, conclusion_text[:4000],
        )
        return conclusion_text
    return relabel_all_points_lines(recompute_and_fix_totals(text))

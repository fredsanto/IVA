"""
pipeline/core/acmg_pp2_bp1.py — mechanically enforces PP2/BP1's exact-verdict
requirement against the CLINVAR_GENE_STATS gene-level verdict.

A real observed failure: a homozygous missense variant got PP2 applied
with the justification "high count of P/LP missense variants (N)" — but
the gene's actual CLINVAR_GENE_STATS verdict for that run was "balanced"
(N missense vs. M nonsense/frameshift, M well above N), not "missense-predominant". The
model substituted its own plausibility read of the raw counts for the
verdict string prompts/conclusion.txt explicitly requires an exact match
against. This module re-derives the verdict from the same raw counts and the
variant's own consequence type, and strips PP2/BP1 when they don't actually
satisfy both gates (Type=missense AND matching verdict).
"""

import re

from pipeline.core.acmg_points import adjust_points_line
from pipeline.core.protein_change import PROTEIN_CHANGE_RE
from pipeline.tools.clinvar_gene_stats import classify_consequence_counts

_TYPE_RE  = re.compile(r"Type=([^,\n]*)")
_HGVS_RE  = re.compile(r"HGVS=(.*?),\s*Zygosity=")
_COUNTS_RE = re.compile(
    r"P/LP missense variants\s*:\s*(\d+)\s*\n\s*P/LP nonsense/frameshift\s*:\s*(\d+)"
)

_PP2_LINE_RE = re.compile(r"^-\s*PP2\b.*$\n?", re.MULTILINE)
_BP1_LINE_RE = re.compile(r"^-\s*BP1\b.*$\n?", re.MULTILINE)

_FRAMESHIFT_MARKERS = ("frameshift", "indel", "insertion", "deletion", "dup")
_NONSENSE_MARKERS   = ("nonsense", "stop")
_SPLICE_MARKERS     = ("splice",)
_SYNONYMOUS_MARKERS = ("synonymous",)


def _consequence_class(variant_type: str, hgvs: str) -> str:
    """Best-effort classification: 'missense' | 'nonsense' | 'frameshift' |
    'synonymous' | 'splice' | 'other'. Type field alone is unreliable — it's
    often just 'SNV' even for missense variants — so this also parses the
    HGVS protein-change notation as a fallback/cross-check."""
    t = variant_type.strip().lower()
    if any(marker in t for marker in _FRAMESHIFT_MARKERS) or "fs" in hgvs.lower():
        return "frameshift"
    if any(marker in t for marker in _NONSENSE_MARKERS):
        return "nonsense"
    if any(marker in t for marker in _SPLICE_MARKERS):
        return "splice"
    if any(marker in t for marker in _SYNONYMOUS_MARKERS):
        return "synonymous"

    m = PROTEIN_CHANGE_RE.search(hgvs)
    if m:
        aa1, _pos, aa2 = m.groups()
        if aa2.upper() in ("TER", "*", "X"):
            return "nonsense"
        if aa1.upper() == aa2.upper():
            return "synonymous"
        return "missense"

    return "other"


def validate_pp2_bp1(conclusion_text: str, variant_context: str) -> str:
    """
    Strips PP2 unless (this variant's own consequence is missense) AND (the
    CLINVAR_GENE_STATS verdict, re-derived from its raw counts, is exactly
    "missense-predominant"). Strips BP1 the same way against
    "nonsense/frameshift-predominant". Adjusts the stated ACMG points/
    classification to match. No-op if neither line is present, or both are
    already properly grounded.
    """
    has_pp2 = bool(_PP2_LINE_RE.search(conclusion_text))
    has_bp1 = bool(_BP1_LINE_RE.search(conclusion_text))
    if not has_pp2 and not has_bp1:
        return conclusion_text

    type_m = _TYPE_RE.search(variant_context)
    hgvs_m = _HGVS_RE.search(variant_context)
    variant_type = type_m.group(1) if type_m else ""
    hgvs         = hgvs_m.group(1) if hgvs_m else ""
    consequence  = _consequence_class(variant_type, hgvs)

    counts_m = _COUNTS_RE.search(variant_context)
    if counts_m:
        missense, nonsense = int(counts_m.group(1)), int(counts_m.group(2))
        verdict = classify_consequence_counts(missense, nonsense)
    else:
        verdict = None  # no gene-level counts evidence at all

    text = conclusion_text

    if has_pp2 and not (consequence == "missense" and verdict == "missense-predominant"):
        text = _PP2_LINE_RE.sub("", text, count=1)
        text = adjust_points_line(text, delta=-1)

    if has_bp1 and not (consequence == "missense" and verdict == "nonsense/frameshift-predominant"):
        text = _BP1_LINE_RE.sub("", text, count=1)
        text = adjust_points_line(text, delta=+1)

    return text

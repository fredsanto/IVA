"""
pipeline/core/acmg_pm2.py — grounds PM2 in the variant's allele frequency:
  - "GNOMAD VARIANT FREQUENCY" block (pipeline/tools/gnomad_frequency.py,
    live lookup run when the input CSV's Frequency is blank): "Not found in
    gnomAD" counts as absent; otherwise the highest genome/exome AF is used.
  - else the input CSV's own "Frequency=" field on the VARIANT line.

Add-only: PM2 [Moderate, +2] is added when that AF is below PM2_AF_THRESHOLD
(or the variant is absent from gnomAD) and the model's list has none. A
model-written PM2 is never stripped — on the full 327-report corpus a strip
at >= 1e-4 removed PM2 from 50 variants (40 in AR genes at AF 2-3e-4, rare
enough for a recessive disease), turning 4 LP into VUS.
"""

import re

from pipeline.core.acmg_points import adjust_points_line, find_criteria, insert_criterion_line

PM2_AF_THRESHOLD = 1e-4

_BLOCK_RE = re.compile(r"GNOMAD VARIANT FREQUENCY \(([^)]*)\):\n(.*?)(?:\n\s*\n|\Z)", re.DOTALL)
_BLOCK_AF_RE = re.compile(r"^\s*(Genome|Exome) AF = ([0-9.eE+-]+)", re.MULTILINE)
_CSV_FREQ_RE = re.compile(r"(?:^|[,\s])Frequency=([^,\s]*)")


def _frequency(variant_context: str) -> tuple[float, str] | None:
    """(AF, source description) for this variant; AF 0.0 when absent from gnomAD."""
    bm = _BLOCK_RE.search(variant_context)
    if bm:
        body = bm.group(2)
        if "Not found in gnomAD" in body:
            return 0.0, f"not found in gnomAD ({bm.group(1)})"
        afs = [(float(v), label) for label, v in _BLOCK_AF_RE.findall(body)]
        if afs:
            af, label = max(afs)
            return af, f"gnomAD {label} AF={af:.6g} ({bm.group(1)})"
    cm = _CSV_FREQ_RE.search(variant_context)
    if cm:
        try:
            af = float(cm.group(1))
        except ValueError:
            return None
        return af, f"Frequency={cm.group(1)} in the variant data"
    return None


def validate_pm2(conclusion_text: str, variant_context: str) -> str:
    """
    Adds PM2 [Moderate, +2] when the variant's AF < PM2_AF_THRESHOLD and the
    model's list has no PM2, adjusting the stated points. Never strips.
    """
    freq = _frequency(variant_context)
    if freq is None or freq[0] >= PM2_AF_THRESHOLD:
        return conclusion_text
    source = freq[1]
    if any(c.code == "PM2" for c in find_criteria(conclusion_text)):
        return conclusion_text
    line = (f"- PM2 [Moderate, +2]: Absent from controls or extremely rare — {source}, "
            f"below the {PM2_AF_THRESHOLD:g} threshold [auto-added — evidence supported this "
            "but it was missing from the model's own criteria list].\n")
    text = insert_criterion_line(conclusion_text, line)
    if text is None:
        return conclusion_text
    return adjust_points_line(text, 2)

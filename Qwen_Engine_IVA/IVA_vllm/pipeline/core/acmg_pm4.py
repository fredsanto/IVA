"""
pipeline/core/acmg_pm4.py — PM4 (protein length change) from the
"PM4 REPEAT CHECK" verdict (pipeline/tools/repeat_region.py).

PM4 applies to in-frame deletions/insertions outside a repeat region and to
stop-loss variants. It was previously never applied: no prompt or validator
defined it, so every in-frame indel lost its +2.

  - verdict MET and no PVS1 in the list -> PM4 [Moderate, +2] added if missing
  - otherwise (NOT MET, no verdict, UCSC lookup failed, or PVS1 applied)
    -> every PM4 line is stripped. PM4 never stacks on PVS1: a null variant's
    length change is what PVS1 already scores.
"""

import re

from pipeline.core.acmg_points import adjust_points_line, find_criteria, insert_criterion_line

_VERDICT_RE = re.compile(r"PM4 REPEAT CHECK:\s*(MET|NOT MET)\s*—\s*([^\n]*)")
_PM4_LINE_RE = re.compile(r"^-\s*PM4(?:_\w+)?\b.*\n?", re.MULTILINE)


def validate_pm4(conclusion_text: str, variant_context: str) -> str:
    """
    Adds PM4 [Moderate, +2] when the repeat check is MET and the list has no
    PVS1 or PM4; strips PM4 otherwise.

    Stripping does NOT adjust stated points — callers must follow with
    acmg_points.recompute_and_fix_totals(), same as validate_pm1.
    """
    m = _VERDICT_RE.search(variant_context)
    criteria = find_criteria(conclusion_text)
    if not m or m.group(1) != "MET" or any(c.code == "PVS1" for c in criteria):
        return _PM4_LINE_RE.sub("", conclusion_text)
    if any(c.code == "PM4" for c in criteria):
        return conclusion_text
    line = (f"- PM4 [Moderate, +2]: Protein length change — {m.group(2).strip()} "
            "(PM4 REPEAT CHECK) [auto-added].\n")
    text = insert_criterion_line(conclusion_text, line)
    if text is None:
        return conclusion_text
    return adjust_points_line(text, 2)

"""
pipeline/core/acmg_bp6.py — mechanically enforces BP6's classification gate
against a genuine observed failure: the model writing a BP6 line that
CONTRADICTS ITSELF in the same sentence.

Real case: "BP6 Supporting (-1 pts): ClinVar classifies this specific
variant as Uncertain significance (VUS), which does not meet the threshold
for BP6 (requires Likely benign or Benign)." — the model correctly recites
its own gate, states the evidence fails it, and applies the criterion anyway.

Rather than try to re-derive ClinVar classification from evidence text (which
varies by source — CSV ClinVar_class, live ClinGen/ClinVar lookups, etc.),
this checks the BP6 line's OWN justification text for a disqualifying
classification term. A genuinely-grounded BP6 line should only ever mention
Benign/Likely benign — if it also says "uncertain"/"pathogenic"/
"conflicting", that's the model documenting its own gate failure.
"""

import re

from pipeline.core.acmg_points import adjust_points_line

_BP6_LINE_RE = re.compile(r"^-\s*BP6\b.*$", re.MULTILINE)

_DISQUALIFYING_TERMS = ("uncertain", "vus", "pathogenic", "conflicting")


def validate_bp6(conclusion_text: str) -> str:
    """
    Strips a BP6 line whose own justification text names a disqualifying
    classification (uncertain significance/VUS, pathogenic, conflicting)
    instead of Benign/Likely benign — the model contradicting its own
    stated gate. Adjusts the stated ACMG points/classification to match.
    """
    m = _BP6_LINE_RE.search(conclusion_text)
    if not m:
        return conclusion_text

    line_lower = m.group(0).lower()
    if not any(term in line_lower for term in _DISQUALIFYING_TERMS):
        return conclusion_text  # no contradiction found — leave as-is

    text = _BP6_LINE_RE.sub("", conclusion_text, count=1)
    return adjust_points_line(text, delta=+1)  # removing a -1 item adds 1 back

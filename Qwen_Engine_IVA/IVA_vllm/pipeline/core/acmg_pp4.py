"""
pipeline/core/acmg_pp4.py — mechanically enforces PP4's single-gene-etiology
gate for well-known genetically heterogeneous conditions, and its FULL
COVERAGE gate against the backend-computed phenotype-cluster verdict.

Real observed failure, TWICE, even after prompts/conclusion.txt was given an
explicit worked negative example naming this exact condition: PP4 applied
for a GENE_X variant in "condition Y" ("a condition with a known genetic
etiology in GENE_X") without ever addressing that condition Y is also
independently caused by two other genes — a textbook heterogeneous condition.
Since prompt-only guidance repeatedly failed on this exact case, this module
mechanically strips PP4 when its own justification names a condition on a
small maintained list of well-known heterogeneous Mendelian conditions,
regardless of how phenotype-specific the match looks.
"""

import re

from pipeline.core.acmg_points import adjust_points_line

_PP4_LINE_RE = re.compile(r"^-\s*PP4\b.*$", re.MULTILINE)

# Matches the backend-injected "--- CLUSTER_PHENOTYPE (...) ---\nCLUSTER_PHENOTYPE: VERDICT"
# block pipeline.py's _cluster_match_block() appends to the reasoning text
# handed to conclusion.run_one() — distinct from, and not parseable by,
# reasoning.parse_cluster_match()'s "CLUSTER_PHENOTYPE: VERDICT"
# one-line format, which only matches Stage 1's own raw reasoning output.
_INJECTED_CLUSTER_MATCH_RE = re.compile(
    r"---\s*CLUSTER_PHENOTYPE\b.*?CLUSTER_PHENOTYPE:\s*(YES|PARTIAL|NONE|INCIDENTAL|UNKNOWN)\b",
    re.IGNORECASE | re.DOTALL,
)


def _parse_injected_cluster_match(reasoning_text: str) -> str:
    m = _INJECTED_CLUSTER_MATCH_RE.search(reasoning_text)
    return m.group(1).upper() if m else "UNKNOWN"


def validate_pp4_full_coverage(conclusion_text: str, reasoning_text: str) -> str:
    """
    Mechanically enforces PP4 condition (3), FULL COVERAGE, from
    prompts/conclusion.txt's own PP4 rule: PP4 may only be applied when the
    backend-computed CLUSTER_PHENOTYPE verdict is YES — a PARTIAL,
    NONE, INCIDENTAL, or UNKNOWN verdict means the phenotype match is, by
    definition, not "highly specific" for what this patient actually has, no
    matter how hallmark-like the matched feature is on its own (INCIDENTAL in
    particular means the match is to a DIFFERENT disease entirely).

    Real observed failure: a base conclusion applied PP4 ("a core feature of
    the gene's associated overgrowth syndrome") in the very same sentence
    admitting the gene "does not explain" several of the patient's other
    findings — a PARTIAL match by its own words — while a different gene in
    the SAME pipeline run correctly withheld PP4 for an identical
    partial-match situation ("failing the 'Full coverage' requirement for
    PP4"). Prompt-only
    guidance is not reliably followed for this condition even when the
    correct behavior is demonstrated elsewhere in the same run; strip PP4
    mechanically whenever the backend's own verdict — not the model's own
    restatement or judgment of it — is not YES. No-op if no PP4 line, or the
    backend verdict is YES.
    """
    m = _PP4_LINE_RE.search(conclusion_text)
    if not m:
        return conclusion_text

    if _parse_injected_cluster_match(reasoning_text) == "YES":
        return conclusion_text

    text = _PP4_LINE_RE.sub("", conclusion_text, count=1)
    return adjust_points_line(text, delta=-1)

# Well-known genetically heterogeneous conditions (disease name substring,
# case-insensitive) that keep tripping up PP4's single-gene-etiology gate.
# Extend as new recurring cases are observed.
_HETEROGENEOUS_CONDITIONS = (
    "familial hypercholesterolemia",
)


def validate_pp4(conclusion_text: str) -> str:
    """
    Strips a PP4 line whose own justification names a condition known to be
    genetically heterogeneous (multiple independent causal genes) — PP4's
    single-gene-etiology condition cannot be satisfied for these regardless
    of how phenotype-specific the match looks. Adjusts the stated ACMG
    points/classification to match. No-op if no PP4 line, or its named
    condition isn't on the known-heterogeneous list.
    """
    m = _PP4_LINE_RE.search(conclusion_text)
    if not m:
        return conclusion_text

    line_lower = m.group(0).lower()
    if not any(cond in line_lower for cond in _HETEROGENEOUS_CONDITIONS):
        return conclusion_text

    text = _PP4_LINE_RE.sub("", conclusion_text, count=1)
    return adjust_points_line(text, delta=-1)

"""
pipeline/core/acmg_ps3.py — deterministic PS3 enforcement from two sources.

PS3 is applied ONLY from one of these tool-side tags:
  - "[functional evidence stated in submitter comment, citing PMID:X, ...]"
    (clinvar_gene_stats.py / ncbi.py) — a ClinVar P/LP submitter's Comment
    reports functional/experimental work (not in-silico, not negated) and the
    submission carries a literature reference for it. The paper itself is
    not re-checked: the submitter's statement plus its reference is the
    evidence. A P/LP submission without such a reference never gives PS3
    (it is reported as CLINVAR "P/LP with case reference" when it cites
    a PMID — scored as PM3/PS4_Supporting by acmg_case_ref.py — "P/LP w/o
    reference" otherwise).
  - "[variant-level functional evidence reported by LitVar2, citing PMID:X]"
    (litvar2.py, variant-level rsID track only) — the summary states
    functional work on THIS variant and cites a paper that track retrieved.
    Gene-level literature never counts.

surface_ps3_citations() then guarantees those PMIDs reach the FINAL REPORT:
the MOI layer blocks and the Clinical Conclusion are re-written by the SLM
downstream and routinely drop them.

validate_ps3():
  - keeps a model-written PS3 line only if it cites at least one tagged PMID;
  - otherwise strips it and subtracts its stated points;
  - adds a PS3 line citing the tagged PMIDs when the model omitted it
    (or wrote an ungrounded one) and tagged PMIDs exist.
"""

import re

from pipeline.core.acmg_points import adjust_points_line

_PS3_LINE_RE = re.compile(r"^-\s*PS3\b.*(?:\n|$)", re.MULTILINE)
_CRITERIA_INSERT_RE = re.compile(r"(?=\*\*ACMG points:\*\*)")
_CLINVAR_TAG_RE = re.compile(
    r"\[functional evidence stated in submitter comment, citing ([^\]]+)\]"
)
_LITVAR2_TAG_RE = re.compile(
    r"\[variant-level functional evidence reported by LitVar2, citing ([^\]]+)\]"
)
_CITED_PMID_RE = re.compile(r"PMID:?\s*(\d+)")
_LINE_POINTS_RE = re.compile(r"\+\s*(\d+(?:\.\d+)?)")


def _confirmed_pmids(variant_context: str) -> tuple[list[str], list[str]]:
    """(ClinVar-submitter-cited PMIDs, LitVar2 variant-level PMIDs), each deduped."""
    def _collect(rx: re.Pattern, skip: list[str]) -> list[str]:
        out: list[str] = []
        for grp in rx.findall(variant_context):
            for p in _CITED_PMID_RE.findall(grp):
                if p not in out and p not in skip:
                    out.append(p)
        return out
    clinvar = _collect(_CLINVAR_TAG_RE, [])
    return clinvar, _collect(_LITVAR2_TAG_RE, clinvar)


def validate_ps3(conclusion_text: str, variant_context: str) -> str:
    """
    Enforces tag-grounded PS3 on the model's criteria list and adjusts
    the stated points/classification to match. See module docstring.
    """
    clinvar_pmids, litvar2_pmids = _confirmed_pmids(variant_context)
    confirmed = clinvar_pmids + litvar2_pmids
    confirmed_set = set(confirmed)

    text = conclusion_text
    grounded = False
    for m in reversed(list(_PS3_LINE_RE.finditer(text))):
        line = m.group(0)
        if confirmed_set & set(_CITED_PMID_RE.findall(line)):
            grounded = True
            continue
        pts_m = _LINE_POINTS_RE.search(line)
        pts = float(pts_m.group(1)) if pts_m else 4.0
        text = text[:m.start()] + text[m.end():]
        text = adjust_points_line(text, delta=-pts)

    if grounded or not confirmed:
        return text

    sources = []
    if clinvar_pmids:
        sources.append("reported by a ClinVar P/LP submitter citing "
                       + ", ".join(f"PMID:{p}" for p in clinvar_pmids[:3]))
    if litvar2_pmids:
        sources.append("reported for this variant in the literature ("
                       + ", ".join(f"PMID:{p}" for p in litvar2_pmids[:3]) + ")")
    line = (
        f"- PS3 [Strong, +4]: Well-established in vitro/in vivo functional evidence "
        f"supports a damaging effect, {'; '.join(sources)} "
        "[auto-added — evidence supported this but it was missing from the model's "
        "own criteria list].\n"
    )
    text = _CRITERIA_INSERT_RE.sub(line, text, count=1)
    return adjust_points_line(text, delta=4)


# A PS3 criterion bullet in any downstream format: "- PS3 [Strong, +4]: ...",
# "- PS3 (Strong, +4): ...", "*   **PS3** (Strong, +4 pts): ...". Prose
# mentions ("...functional evidence (PS3)...") are not bullets and are left alone.
_PS3_BULLET_RE = re.compile(r"(?m)^[ \t]*[-*][ \t]*(?:\*\*)?PS3\b(?:\*\*)?[^\n]*$")
_BASE_GENE_RE = re.compile(r"(?m)^#\s*Variant\s+\d+\s*[—-]\s*(\S+)")


def surface_ps3_citations(final_report: str, base_conclusions: list[str]) -> str:
    """
    Appends the grounding PMID(s) to every PS3 bullet in the FINAL REPORT
    that doesn't already cite a PMID. Sources are the Stage-4 base
    conclusions, whose PS3 lines validate_ps3() already guaranteed cite a
    literature PMID. The gene is resolved from the nearest mention (on or before the line)
    of a source gene in the report (or the sole source, if only one variant
    carries PS3). No-op when no base conclusion carries PS3.
    """
    sources: dict[str, list[str]] = {}
    for base in base_conclusions:
        m = _PS3_LINE_RE.search(base)
        g = _BASE_GENE_RE.search(base)
        if not m or not g:
            continue
        pmids = sources.setdefault(g.group(1), [])
        for p in _CITED_PMID_RE.findall(m.group(0)):
            if p not in pmids:
                pmids.append(p)
    sources = {g: p for g, p in sources.items() if p}
    if not sources:
        return final_report

    all_pmids = {p for pm in sources.values() for p in pm}
    gene_res = {g: re.compile(rf"\b{re.escape(g)}\b") for g in sources}

    def _fix(m: re.Match) -> str:
        line = m.group(0)
        if set(_CITED_PMID_RE.findall(line)) & all_pmids:
            return line
        if len(sources) == 1:
            gene = next(iter(sources))
        else:
            before = final_report[:m.end()]
            last = {g: max((x.start() for x in r.finditer(before)), default=-1)
                    for g, r in gene_res.items()}
            gene = max(last, key=last.get)
            if last[gene] < 0:
                return line
        cite = ", ".join(f"PMID:{p}" for p in sources[gene][:3])
        return f"{line} [PS3 literature: {cite}]"

    return _PS3_BULLET_RE.sub(_fix, final_report)


_PVS1_PRESENT_RE = re.compile(r"(?m)^-\s*PVS1\b")


def block_ps3_under_pvs1(conclusion_text: str) -> str:
    """
    Strips PS3 when PVS1 applies (at any strength) — the functional study of
    a null variant shows the loss of function PVS1 already scores, so PS3
    would double-count it (ClinGen SVI). Same treatment as PS1/PM5 on
    non-missense variants. Must run AFTER validate_pvs1() (so PVS1's own
    strip/cap is final) and BEFORE recompute_and_fix_totals(), which re-sums
    the totals from the remaining bullets.
    """
    if not _PVS1_PRESENT_RE.search(conclusion_text):
        return conclusion_text
    return _PS3_LINE_RE.sub("", conclusion_text)

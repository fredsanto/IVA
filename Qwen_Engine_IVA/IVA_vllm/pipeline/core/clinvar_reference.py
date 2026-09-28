"""
pipeline/core/clinvar_reference.py — mechanically appends this variant's own
ClinVar reference (variation ID + URL) to a MOI-layer/final-report block.

The base conclusion stage's "**ClinVar:**" field is prose (classification +
condition), not a clickable reference, and MOI-layer stages only copy the
"ACMG criteria:" list verbatim from it (see prompts/moi_*.txt) — the ClinVar
ID itself is never guaranteed to reach the final printed report. Rather than
ask yet another SLM call to remember to carry it forward (the exact failure
mode this session repeatedly ran into for other fields), this reads the
variation ID directly out of the retrieval-stage evidence text — ClinGenAlleleTool's
"ClinVar variation ID: NNNN" line (this variant's own cross-referenced allele)
or, failing that, ClinVarGeneStatsTool's submission-tally header — and appends
a plain reference line deterministically.

The line also carries this variant's CLINVAR status ("CLINVAR: <value>", one
of CLINVAR_STATUSES below) read from the tool's own "CLINVAR
STATUS:" line, so every MOI/unclassified block the final report is written
from states it, and add_clinvar_status_to_findings() can print it next to
every finding in the final report's sections 2-4.
"""

import re


# ── per-variant CLINVAR status ────────────────────────────────────────────────
CLINVAR_PLP_FUNC_REF = "P/LP with functional reference"
CLINVAR_PLP_CASE_REF = "P/LP with case reference"
CLINVAR_PLP_NO_REF   = "P/LP w/o reference"
CLINVAR_VUS          = "VUS"
CLINVAR_LB_B         = "LB/B"
CLINVAR_UNDEFINED    = "undefined"
CLINVAR_CONF_PLP_VUS = "conflicting P/LP/VUS"
CLINVAR_CONF_VUS_B   = "conflicting VUS/LB/B"
CLINVAR_STATUSES = (
    CLINVAR_PLP_FUNC_REF, CLINVAR_PLP_CASE_REF, CLINVAR_PLP_NO_REF, CLINVAR_VUS, CLINVAR_LB_B,
    CLINVAR_UNDEFINED, CLINVAR_CONF_PLP_VUS, CLINVAR_CONF_VUS_B,
)


def clinvar_status(counts: dict[str, int] | None, has_functional_ref: bool,
                   has_case_ref: bool = False) -> str:
    """
    Per-variant CLINVAR status from the per-submission tally buckets.
    "with functional reference" = at least one P/LP submission reports
    functional work with a literature reference (the PS3 trigger). "with case
    reference" = otherwise, at least one P/LP submission cites a PMID — the
    literature it rests on is case observations (the PM3/PS4_Supporting trigger,
    acmg_case_ref.py). Functional wins when both apply. Pathogenic and Likely
    pathogenic together are NOT a conflict. A record carrying both P/LP and
    LB/B submissions goes to whichever side has more submissions (ties to
    the P/LP side, so a possible pathogenic call is never under-reported).
    No record, a failed fetch, or no classified submission -> "undefined".
    """
    counts = counts or {}
    plp = counts.get("Pathogenic", 0) + counts.get("Likely pathogenic", 0)
    vus = counts.get("VUS", 0)
    blb = counts.get("Likely benign", 0) + counts.get("Benign", 0)
    if not (plp or vus or blb):
        return CLINVAR_UNDEFINED
    if plp and not vus and not blb:
        if has_functional_ref:
            return CLINVAR_PLP_FUNC_REF
        return CLINVAR_PLP_CASE_REF if has_case_ref else CLINVAR_PLP_NO_REF
    if vus and not plp and not blb:
        return CLINVAR_VUS
    if blb and not plp and not vus:
        return CLINVAR_LB_B
    if plp and plp >= blb:
        return CLINVAR_CONF_PLP_VUS
    return CLINVAR_CONF_VUS_B


_CLINGEN_ID_RE = re.compile(r"ClinVar variation ID:\s*(\d+)")
_TALLY_ID_RE   = re.compile(r"CLINVAR VARIANT-LEVEL SUBMISSION TALLY \(variation ID (\d+)")

_REFERENCE_MARKER = "**ClinVar reference:**"
_STATUS_RE = re.compile(
    r"CLINVAR STATUS:\s*(" + "|".join(re.escape(v) for v in CLINVAR_STATUSES) + r")"
)


def clinvar_status_from_context(evidence_context: str) -> str:
    """This variant's CLINVAR status from its retrieval evidence; "undefined"
    when the ClinVar tool produced no status line (not run / failed)."""
    m = _STATUS_RE.search(evidence_context or "")
    return m.group(1) if m else CLINVAR_UNDEFINED

# ClinVarGeneStatsTool.run() prints exactly one line per bucket, in this
# order, formatted as f"{label:<18}: {count}" (see its _TALLY_ORDER /
# variant_block construction) — e.g. "Pathogenic        : 46". Matched
# per-line rather than as one block regex so it doesn't care about the
# gene-level counts / conflicting-evidence text that may precede or follow
# it in the same raw_output string.
_TALLY_LINE_RE = re.compile(
    r"^(Pathogenic|Likely pathogenic|VUS|Likely benign|Benign)\s*:\s*(\d+)\s*$",
    re.MULTILINE,
)


def submission_counts(evidence_context: str | None) -> dict[str, int]:
    """Per-bucket submission counts from this variant's CLINVAR VARIANT-LEVEL
    SUBMISSION TALLY lines; {} when no tally is present."""
    return {m.group(1): int(m.group(2)) for m in _TALLY_LINE_RE.finditer(evidence_context or "")}


def resolve_variant_clinvar_status(clinvar_gene_stats_raw: str | None) -> str | None:
    """
    Deterministically resolve this variant's own ClinVar clinical
    significance from ClinVarGeneStatsTool's already-fetched "CLINVAR
    VARIANT-LEVEL SUBMISSION TALLY" block — real per-submission ClinVar
    data (NCBI efetch VCV record), not an LLM guess and not dependent on
    the input CSV's own ClinVar_class column (which can be missing or
    mismapped by the header interpreter — see acmg_sf.build_actionable_set,
    which uses this as a fallback ground truth for ACMG SF actionable-gene
    detection when that column is unusable).

    Returns one of "Pathogenic", "Likely pathogenic", "VUS", "Likely
    benign", "Benign" (the single bucket with submissions, when exactly one
    bucket is non-zero), "Conflicting interpretations of pathogenicity"
    (more than one bucket has submissions — ClinVar's own definition of
    conflicting), or None (no tally block present, the fetch failed, or
    ClinVar has no resolvable record for this specific variant — all of
    which must be treated as "unknown," never silently as "not pathogenic").
    """
    if not clinvar_gene_stats_raw:
        return None
    counts = {
        m.group(1): int(m.group(2))
        for m in _TALLY_LINE_RE.finditer(clinvar_gene_stats_raw)
    }
    nonzero = {bucket: n for bucket, n in counts.items() if n > 0}
    if not nonzero:
        return None
    if len(nonzero) > 1:
        return "Conflicting interpretations of pathogenicity"
    return next(iter(nonzero))


def _extract_variation_id(evidence_context: str) -> str | None:
    m = _CLINGEN_ID_RE.search(evidence_context)
    if m:
        return m.group(1)
    m = _TALLY_ID_RE.search(evidence_context)
    if m:
        return m.group(1)
    return None


def append_clinvar_reference(block: str, evidence_context: str) -> str:
    """
    Appends "**ClinVar reference:** <url>" (or an explicit "not found" note)
    to `block`, unless it already contains a ClinVar reference line. Always
    states one or the other — silence here reads as "not checked", which is
    worse than an explicit negative.
    """
    if _REFERENCE_MARKER in block:
        return block  # already present (MOI prompt happened to carry it forward)

    return block.rstrip() + "\n\n" + _reference_line(evidence_context)


def append_clinvar_references(block: str, labeled_contexts: list[tuple[str, str]]) -> str:
    """
    Same as append_clinvar_reference, for MOI blocks covering MULTIPLE
    variants (e.g. a compound-het pair) — one reference line per
    (label, evidence_context) entry, each resolved independently so a
    variant with no match doesn't shadow one that does.
    """
    if _REFERENCE_MARKER in block:
        return block

    lines = [f"{_REFERENCE_MARKER} {label}: {_reference_line(ctx, bare=True)}" for label, ctx in labeled_contexts]
    return block.rstrip() + "\n\n" + "\n".join(lines)


def _reference_line(evidence_context: str, bare: bool = False) -> str:
    """`bare=True` omits the leading marker (caller supplies its own prefix)."""
    variation_id = _extract_variation_id(evidence_context)
    prefix = "" if bare else f"{_REFERENCE_MARKER} "
    status = clinvar_status_from_context(evidence_context)
    if status != CLINVAR_UNDEFINED:  # undefined -> print no status at all
        prefix += f"CLINVAR: {status} — "
    if variation_id:
        # Not prefixed "VCV{id}" — the zero-padded VCV accession format isn't
        # reconstructible from the bare numeric ID alone; the URL form is
        # unambiguous and always correct regardless of accession formatting.
        return (
            f"{prefix}ClinVar Variation ID {variation_id} — "
            f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{variation_id}/"
        )
    return f"{prefix}Not found in ClinVar (novel/unresolvable allele)."


# Final-report section headers "2) Causative ...", "3) Actionable ...",
# "4) Notable VUS ...", "5) Summary ..." (optionally bolded/indented, or as a
# markdown heading "## 2) ...").
_SECTION_RE = re.compile(r"(?m)^[ \t#*]*([1-9])\)")
_UNDEFINED_STATUS_RE = re.compile(r"[ \t]*[—-]?[ \t]*CLINVAR:[ \t]*" + re.escape(CLINVAR_UNDEFINED) + r"\b")
_CDNA_RE = re.compile(r"c\.[^\s:;|(),]+")


def add_clinvar_status_to_findings(final_summary: str,
                                   variants: list[tuple[str, str, str]]) -> str:
    """
    Appends " — CLINVAR: <status>" to every finding line in sections 2
    (causative, over threshold), 3 (actionable) and 4 (notable VUS) of the
    Clinical Conclusion that doesn't already state it. An "undefined" status
    is never printed, and any "CLINVAR: undefined" already in the text is
    removed.

    variants: (gene, hgvs, status) for every variant that reached the
    conclusion stage. A finding line is one that starts (after bullet/bold
    markers) with one of those gene symbols. When several variants share the
    gene, the line's own c. notation picks the variant; if it can't, every
    candidate's status is listed with its c. notation.
    """
    by_gene: dict[str, list[tuple[str, str]]] = {}
    for gene, hgvs, status in variants:
        if gene and gene != "NA":
            by_gene.setdefault(gene, []).append((hgvs or "", status))
    if not by_gene:
        return _UNDEFINED_STATUS_RE.sub("", final_summary)
    gene_alt = "|".join(re.escape(g) for g in sorted(by_gene, key=len, reverse=True))
    entry_re = re.compile(rf"^[ \t]*(?:[-*•][ \t]+)?\**[ \t]*\*?({gene_alt})\b")

    def _status_for(gene: str, line: str) -> str:
        cands = by_gene[gene]
        if len(cands) > 1:
            line_cdnas = set(_CDNA_RE.findall(line))
            hit = [c for c in cands if line_cdnas & set(_CDNA_RE.findall(c[0]))]
            if len(hit) == 1:
                cands = hit
        cands = [(h, s) for h, s in cands if s != CLINVAR_UNDEFINED]
        if not cands:
            return ""
        if len({s for _, s in cands}) == 1:
            return cands[0][1]
        return "; ".join(
            f"{(_CDNA_RE.findall(h) or [h])[0]} {s}" for h, s in cands
        )

    out, section = [], None
    for line in final_summary.split("\n"):
        line = _UNDEFINED_STATUS_RE.sub("", line)
        sm = _SECTION_RE.match(line)
        if sm:
            section = sm.group(1)
        em = entry_re.match(line) if section in ("2", "3", "4") else None
        if em and "CLINVAR:" not in line:
            status = _status_for(em.group(1), line)
            if status:
                line = f"{line.rstrip()} — CLINVAR: {status}"
        out.append(line)
    return "\n".join(out)

"""
pipeline/core/acmg_ps1_pm5.py — mechanically ADDS a PS1/PM5 criterion the SLM
omitted despite the deterministic CLINVAR_RESIDUE_SEARCH evidence supporting it.

Real observed failure: the SAME evidence block (containing both a "SAME amino
acid change ... Likely pathogenic" hit and a "DIFFERENT amino acid change ...
Likely pathogenic" hit at the same residue) led the model to write only ONE
of PS1/PM5 into its criteria list — and inconsistently which one across
otherwise-identical runs — despite prompts/conclusion.txt explicitly stating
both may apply from separate hits. This module re-derives PS1/PM5
eligibility directly from the tool's own structured tags (SAME/DIFFERENT
amino acid change, [PATHOGENIC/LIKELY PATHOGENIC] flag, Variation ID) and
ADDS whichever criterion the evidence supports but the model's own list is
missing — the mirror image of acmg_pp3.py / acmg_pp2_bp1.py, which STRIP
wrongly-applied criteria; this one fills in under-applied ones.
"""

import re

from pipeline.core.acmg_points import adjust_points_line
from pipeline.core.acmg_sf import is_pathogenic_clinvar
from pipeline.core.protein_change import PROTEIN_CHANGE_RE, _to_aa3

# Matches the CLINVAR RESIDUE-LEVEL SEARCH block up to the next blank-line +
# capitalized section header, or end of string.
_RESIDUE_BLOCK_RE = re.compile(r"CLINVAR RESIDUE-LEVEL SEARCH.*?(?=\n\n[A-Z]|\Z)", re.DOTALL)

# One hit line, e.g.:
#   - c.100A>C (p.Lys34Thr): Likely pathogenic (...) — SAME amino acid
#     change as this variant [PATHOGENIC/LIKELY PATHOGENIC] — ClinVar
#     Variation ID 123456 (https://.../123456/)
_HIT_RE = re.compile(
    r"-\s*(?P<nt>c\.\S+)\s*\(p\.(?P<protein>[A-Za-z*]+\d+[A-Za-z*]+)\):\s*"
    r"(?P<classification>[^—\n]+?)\s*—\s*(?P<tag>SAME|DIFFERENT) amino acid change"
    r"[^\n]*?ClinVar Variation ID (?P<vid>\d+)"
)

# This variant's own cDNA token(s) — scoped to the HGVS= field specifically,
# NOT run over the whole variant_context blob (which also contains the
# CLINVAR RESIDUE-LEVEL SEARCH block's OTHER variants' own cDNA notations —
# extracting over the full text would misclassify every hit as "this
# variant's own record" and silently find nothing). Same multi-transcript-
# aware lesson as build_detection.py: a pipe-separated HGVS carries a
# different cDNA number per transcript for the same physical variant.
_HGVS_FIELD_RE = re.compile(r"HGVS=(.*?)(?:,\s*Zygosity=|\n)")
_CDNA_TOKEN_RE = re.compile(r"c\.[^\s:;|]+")


def _own_cdnas(variant_context: str) -> set[str]:
    m = _HGVS_FIELD_RE.search(variant_context)
    if not m:
        return set()
    return set(_CDNA_TOKEN_RE.findall(m.group(1)))


# PS1 and PM5 are missense-only criteria. A stop-gained candidate is PVS1
# territory — PS1 on "same p.XxxNNNTer via a different nucleotide" double-
# counts the same loss-of-function evidence PVS1 already scores, and a
# nonsense hit at the residue is not a "different missense change" for PM5.
_PROTEIN_TOKEN_RE = re.compile(r"p\.\(?([A-Za-z*]+?)(\d+)([A-Za-z*=?]+)\)?")
_STOP_TOKENS = {"x", "*", "ter"}


def _is_missense(ref: str, alt: str) -> bool:
    ref_l, alt_l = ref.lower(), alt.lower()
    if alt_l in _STOP_TOKENS or "ter" in alt_l or "fs" in alt_l or "*" in alt_l:
        return False
    if any(k in alt_l for k in ("del", "ins", "dup", "ext", "=", "?")):
        return False
    return alt_l != ref_l


def _own_is_missense(variant_context: str) -> bool:
    """True if any protein change in this variant's own HGVS= field is a
    missense substitution (transcripts of one physical variant agree on
    consequence class, so any() vs all() only matters for malformed rows)."""
    m = _HGVS_FIELD_RE.search(variant_context)
    if not m:
        return False
    return any(_is_missense(ref, alt) for ref, _pos, alt in _PROTEIN_TOKEN_RE.findall(m.group(1)))


def _hit_is_missense(protein: str) -> bool:
    pm = re.match(r"([A-Za-z*]+?)(\d+)([A-Za-z*]+)$", protein)
    return bool(pm) and _is_missense(pm.group(1), pm.group(3))

_PS1_LINE_RE = re.compile(r"^-\s*PS1\b.*$", re.MULTILINE)
_PM5_LINE_RE = re.compile(r"^-\s*PM5\b.*$", re.MULTILINE)
_CRITERIA_INSERT_RE = re.compile(r"(?=\*\*ACMG points:\*\*)")


def _find_hits(variant_context: str) -> list[dict]:
    m = _RESIDUE_BLOCK_RE.search(variant_context)
    if not m:
        return []
    block = m.group(0)
    hits = []
    for hm in _HIT_RE.finditer(block):
        hits.append({
            "nt": hm.group("nt"),
            "protein": hm.group("protein"),
            "classification": hm.group("classification").strip(),
            "tag": hm.group("tag"),
            "vid": hm.group("vid"),
            "pathogenic": is_pathogenic_clinvar(hm.group("classification")),
            "missense": _hit_is_missense(hm.group("protein")),
        })
    return hits


def validate_ps1_pm5(conclusion_text: str, variant_context: str) -> str:
    """
    Scans the CLINVAR RESIDUE-LEVEL SEARCH evidence (if present) for a
    PS1-eligible hit (SAME amino acid change, pathogenic/likely pathogenic,
    a different nucleotide than this variant's own) and a PM5-eligible hit
    (DIFFERENT amino acid change, pathogenic/likely pathogenic). Adds
    whichever the model's own criteria list is missing, with the ClinVar
    Variation ID/URL inline, and adjusts the stated points/classification.
    Missense-only: no-op if this variant is not a missense substitution,
    and non-missense (stop/frameshift) hits are never used.
    No-op if no CLINVAR RESIDUE-LEVEL SEARCH block, or both are already
    present, or neither hit type exists in the evidence.
    """
    if not _own_is_missense(variant_context):
        return conclusion_text
    hits = _find_hits(variant_context)
    if not hits:
        return conclusion_text

    own_cdnas = _own_cdnas(variant_context)

    has_ps1 = bool(_PS1_LINE_RE.search(conclusion_text))
    has_pm5 = bool(_PM5_LINE_RE.search(conclusion_text))
    if has_ps1 and has_pm5:
        return conclusion_text

    ps1_hit = None
    pm5_hit = None
    for h in hits:
        if h["nt"] in own_cdnas or not h["pathogenic"] or not h["missense"]:
            continue  # own record, not an established P/LP precedent, or not missense
        if h["tag"] == "SAME" and ps1_hit is None:
            ps1_hit = h
        elif h["tag"] == "DIFFERENT" and pm5_hit is None:
            pm5_hit = h

    text = conclusion_text
    delta = 0.0

    if ps1_hit and not has_ps1:
        line = (
            f"- PS1 [Strong, +4]: Same amino acid change (p.{ps1_hit['protein']}) as a "
            f"previously established {ps1_hit['classification']} allele ({ps1_hit['nt']}) at "
            f"the same residue, via a different nucleotide change — ClinVar Variation ID "
            f"{ps1_hit['vid']} (https://www.ncbi.nlm.nih.gov/clinvar/variation/{ps1_hit['vid']}/) "
            "[auto-added from CLINVAR RESIDUE-LEVEL SEARCH — evidence supported this but it "
            "was missing from the model's own criteria list].\n"
        )
        text = _CRITERIA_INSERT_RE.sub(line, text, count=1)
        delta += 4

    if pm5_hit and not has_pm5:
        line = (
            f"- PM5 [Moderate, +2]: Different missense change (p.{pm5_hit['protein']}) at the "
            f"same residue as this variant, previously established as {pm5_hit['classification']} "
            f"— ClinVar Variation ID {pm5_hit['vid']} "
            f"(https://www.ncbi.nlm.nih.gov/clinvar/variation/{pm5_hit['vid']}/) "
            "[auto-added from CLINVAR RESIDUE-LEVEL SEARCH — evidence supported this but it "
            "was missing from the model's own criteria list].\n"
        )
        text = _CRITERIA_INSERT_RE.sub(line, text, count=1)
        delta += 2

    if delta:
        text = adjust_points_line(text, delta)

    return text


_VID_LIST_RE = re.compile(r"Variation ID[s]?\s*((?:\d+(?:\s*,\s*|\s+and\s+)?)+)")
_VID_URL_RE = re.compile(r"clinvar/variation/(\d+)")


def _cited_vids(line: str) -> set[str]:
    vids = set(_VID_URL_RE.findall(line))
    for grp in _VID_LIST_RE.findall(line):
        vids.update(re.findall(r"\d+", grp))
    return vids


def strip_ungrounded_ps1_pm5(conclusion_text: str, variant_context: str) -> str:
    """
    Strips every model-written PS1/PM5 line that is not grounded in a
    qualifying CLINVAR RESIDUE-LEVEL SEARCH hit. Each line is evaluated on
    its own citations (Stage 4 and Stage 5 texts can each carry independent,
    separately-hallucinated criteria lists).

    Removed when ANY of:
      - this variant is not a missense substitution (PS1/PM5 are
        missense-only; a stop-gained candidate is PVS1 territory);
      - there is no residue-search block / no parsed hit to check against
        (no grounding — never guess);
      - the line cites no hit it can be matched to. A line citing no cDNA
        token and no Variation ID used to be kept unchecked — a real
        observed failure: a PS1 line citing only protein changes and a
        Variation ID of a DIFFERENT-amino-acid hit, while the only SAME hit
        was this variant's own record, survived with +4;
      - none of the matched hits qualifies:
          PS1: tagged SAME, P/LP, missense, and not this variant's own cDNA;
          PM5: tagged DIFFERENT, P/LP, missense.

    Hits are matched by cDNA token or ClinVar Variation ID (either cited
    form); PM5 additionally by the hit's protein change (unambiguous for a
    DIFFERENT hit — it differs from this variant's own by definition).

    Does NOT adjust stated points — callers must follow with
    acmg_points.recompute_and_fix_totals(), which re-sums every affected
    total from whatever criteria bullets remain in its own scope.
    """
    own_missense = _own_is_missense(variant_context)
    hits = _find_hits(variant_context) if own_missense else []
    own = _own_cdnas(variant_context)
    by_nt = {h["nt"]: h for h in hits}
    by_vid = {h["vid"]: h for h in hits}
    by_protein = {h["protein"].lower(): h for h in hits if h["tag"] == "DIFFERENT"}

    def _qualifies(h: dict, criterion: str) -> bool:
        if not (h["pathogenic"] and h["missense"]):
            return False
        if criterion == "PS1":
            return h["tag"] == "SAME" and h["nt"] not in own
        return h["tag"] == "DIFFERENT"

    def _make_validator(criterion: str):
        def _validate(m: re.Match) -> str:
            if not hits:
                return ""
            line = m.group(0)
            matched = []
            for nt in {t.rstrip(").,;:") for t in _CDNA_TOKEN_RE.findall(line)}:
                if nt in by_nt:
                    matched.append(by_nt[nt])
            for vid in _cited_vids(line):
                if vid in by_vid:
                    matched.append(by_vid[vid])
            if criterion == "PM5":
                # Normalized to 3-letter codes: the model mixes notations
                # (e.g. "p.E123Gly") while hit lines are always 3-letter.
                for ref, pos, alt in PROTEIN_CHANGE_RE.findall(line):
                    key = f"{_to_aa3(ref)}{pos}{_to_aa3(alt)}".lower()
                    if key in by_protein:
                        matched.append(by_protein[key])
            if any(_qualifies(h, criterion) for h in matched):
                return line
            return ""
        return _validate

    text = _PS1_LINE_RE.sub(_make_validator("PS1"), conclusion_text)
    return _PM5_LINE_RE.sub(_make_validator("PM5"), text)

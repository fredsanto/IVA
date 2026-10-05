"""
pipeline/core/evidence_sources.py — which source a piece of retrieved evidence
came from, found by code, so the clinical report can cite it.

The retrieved evidence for a variant is a sequence of tool blocks. A text
(a condition name, an inheritance quote) is attributed to the block lines that
contain it:
  - inside a GeneReviews chapter ("--- <title> (GeneReviews NBKxxxx ...") ->
    "GeneReviews NBKxxxx";
  - a PubMed source-list line ("- [PMID:x, year] <title>") -> that PMID;
  - any other line -> the PMIDs in the sentence(s) of that line containing it;
  - the "conditions:" line of a CGD-sourced literature block -> "CGD".
Conditions are also matched against the gene's MedGen concepts (MedGen CUI and
OMIM numbers) and the input file's OMIM_phenotype field.
"""

import re

from pipeline.tools.medgen_features import gene_disease_ids

MAX_REFS = 4

_NBK_HEADER_RE = re.compile(r"^---\s*(.+?)\s*\(GeneReviews (NBK\d+)")
# Start of a new tool block (ends a GeneReviews chapter / CGD block).
_BLOCK_HEADER_RE = re.compile(
    r"^(?:[A-Z][A-Z0-9 /()'+.,-]{5,}(?:\(|:)|PubMed gene-disease evidence|LITERATURE SEARCH|VARIANT \d+:)")
# Section labels inside a GeneReviews chapter summary — not new blocks.
_GR_SECTION_RE = re.compile(r"^(?:CLINICAL CHARACTERISTICS|DIAGNOSIS/TESTING|GENETIC COUNSELING|MANAGEMENT)\b")
_SOURCE_LINE_RE = re.compile(r"^\s*-\s*\[PMID:(\d+)")
_PMID_RE = re.compile(r"PMID:?\s*(\d{6,9})")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def refs_for(text: str, evidence: str) -> list[str]:
    """Source IDs ("GeneReviews NBKx", "PMID:x", "CGD") of the evidence lines
    that contain `text` (case/punctuation-insensitive), at most MAX_REFS."""
    target = _norm(text)
    if len(target) < 4:
        return []
    refs: list[str] = []
    nbk, cgd = None, False
    for line in evidence.splitlines():
        stripped = line.strip()
        m = _NBK_HEADER_RE.match(stripped)
        if m:
            nbk, cgd = m.group(2), False
        elif _BLOCK_HEADER_RE.match(stripped) and not (nbk and _GR_SECTION_RE.match(stripped)):
            nbk, cgd = None, "source: CGD" in stripped
        if target not in _norm(line):
            continue
        if nbk:
            refs.append(f"GeneReviews {nbk}")
        elif (s := _SOURCE_LINE_RE.match(line)):
            refs.append(f"PMID:{s.group(1)}")
        elif cgd and "conditions:" in stripped:
            refs.append("CGD")
        else:
            refs += [f"PMID:{p}" for sent in _SENTENCE_SPLIT_RE.split(line)
                     if target in _norm(sent) for p in _PMID_RE.findall(sent)]
    return list(dict.fromkeys(refs))[:MAX_REFS]


def condition_refs(condition: str, gene: str, evidence: str, omim_phenotype: str = "") -> list[str]:
    """Source IDs for one of the gene's conditions: OMIM number and MedGen CUI
    when a MedGen concept of the gene carries this name, "OMIM (input file)"
    when the input's OMIM_phenotype field names it, then the evidence lines
    naming it (refs_for)."""
    refs: list[str] = []
    want = _norm(condition)
    for title, ids in gene_disease_ids(gene).items():
        if _norm(title) == want:
            refs += [f"OMIM:{m}" for m in ids["omim"]]
            if ids["cui"]:
                refs.append(f"MedGen {ids['cui']}")
    if want and want in _norm(omim_phenotype) and not any(r.startswith("OMIM:") for r in refs):
        refs.append("OMIM (input file)")
    return list(dict.fromkeys(refs + refs_for(condition, evidence)))


def ids_in(text: str) -> set[str]:
    """Every source ID token in `text`, in the forms refs_for/condition_refs
    produce plus UniProt/NCBI Gene/ClinVar/SCV/URL citations."""
    found = set(f"PMID:{p}" for p in _PMID_RE.findall(text))
    found |= set(re.findall(r"NBK\d+", text))
    found |= set(re.findall(r"OMIM:\d+", text))
    found |= set(re.findall(r"\bC\d{7}\b", text))
    return found

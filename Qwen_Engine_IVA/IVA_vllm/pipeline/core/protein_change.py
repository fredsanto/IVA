"""
pipeline/core/protein_change.py — shared protein-change (p.XxxNNNYyy) parsing.

Used by both websearch_agent.py (residue-level web search seeds) and
clinvar_residue_search.py (deterministic ClinVar residue query) so both stay
on identical amino-acid parsing/formatting — a mismatch between the two would
silently produce inconsistent PS1/PM5 evidence between the two tools.
"""

import re

AA_1TO3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln",
    "E": "Glu", "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys",
    "M": "Met", "F": "Phe", "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp",
    "Y": "Tyr", "V": "Val", "X": "Ter", "*": "Ter",
}

PROTEIN_CHANGE_RE = re.compile(r"p\.\(?([A-Za-z*]{1,3})(\d+)([A-Za-z*]{1,3})\)?")


def _to_aa3(token: str) -> str:
    return AA_1TO3.get(token.upper(), token.capitalize()) if len(token) == 1 else token.capitalize()


def extract_residue_queries(hgvs: str) -> list[dict]:
    """
    One entry per distinct residue position across all transcript segments of
    a (possibly multi-transcript, pipe-separated) HGVS string — e.g.
    [{"position": "34", "query": "p.Lys34", "own_change": "p.Lys34Thr",
      "own_aa3": "Thr"}].

    Different transcripts of the same physical variant can carry different
    residue numbers (alternate first exons/UTRs shift the coding start), so
    every distinct position is extracted, not just the first segment.
    3-letter form matches ClinVar's own variant-naming convention (e.g.
    "Lys34"), which a bare 1-letter query ("K34") often misses.
    """
    seeds: list[dict] = []
    seen_positions: set[str] = set()
    for m in PROTEIN_CHANGE_RE.finditer(hgvs):
        aa1, pos, aa2 = m.group(1), m.group(2), m.group(3)
        if pos in seen_positions:
            continue
        seen_positions.add(pos)
        aa1_3 = _to_aa3(aa1)
        aa2_3 = _to_aa3(aa2)
        seeds.append({
            "position":   pos,
            "query":      f"p.{aa1_3}{pos}",
            "own_change": f"p.{aa1_3}{pos}{aa2_3}",
            "own_aa3":    aa2_3,
        })
    return seeds

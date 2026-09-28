"""
pipeline/core/acmg_pm1.py — grounds PM1 in deterministic tool verdicts:
  - "PM1 HOTSPOT: MET" — CLINVAR HOTSPOT WINDOW (pipeline/tools/clinvar_hotspot.py,
    VarSome hotspot rule), or
  - "PM1 DOMAIN: MET"  — UNIPROT DOMAIN EVIDENCE (pipeline/tools/uniprot_domain.py:
    UniProt domain with >= 4 ClinVar P/LP missense, <= 10% benign incl. common /
    homozygous gnomAD missense, and independent UniProt support).

PM1 is kept or added only when one of them is MET; every model-written PM1
line is stripped otherwise (no block, NOT MET, or NOT EVALUABLE) — a gene-level
statement that a domain exists says nothing about whether this residue sits in
a pathogenic-variant cluster free of benign variation.
"""

import re

from pipeline.core.acmg_points import adjust_points_line, find_criteria, insert_criterion_line

_VERDICT_RE = re.compile(r"PM1 HOTSPOT:\s*(MET|NOT MET|NOT EVALUABLE)")
_DOMAIN_VERDICT_RE = re.compile(r"PM1 DOMAIN:\s*(MET|NOT MET|NOT EVALUABLE)")
_WINDOW_LINE_RE = re.compile(r"^[ \t]*Residue p\.\S+, window .*— MET$", re.MULTILINE)
_DOMAIN_LINE_RE = re.compile(r"^Residue (p\.\S+) lies in UniProt (\S+) (.+)\.$", re.MULTILINE)
_DOMAIN_COUNTS_RE = re.compile(r"^ClinVar missense in this region: ([^\n]+?) \(this variant excluded\)\.$", re.MULTILINE)
_DOMAIN_SUPPORT_RE = re.compile(r"^Independent UniProt support in region: \d+ — ([^\n]+)$", re.MULTILINE)
_PM1_LINE_RE = re.compile(r"^-\s*PM1(?:_\w+)?\b.*\n?", re.MULTILINE)


def _hotspot_met(variant_context: str) -> bool:
    m = _VERDICT_RE.search(variant_context)
    return bool(m) and m.group(1) == "MET"


def _domain_met(variant_context: str) -> bool:
    m = _DOMAIN_VERDICT_RE.search(variant_context)
    return bool(m) and m.group(1) == "MET"


def validate_pm1(conclusion_text: str, variant_context: str) -> str:
    """
    Strips every PM1 line unless the hotspot or the domain verdict is MET;
    when one is MET and the model's list has no PM1, adds PM1 [Moderate, +2]
    citing that evidence (hotspot preferred) and adjusts the stated points.

    Stripping does NOT adjust stated points — callers must follow with
    acmg_points.recompute_and_fix_totals(), same as strip_ungrounded_ps1_pm5.
    """
    hotspot, domain = _hotspot_met(variant_context), _domain_met(variant_context)
    if not (hotspot or domain):
        return _PM1_LINE_RE.sub("", conclusion_text)
    if any(c.code == "PM1" for c in find_criteria(conclusion_text)):
        return conclusion_text
    if hotspot:
        wm = _WINDOW_LINE_RE.search(variant_context)
        detail = wm.group(0).strip().removesuffix(" — MET") if wm else "ClinVar hotspot window"
        why = f"Located in a mutational hotspot — {detail} (CLINVAR HOTSPOT WINDOW, VarSome hotspot rule)"
    else:
        dm = _DOMAIN_LINE_RE.search(variant_context)
        cm = _DOMAIN_COUNTS_RE.search(variant_context)
        sm = _DOMAIN_SUPPORT_RE.search(variant_context)
        where = f"{dm.group(3)} (UniProt {dm.group(2)})" if dm else "a UniProt functional domain"
        why = (f"Located in a critical functional domain without benign variation — {where}; "
               f"ClinVar missense in the domain: {cm.group(1) if cm else 'see UNIPROT DOMAIN EVIDENCE'}"
               + (f"; UniProt support: {sm.group(1)}" if sm else "") + " (UNIPROT DOMAIN EVIDENCE)")
    line = (f"- PM1 [Moderate, +2]: {why} [auto-added — evidence supported this but it was "
            "missing from the model's own criteria list].\n")
    text = insert_criterion_line(conclusion_text, line)
    if text is None:
        return conclusion_text
    return adjust_points_line(text, 2)

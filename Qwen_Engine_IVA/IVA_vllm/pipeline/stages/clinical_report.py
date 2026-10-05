"""
pipeline/stages/clinical_report.py — the Clinical Conclusion of the report.

Code gathers the facts, the model writes the text.

Code decides, per variant, which section it belongs to:
  3) causative — phenotype verdict YES/PARTIAL, included by second triage on
     its own (not by the ACMG SF override), and its best layer block is a
     CAUSATIVE compound-het pair or totals >= 6 points outside a pair;
  4) VUS — same phenotype/inclusion gate, best block total in [4, 6);
  5) secondary findings — the ACMG SF list (P/LP in an SF gene), whatever the
     phenotype; being an SF gene never makes a variant causative.
Section 2 has one entry per gene with a causative variant. Code then gathers,
with their source IDs: the gene's function (gene_function.function_of), the
conditions matched to the patient (evidence_sources.condition_refs), the mode
of inheritance (moi gene_mode_sources_cache), each variant's ACMG criteria
lines from its layer block, the SF condition and inheritance (ACMG SF v3.2).

The model (prompts/clinical_report.txt) writes one text field per item,
returned as JSON by constrained decoding with exactly the fields code asked
for — it cannot add, drop or move an item. Code then removes any cited ID not
given for that item and any sentence naming another gene of the run, and
renders the sections with fixed headers and the criteria copied verbatim.
If the model call fails, the report is rendered from the facts alone.

Public API:
    run(...) -> str
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.core import acmg_sf, moi
from pipeline.core.evidence_sources import condition_refs
from pipeline.stages.final_conclusion import _CDNA_RE, _block_acmg_entries, _fmt_points
from pipeline.tools.gene_function import function_of
from pipeline.tools.medgen_features import split_condition_list

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "clinical_report.txt"
MAX_NEW_TOKENS = 3000
CAUSATIVE_THRESHOLD = 6.0
VUS_MIN = 4.0
_PHENOTYPE_MATCH = {"YES", "PARTIAL"}

_SEGREGATION_TEXT = {
    "de_novo": "de novo (absent in both parents)",
    "maternal": "inherited from the mother",
    "paternal": "inherited from the father",
    "both_carriers": "both parents carry it",
    "homozygous_parent": "a parent is homozygous for it",
    "uncertain": "uncertain from parental data",
    "insufficient_data": "no parental data",
}

# Citation IDs as they appear in facts and in the model's text.
_ID_PATTERNS = [
    (re.compile(r"PMID:?\s*(\d{6,9})"), "PMID:{}"),
    (re.compile(r"OMIM:?\s*#?(\d{6})"), "OMIM:{}"),
    (re.compile(r"(?:GeneReviews\s+)?(NBK\d+)"), "GeneReviews {}"),
    (re.compile(r"(?:MedGen\s+)?\b(C\d{7})\b"), "MedGen {}"),
    (re.compile(r"UniProt\s+([A-Z0-9]{6,10})"), "UniProt {}"),
    (re.compile(r"NCBI Gene\s+(\d+)"), "NCBI Gene {}"),
    (re.compile(r"ClinVar Variation ID\s+(\d+)"), "ClinVar Variation ID {}"),
    (re.compile(r"\b(SCV\d+)\b"), "{}"),
]
_PAREN_RE = re.compile(r"\s*\(([^()]*)\)")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _ids(text: str) -> set[str]:
    return {fmt.format(m.group(1)) for pat, fmt in _ID_PATTERNS for m in pat.finditer(text)} | (
        {"CGD"} if re.search(r"\bCGD\b", text) else set()) | (
        {"OMIM (input file)"} if "OMIM (input file)" in text else set()) | (
        {acmg_sf.ACMG_SF_REFERENCE} if acmg_sf.ACMG_SF_REFERENCE in text else set())


def _clean(text: str, allowed: set[str], other_genes: set[str]) -> str:
    """Drop cited IDs not in `allowed` (a parenthetical left with none goes)
    and sentences naming a gene of `other_genes`."""
    def _paren(m: re.Match) -> str:
        inner = m.group(1)
        if not _ids(inner):
            return m.group(0)
        kept = [part.strip() for part in re.split(r"[;,]", inner)
                if _ids(part) and _ids(part) <= allowed]
        return f" ({'; '.join(kept)})" if kept else ""
    text = _PAREN_RE.sub(_paren, text)
    for pat, fmt in _ID_PATTERNS:
        text = pat.sub(lambda m: m.group(0) if fmt.format(m.group(1)) in allowed else "", text)
    sentences = [s for s in _SENTENCE_RE.split(text.strip())
                 if not any(re.search(rf"(?<![\w-]){re.escape(g)}(?![\w-])", s) for g in other_genes)]
    return re.sub(r"\s{2,}", " ", " ".join(sentences)).replace(" .", ".").strip()


def _allow(refs) -> set[str]:
    """Allowed citation IDs for a field: each ref and the IDs inside it."""
    return {x for r in refs for x in ({r} | _ids(r))}


def _refs(refs: list[str]) -> str:
    return f" [{'; '.join(refs)}]" if refs else ""


def _variant_index(entry: dict, variants: list[dict], indices: list[int]) -> int | None:
    for i in indices:
        v = variants[i]
        if v.get("Gene") == entry["gene"] and entry["cdnas"] & set(_CDNA_RE.findall(v.get("HGVS", ""))):
            return i
    return None


def _best_entry(cands: list[dict]) -> dict:
    causative_pairs = [e for e in cands if e["causative_pair"]]
    return max(causative_pairs or cands, key=lambda e: e["points"])


def _is_causative(e: dict) -> bool:
    return e["causative_pair"] or (not e["pair"] and e["points"] >= CAUSATIVE_THRESHOLD)


def run(
    patient_phenotype: str,
    variants: list[dict],
    layer_outputs: dict[str, list[str]],
    include_indices: list[int],
    sf_forced: set[int],
    cluster_match: dict[int, str],
    phenotype_lists: dict[int, str],
    overlap_texts: dict[int, str],
    gene_modes: dict[str, str],
    gene_mode_sources: dict[str, list[dict]],
    evidence: dict[int, str],
    segregation: dict[int, str],
    clinvar_status: dict[int, str],
    actionable_flagged: list[dict],
    llm: "LLMClient",
) -> str:
    # ── Section membership (code) ───────────────────────────────────────────
    by_index: dict[int, list[dict]] = {}
    for e in _block_acmg_entries(layer_outputs):
        i = _variant_index(e, variants, include_indices)
        if i is not None:
            by_index.setdefault(i, []).append(e)
    eligible = [i for i in sorted(by_index)
                if cluster_match.get(i) in _PHENOTYPE_MATCH and i not in sf_forced]
    best = {i: _best_entry(by_index[i]) for i in eligible}
    causative = [i for i in eligible if _is_causative(best[i])]
    vus = [i for i in eligible if i not in causative and VUS_MIN <= best[i]["points"] < CAUSATIVE_THRESHOLD]
    causative_genes = list(dict.fromkeys(variants[i].get("Gene", "NA") for i in causative))
    run_genes = {v.get("Gene") for v in variants if v.get("Gene") not in (None, "", "NA")}

    # ── Facts (code) ────────────────────────────────────────────────────────
    facts: list[str] = []
    fields: dict[str, set[str]] = {"phenotype": set()}  # field -> allowed IDs
    field_gene: dict[str, str] = {}
    gene_facts: dict[str, dict] = {}
    for gene in causative_genes:
        idxs = [i for i in causative if variants[i].get("Gene") == gene]
        fn = function_of(gene)
        fn_refs = ([fn["source"]] + [f"PMID:{p}" for p in fn["pmids"]]) if fn else []
        matched: list[str] = []
        for i in idxs:
            for c in (moi.matched_conditions(phenotype_lists.get(i, ""), overlap_texts.get(i, ""))
                      or split_condition_list(phenotype_lists.get(i, ""))):
                if c not in matched:
                    matched.append(c)
        ev = "\n".join(evidence.get(i, "") for i in idxs)
        omim_field = " ".join(variants[i].get("OMIM_phenotype", "") for i in idxs)
        conditions = [(c, condition_refs(c, gene, ev, omim_field)) for c in matched]
        mode = gene_modes.get(gene, "")
        mode_sources = gene_mode_sources.get(gene, [])
        gene_facts[gene] = {"function": fn, "fn_refs": fn_refs, "conditions": conditions,
                            "mode": mode, "mode_sources": mode_sources}
        facts.append(f"GENE {gene}")
        facts.append(f"  FUNCTION: {fn['text'] if fn else 'not available'}{_refs(fn_refs)}")
        for c, refs in conditions:
            facts.append(f"  CONDITIONS: {c}{_refs(refs)}")
        facts.append(f"  INHERITANCE: {moi.MODE_LABELS.get(mode, mode)}")
        for s in mode_sources:
            detail = f' "{s["detail"]}"' if s["detail"] else ""
            facts.append(f"  INHERITANCE SOURCE: {s['source']}: {s['mode']}{detail}{_refs(s['refs'])}")
        for key, allowed in (("function", _allow(fn_refs)),
                             ("condition", _allow(r for _, refs in conditions for r in refs)),
                             ("inheritance", _allow(r for s in mode_sources for r in s["refs"]))):
            fields[f"gene_{gene}_{key}"] = allowed
            field_gene[f"gene_{gene}_{key}"] = gene

    def _variant_facts(kind: str, i: int) -> None:
        v, e = variants[i], best[i]
        facts.append(f"{kind.upper()} V{i + 1}: {v.get('Gene')} {v.get('HGVS', v.get('Variant', ''))}")
        facts.append(f"  Zygosity: {v.get('Zygosity', 'NA')}; segregation: "
                     f"{_SEGREGATION_TEXT.get(segregation.get(i, ''), 'not assessed')}; "
                     f"population frequency: {v.get('Frequency', 'NA')}; "
                     f"ClinVar: {clinvar_status.get(i, 'undefined')}")
        facts.append(f"  ACMG criteria lines ({e['layer']} layer):")
        facts.extend(f"    {line}" for line in e["lines"])
        if any("PHASE MUST BE CHECKED" in line for line in e["lines"]):
            facts.append("  PHASE NOTE: trans phase is assumed, not shown — one allele is de novo; "
                         "phase must be checked by long-read sequencing or cloning.")
        fields[f"{kind}_V{i + 1}"] = _ids("\n".join(e["lines"]))
        field_gene[f"{kind}_V{i + 1}"] = v.get("Gene", "")

    for i in causative:
        _variant_facts("variant", i)
    for i in vus:
        _variant_facts("vus", i)
    for a in actionable_flagged:
        gene = a["gene"]
        facts.append(f"SF V{a['index'] + 1}: {gene} {a['hgvs']}; condition: {a['condition']}; "
                     f"inheritance: {acmg_sf.ACMG_SF_INHERITANCE.get(gene, 'NA')} "
                     f"[{acmg_sf.ACMG_SF_REFERENCE}]; classification: {a['classification']}; "
                     f"zygosity: {a['zygosity']}")
        fields[f"sf_V{a['index'] + 1}"] = _allow([acmg_sf.ACMG_SF_REFERENCE])
        field_gene[f"sf_V{a['index'] + 1}"] = gene

    # ── Text (model) ────────────────────────────────────────────────────────
    texts = _write(patient_phenotype, "\n".join(facts), list(fields), llm)
    for name, allowed in fields.items():
        own = field_gene.get(name, "")
        texts[name] = _clean(texts.get(name, ""), allowed, run_genes - {own}) if texts.get(name) else ""

    # ── Render (code) ───────────────────────────────────────────────────────
    out = ["# Clinical Conclusion", "", "1) Clinical phenotype", "",
           texts["phenotype"] or f"As submitted: {patient_phenotype}", ""]

    out += ["2) Causative gene(s)", ""]
    if not causative_genes:
        out += ["None identified.", ""]
    for gene in causative_genes:
        g = gene_facts[gene]
        fn_text = texts[f"gene_{gene}_function"] or (
            (g["function"]["text"] + _refs(g["fn_refs"])) if g["function"] else "not available")
        out += [f"### {gene}",
                f"**Function:** {fn_text}",
                f"**Associated condition(s):** "
                f"{texts[f'gene_{gene}_condition'] or '; '.join(c for c, _ in g['conditions']) or 'none listed'}",
                f"**Condition sources:** " + ("; ".join(f"{c}{_refs(r)}" for c, r in g["conditions"]) or "none"),
                f"**Mode of inheritance:** {moi.MODE_LABELS.get(g['mode'], g['mode'])}. "
                f"{texts[f'gene_{gene}_inheritance']}",
                f"**Inheritance sources:** " + ("; ".join(
                    f"{s['source']}: {s['mode']}{_refs(s['refs'])}" for s in g["mode_sources"]) or "none"),
                ""]

    def _variant_block(kind: str, i: int) -> list[str]:
        v, e = variants[i], best[i]
        status = clinvar_status.get(i, "undefined")
        head = (f"### {v.get('Gene')} {v.get('HGVS', v.get('Variant', ''))}",
                f"Zygosity: {v.get('Zygosity', 'NA')} | Segregation: "
                f"{_SEGREGATION_TEXT.get(segregation.get(i, ''), 'not assessed')} | Analysed in: {e['layer']} layer"
                + (f" | ClinVar: {status}" if status and status != "undefined" else ""))
        return [*head, "", texts[f"{kind}_V{i + 1}"],
                f"**ACMG criteria ({e['layer']} layer):**", *e["lines"],
                f"**ACMG classification:** {e['label']} ({_fmt_points(e['points'])} pts total)", ""]

    out += ["3) Causative variant(s)", ""]
    if not causative:
        out += ["None identified.", ""]
    for i in causative:
        out += _variant_block("variant", i)

    out += ["4) Variants of uncertain significance (4-5 points)", ""]
    if not vus:
        out += ["None identified.", ""]
    for i in vus:
        out += _variant_block("vus", i)

    out += [f"5) Secondary findings — {acmg_sf.ACMG_SF_REFERENCE}", ""]
    if not actionable_flagged:
        out += ["None identified.", ""]
    for a in actionable_flagged:
        gene, key = a["gene"], f"sf_V{a['index'] + 1}"
        out += [f"### {gene} {a['hgvs']}",
                f"Condition: {a['condition']} | Inheritance: {acmg_sf.ACMG_SF_INHERITANCE.get(gene, 'NA')} | "
                f"Classification: {a['classification']} | Zygosity: {a['zygosity']}",
                texts[key], ""]
    return "\n".join(line for line in out if line is not None).rstrip() + "\n"


def _write(patient_phenotype: str, facts: str, field_names: list[str], llm: "LLMClient") -> dict[str, str]:
    """{field: text} from one constrained-decoding call; {} on failure."""
    schema = {"type": "object",
              "properties": {f: {"type": "string"} for f in field_names},
              "required": field_names, "additionalProperties": False}
    prompt = (_PROMPT_PATH.read_text(encoding="utf-8")
              .replace("{patient_phenotype}", patient_phenotype)
              .replace("{facts}", facts))
    try:
        return json.loads(llm.generate(
            system="You are a clinical geneticist writing a diagnostic report.",
            user=prompt, max_tokens=MAX_NEW_TOKENS, json_schema=schema))
    except Exception as exc:
        logger.warning("[ClinicalReport] text generation failed (%s) — report rendered from facts only.", exc)
        return {}

"""
pipeline/core/moi.py — Mode-of-Inheritance (MOI) classification.

Relocated from inline closures in pipeline/pipeline.py's Pipeline.run()
(Phase 0 of the MOI-layer restructuring) so the new per-MOI layer stages
(de novo, dominant, recessive, X-linked) can import this logic directly
instead of it being trapped inside one giant run() method. Behavior is
unchanged from the original inline version.

Public API:
    classify_inheritance_mode(text, allow_x_linked=True) -> str
    gene_chromosome(variants, idxs) -> str
    zygosity_is_confirmed_hom(zyg) -> bool
    build_gene_mode_cache(variants, kept_indices, evidence_by_index, condition_tags,
        overlap_texts, group_by_gene, llm) -> (dict[str, str], dict[str, str])
    matched_conditions(condition_tag, overlap_text) -> list[str]
    combine_modes(labels, gene_chrom) -> str
    build_recessive_gene_groups(variants, gene_mode_cache, kept_indices, group_by_gene) -> dict[str, list[int]]
    MODE_LABELS: dict[str, str]
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

MAX_WORKERS_MODE_CLASSIFICATION = 16
MAX_NEW_TOKENS_MODE_CLASSIFICATION = 500

_MODE_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "moi_condition_classification.txt"

# The literature reading answers each mode on its own line ("Dominant: YES
# "<quote>" | NO"); these are the line labels and the mode each one reports.
_MODE_LINE_LABELS = {
    "dominant": "AD",
    "recessive": "AR",
    "x-linked recessive": "XLR",
    "x-linked dominant": "XLD",
    "x-linked unspecified": "XL",
}


def _load_mode_prompt() -> str:
    if _MODE_PROMPT_PATH.exists():
        return _MODE_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"moi_condition_classification prompt not found at {_MODE_PROMPT_PATH}.")

# ── inheritance-mode regexes ─────────────────────────────────────────────────
_XLR_RE       = re.compile(r"x-?linked\s*recessive|\bXLR\b", re.IGNORECASE)
_XLD_RE       = re.compile(r"x-?linked\s*dominant|\bXLD\b", re.IGNORECASE)
_XL_RE        = re.compile(r"\bXL\b|x-?linked", re.IGNORECASE)
_AD_AR_RE     = re.compile(r"AD\s*/\s*AR|AR\s*/\s*AD", re.IGNORECASE)
_AR_TOKEN_RE  = re.compile(r"\bAR\b", re.IGNORECASE)
_BIALLELIC_RE = re.compile(r"bi-?allelic", re.IGNORECASE)
_AD_TOKEN_RE  = re.compile(r"\bAD\b", re.IGNORECASE)

_CHR_RE = re.compile(r"\bchr([0-9XYM]+)\b", re.IGNORECASE)

# Markers LitVar2SummaryTool emits (pipeline/tools/litvar2.py) — a NEGATIVE
# gene-disease signal always contains "NO DISEASE LINK" (both the gene-only
# and gene+phenotype no-hit branches use this literal phrase); a POSITIVE
# signal from either the primary gene+phenotype track or the supplemental
# known-disease/CGD track always starts its header "PubMed gene-disease
# evidence for ...". The rsID-level LitVar2 track ("LitVar2 variant-specific
# evidence for ...") is deliberately NOT treated as a gene-disease pertinence
# signal here — it answers a different question (is this specific variant
# discussed in the literature) than "is this gene linked to this phenotype".
_NO_DISEASE_LINK_RE       = re.compile(r"NO DISEASE LINK", re.IGNORECASE)
_POSITIVE_GENE_DISEASE_RE = re.compile(r"PubMed gene-disease evidence for", re.IGNORECASE)

PERTINENCE_LABELS = {
    True:  "Positive gene-phenotype literature link found (LitVar2 PubMed search).",
    False: "No gene-phenotype literature link found — LitVar2 searched and found none.",
    None:  "Unknown — gene-phenotype literature link not assessed or inconclusive.",
}

MODE_LABELS = {
    "AR":      "Autosomal recessive (AR)",
    "AD":      "Autosomal dominant (AD)",
    "AD_AR":   "Autosomal dominant/recessive (AD/AR) — both mechanisms reported for this gene",
    "XLR":     "X-linked recessive (XLR)",
    "XLD":     "X-linked dominant (XLD)",
    "XLD_XLR": "X-linked (both dominant and recessive reported for this gene)",
    "XL":      "X-linked (recessive/dominant not specified in available evidence)",
    "":        "Unknown — no inheritance stated by MedGen, the CSV field, the literature/GeneReviews evidence, or CGD",
}


def gene_chromosome(variants: list[dict], idxs: list[int]) -> str:
    """Best-effort chromosome for a gene, from the CSV Chromosome field if
    present, else parsed out of the free-text Variant field (e.g.
    "chr1:1000000 T?G" -> "1"). Returns "" if it can't be determined —
    callers must treat that as unknown, not as "not X"."""
    for i in idxs:
        chrom_field = str(variants[i].get("Chromosome", "") or "").strip()
        if chrom_field and chrom_field.upper() != "NA":
            m = _CHR_RE.search(f"chr{chrom_field}") or _CHR_RE.search(chrom_field)
            if m:
                return m.group(1).upper()
        m = _CHR_RE.search(str(variants[i].get("Variant", "") or ""))
        if m:
            return m.group(1).upper()
    return ""


def classify_inheritance_mode(text: str, allow_x_linked: bool = True) -> str:
    """Classify free text into "AR" / "AD" / "AD_AR" / "XLR" / "XLD" /
    "XLD_XLR" / "XL" (X-linked, recessive/dominant unspecified) / ""
    (unknown).

    Some genes genuinely have dual inheritance (e.g. this pipeline has
    seen CRYAA reported as "AD/AR" in the CSV) — collapsing that to a
    single mode silently drops one mechanism, so AD and AR signals are
    detected independently and combined into "AD_AR" when both are
    present, rather than first-match-wins.

    A bare/unqualified "XL"/"X-linked" mention (no recessive/dominant
    qualifier) is checked LAST, after autosomal AD/AR signals — this
    token is the weakest, most easily false-positived signal (e.g. a
    differential-diagnosis sentence mentioning an unrelated X-linked
    syndrome can pollute evidence text that also contains a real,
    specific AD/AR statement for the actual gene). Specific X-linked
    signals (XLR/XLD, which include their own "recessive"/"dominant"
    qualifier) are still checked first since they are unambiguous.

    allow_x_linked=False hard-disables all X-linked branches (XLR,
    XLD, XL) regardless of text content — used when the variant's own
    chromosome is confirmed non-X, since a gene not on chrX cannot be
    X-linked no matter what any retrieved text claims. Any text signal
    that would otherwise have classified as X-linked is discarded and
    classification falls through to whatever AR/AD signal remains."""
    if not text:
        return ""
    has_xlr = allow_x_linked and bool(_XLR_RE.search(text))
    has_xld = allow_x_linked and bool(_XLD_RE.search(text))
    has_xl  = allow_x_linked and bool(_XL_RE.search(text))
    has_ar  = bool(
        _AD_AR_RE.search(text) or _AR_TOKEN_RE.search(text)
        or _BIALLELIC_RE.search(text) or "recessive" in text.lower()
    )
    has_ad  = bool(
        _AD_AR_RE.search(text) or _AD_TOKEN_RE.search(text)
        or "dominant" in text.lower()
    )

    if has_xlr and has_xld:
        return "XLD_XLR"
    if has_xlr:
        return "XLR"
    if has_xld:
        return "XLD"
    if has_ar and has_ad:
        return "AD_AR"
    if has_ar:
        return "AR"
    if has_ad:
        return "AD"
    if has_xl:
        return "XL"
    return ""


def _tokens(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _quote_in_evidence(quote: str, evidence_tokens: str) -> bool:
    """True when the quote occurs in the evidence, word for word (case,
    punctuation and spacing ignored; an ellipsis may join two quoted parts,
    each of which must occur and be at least 3 words long)."""
    parts = [_tokens(p) for p in re.split(r"\.\.\.|…", quote)]
    parts = [p for p in parts if p]
    return bool(parts) and all(len(p.split()) >= 3 and p in evidence_tokens for p in parts)


def _parse_mode_lines(
    result: str, evidence: str, allow_x_linked: bool,
) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    """(modes answered YES with a quote found in the evidence,
    [(line label, quote)] accepted, [(line label, quote)] discarded because the
    quote is not in the evidence)."""
    evidence_tokens = _tokens(evidence)
    modes: list[str] = []
    accepted: list[tuple[str, str]] = []
    discarded: list[tuple[str, str]] = []
    for line in result.splitlines():
        text = line.strip().lstrip("-*• ").replace("**", "")
        label, sep, answer = text.partition(":")
        mode = _MODE_LINE_LABELS.get(label.strip().lower())
        if not sep or not mode or not answer.strip().upper().startswith("YES"):
            continue
        quote = answer.strip()[3:].strip().strip("\"“”'")
        if mode in ("XLR", "XLD", "XL") and not allow_x_linked:
            continue
        if _quote_in_evidence(quote, evidence_tokens):
            modes.append(mode)
            accepted.append((label.strip(), quote))
        else:
            discarded.append((label.strip(), quote))
    return modes, accepted, discarded


def _llm_classify_mode(
    gene: str,
    conditions: list[str],
    evidence_text: str,
    llm: "LLMClient",
    allow_x_linked: bool,
) -> tuple[list[str], str, list[tuple[str, str]], list[tuple[str, str]]]:
    """Modes of inheritance with which the gene causes *conditions*, as stated in
    the retrieved evidence (GeneReviews summaries, literature, gnomAD constraint).
    The LLM answers each mode separately with a verbatim quote
    (prompts/moi_condition_classification.txt); a mode counts only when its quote
    occurs in the evidence. The modes are combined by the caller (combine_modes).

    Returns (modes, reasoning_text, accepted, discarded) — modes [] means none
    stated; accepted/discarded are [(mode line label, quote)] for YES answers
    whose quote was / was not found in the evidence.
    allow_x_linked=False ignores X-linked answers (the gene's chromosome is
    known and is not X)."""
    conditions_block = (
        "\n".join(f"- {c}" for c in conditions) if conditions
        else "- (no condition named in the evidence — classify the gene's inheritance as stated)"
    )
    user_prompt = (_load_mode_prompt()
                   .replace("{gene}", gene)
                   .replace("{conditions_block}", conditions_block)
                   .replace("{evidence_text}", evidence_text))
    try:
        result = llm.generate(
            system=(
                "You are an expert clinical geneticist determining the mode of inheritance "
                "of a gene's conditions from the evidence. Limit your response to 200 words maximum."
            ),
            user=user_prompt,
            max_tokens=MAX_NEW_TOKENS_MODE_CLASSIFICATION,
        )
    except Exception as exc:
        logger.warning("[MOI] LLM mode classification failed for %s: %s", gene, exc)
        return [], "", [], []

    modes, accepted, discarded = _parse_mode_lines(result, evidence_text, allow_x_linked)
    for label, quote in discarded:
        logger.info("[MOI] Gene %s: %s YES discarded — quote not in evidence: %r", gene, label, quote)
    reasoning = result.split("\nDominant:")[0].replace("Reasoning:", "", 1).strip()
    return modes, reasoning, accepted, discarded


def _norm_name(name: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", name.lower()))


def matched_conditions(condition_tag: str, overlap_text: str) -> list[str]:
    """Conditions of Stage 1b's list that Stage 1c marked MATCH for this patient.
    Stage 1c writes one line per listed condition ("- <condition>: MATCH | NO
    MATCH — ..."); each line is recognised by its known condition name, then the
    verdict word that follows it."""
    from pipeline.tools.medgen_features import split_condition_list
    names = split_condition_list(condition_tag)
    matched = []
    for line in (overlap_text or "").splitlines():
        text = line.strip().lstrip("-*• ").strip()
        for name in names:
            if not text.lower().startswith(name.lower()):
                continue
            rest = text[len(name):].lstrip(" *")
            if rest.startswith("("):
                # the MedGen title echoed from the prompt: "<name> (MedGen: ...): MATCH"
                depth = 0
                for k, ch in enumerate(rest):
                    depth += (ch == "(") - (ch == ")")
                    if depth == 0:
                        rest = rest[k + 1:]
                        break
            if rest.lstrip(" *:").upper().startswith("MATCH") and name not in matched:
                matched.append(name)
    return matched


def _atoms(label: str) -> set[str]:
    return {"AD_AR": {"AD", "AR"}, "XLD_XLR": {"XLD", "XLR"}}.get(label, {label} if label else set())


def combine_modes(labels: list[str], gene_chrom: str) -> str:
    """Union of the modes reported by every source, as one label: AD + AR →
    AD_AR (dual inheritance, per protocol), XLD + XLR → XLD_XLR, and an
    unspecified XL is absorbed by a specific XLD/XLR. Autosomal and X-linked
    labels are exclusive — the gene's chromosome picks between them (autosomal
    unless the gene is on chrX)."""
    atoms: set[str] = set()
    for label in labels:
        atoms |= _atoms(label)
    auto = atoms & {"AD", "AR"}
    xl = atoms & {"XLD", "XLR", "XL"}
    use_x = bool(xl) and (gene_chrom == "X" or not auto)
    if use_x:
        if {"XLD", "XLR"} <= xl:
            return "XLD_XLR"
        for mode in ("XLR", "XLD", "XL"):
            if mode in xl:
                return mode
    if auto == {"AD", "AR"}:
        return "AD_AR"
    return next(iter(auto), "")


def zygosity_is_confirmed_hom(zyg) -> bool:
    """True only when the CSV Zygosity field explicitly reads homozygous —
    a confirmed-hom recessive call already explains the disease on its own
    and doesn't need a compound-het partner, so it's the only case worth
    excluding from sibling-context injection in Python.

    Deliberately NOT the inverse of "is het": Zygosity is frequently NA in
    the CSV (het status only visible via allelic balance), and previously
    this gate required proof of het rather than absence of hom, which
    silently dropped genuine compound-het pairs whenever Zygosity was NA
    (e.g. two variants in one gene, both Zygosity=NA, real het via AB ~0.5).
    Zygosity is not something this code should determine — the LLM sees the
    ALLELIC BALANCE block directly in the prompt and is instructed to derive
    zygosity from it when the CSV field is unhelpful. Any case that isn't a
    confirmed hom is handed to the model as a sibling candidate."""
    z = str(zyg or "").strip().lower()
    return "hom" in z or z in ("1/1", "1|1")


def build_gene_mode_cache(
    variants: list[dict],
    kept_indices: list[int],
    evidence_by_index: dict[int, str],
    condition_tags: dict[int, str],
    overlap_texts: dict[int, str],
    group_by_gene: Callable[[list[dict]], dict[str, list[int]]],
    llm: "LLMClient",
) -> tuple[dict[str, str], dict[str, str]]:
    """Mode of inheritance per gene (genes with >=1 kept variant), decided after
    the phenotype steps for the conditions that match the patient.

    Target conditions: the gene's Stage 1b conditions that Stage 1c marked
    MATCH; when none matched, all of the gene's conditions. Sources, all
    consulted and combined (combine_modes — AD from one source and AR from
    another gives AD_AR):
      1. MedGen — inheritance of the MedGen disease concepts linked to the
         gene's NCBI Gene record whose title is a target condition (all linked
         concepts when no condition matched);
      2. the CSV Inheritance / OMIM_inheritance fields;
      3. the literature and GeneReviews evidence, read by the LLM for the target
         conditions — always run, even when 1-2 already give a mode.
    CGD's gene-level inheritance is used only when all three give nothing.
    An X-linked mode is excluded for a gene whose chromosome is known and not X.

    Returns (gene_mode_cache, gene_mode_reasoning_cache); mode "" = unknown.
    The reasoning text lists each source's contribution."""
    from pipeline.tools.litvar2 import LitVar2SummaryTool
    from pipeline.tools.medgen_features import gene_disease_inheritance, split_condition_list

    kept_set = set(kept_indices)
    targets: list[tuple] = []
    for gene, idxs in group_by_gene(variants).items():
        kept_in_gene = [i for i in idxs if i in kept_set]
        if not kept_in_gene or gene == "NA":
            continue
        gene_chrom = gene_chromosome(variants, kept_in_gene)
        allow_x_linked = not (gene_chrom and gene_chrom != "X")
        all_names: list[str] = []
        matched: list[str] = []
        for i in kept_in_gene:
            for name in split_condition_list(condition_tags.get(i, "")):
                if name not in all_names:
                    all_names.append(name)
            for name in matched_conditions(condition_tags.get(i, ""), overlap_texts.get(i, "")):
                if name not in matched:
                    matched.append(name)
        targets.append((gene, kept_in_gene, gene_chrom, allow_x_linked, all_names, matched))

    def _resolve_one(item: tuple) -> tuple[str, str, str]:
        gene, kept_in_gene, gene_chrom, allow_x_linked, all_names, matched = item
        conditions = matched or all_names
        labels: list[str] = []
        notes: list[str] = []

        medgen = gene_disease_inheritance(gene)
        if medgen is None:
            notes.append("MedGen: lookup failed")
        else:
            wanted = {_norm_name(n) for n in matched}
            used = [(t, m) for t, m in medgen if not wanted or _norm_name(t) in wanted]
            medgen_labels = [classify_inheritance_mode(mode_name, allow_x_linked)
                             for _, modes in used for mode_name in modes]
            medgen_labels = [lab for lab in medgen_labels if lab]
            labels += medgen_labels
            notes.append(
                f"MedGen: {combine_modes(medgen_labels, gene_chrom) or 'none'}"
                + (f" ({'; '.join(t for t, m in used if m)})" if any(m for _, m in used) else "")
            )

        csv_text = " ".join(
            f"{variants[i].get('Inheritance', '')} {variants[i].get('OMIM_inheritance', '')}"
            for i in kept_in_gene
        )
        csv_label = classify_inheritance_mode(csv_text, allow_x_linked)
        if csv_label:
            labels.append(csv_label)
        notes.append(f"CSV field: {csv_label or 'none'}")

        evidence = " ".join(evidence_by_index.get(i, "") for i in kept_in_gene)
        llm_modes, llm_reasoning, accepted, discarded = _llm_classify_mode(
            gene, conditions, evidence, llm, allow_x_linked)
        labels += llm_modes
        notes.append(
            f"Literature/GeneReviews: {combine_modes(llm_modes, gene_chrom) or 'UNKNOWN'}"
            + "".join(f' [{label}: "{quote}"]' for label, quote in accepted)
            + "".join(f' [{label} YES discarded — quote not in evidence: "{quote}"]'
                      for label, quote in discarded)
        )

        mode = combine_modes(labels, gene_chrom)
        if not mode:
            try:
                cgd_label = classify_inheritance_mode(
                    LitVar2SummaryTool.get_cgd_inheritance(gene), allow_x_linked)
            except Exception as exc:
                logger.warning("[MOI] CGD inheritance lookup failed for %s: %s", gene, exc)
                cgd_label = ""
            mode = cgd_label
            notes.append(f"CGD (fallback): {cgd_label or 'none'}")

        scope = (f"conditions matching the patient: {', '.join(matched)}" if matched
                 else "no condition matched the patient — all of the gene's conditions")
        reasoning = f"Sources ({scope}) — " + "; ".join(notes) + "."
        if llm_reasoning:
            reasoning += f" Literature reading: {llm_reasoning}"
        return gene, mode, reasoning

    gene_mode_cache: dict[str, str] = {}
    gene_mode_reasoning_cache: dict[str, str] = {}
    if targets:
        workers = min(MAX_WORKERS_MODE_CLASSIFICATION, len(targets))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for future in as_completed([pool.submit(_resolve_one, t) for t in targets]):
                gene, mode, reasoning = future.result()
                gene_mode_cache[gene] = mode
                gene_mode_reasoning_cache[gene] = reasoning
                logger.info("[MOI] Gene %s: mode=%r — %s", gene, mode or "UNKNOWN",
                            reasoning.split(" Literature reading:")[0])
    return gene_mode_cache, gene_mode_reasoning_cache


def build_recessive_gene_groups(
    variants: list[dict],
    gene_mode_cache: dict[str, str],
    kept_indices: list[int],
    group_by_gene: Callable[[list[dict]], dict[str, list[int]]],
) -> dict[str, list[int]]:
    """Compound-het candidate groups: recessive-relevant genes (AR, XLR, or a
    gene with dual AD_AR/XLD_XLR inheritance where the recessive mechanism
    is still in play) with >=2 kept variants — a second variant in a purely
    dominant/XLD gene doesn't change the first variant's standing, so
    purely-dominant genes are excluded from grouping."""
    kept_set = set(kept_indices)
    recessive_gene_groups: dict[str, list[int]] = {}
    for gene, idxs in group_by_gene(variants).items():
        kept_in_gene = [i for i in idxs if i in kept_set]
        if len(kept_in_gene) >= 2 and gene_mode_cache.get(gene) in ("AR", "XLR", "AD_AR", "XLD_XLR"):
            recessive_gene_groups[gene] = kept_in_gene
    return recessive_gene_groups


def build_gene_chrom_cache(
    variants: list[dict],
    kept_indices: list[int],
    group_by_gene: Callable[[list[dict]], dict[str, list[int]]],
) -> dict[str, str]:
    """Persisted per-gene chromosome, over genes with >=1 kept variant. Layer 6
    (X-linked) filters on this directly (chromosome == "X") rather than on
    gene_mode_cache — a different filter shape than Layers 3-5, which filter
    on inheritance mode. Previously gene_chromosome() was only called
    transiently inside build_gene_mode_cache to gate allow_x_linked; this
    cache stores that same result for reuse instead of recomputing it."""
    kept_set = set(kept_indices)
    gene_chrom_cache: dict[str, str] = {}
    for gene, idxs in group_by_gene(variants).items():
        kept_in_gene = [i for i in idxs if i in kept_set]
        if not kept_in_gene or gene == "NA":
            continue
        gene_chrom_cache[gene] = gene_chromosome(variants, kept_in_gene)
    return gene_chrom_cache


def derive_pertinence(gene: str, context_slice: str) -> bool | None:
    """Phenotype-pertinence for one variant's gene, sourced from the
    LitVar2SummaryTool evidence already retrieved into its context slice —
    not re-derived via a new tool/LLM call.

    True  — a positive gene-disease/phenotype hit exists somewhere in the
             slice (either the primary gene+phenotype PubMed track or the
             supplemental known-disease/CGD track — either is sufficient,
             so a track-1 miss doesn't override a track-2 hit).
    False — the only gene-disease signal present is a "NO DISEASE LINK"
             marker, with no positive signal anywhere else in the slice.
    None  — neither marker is present (e.g. the tool was gated off, failed,
             or produced no structured gene-disease signal at all) — absence
             of proof isn't proof of absence, so this stays unknown rather
             than being forced to True or False."""
    if not context_slice:
        return None
    if _POSITIVE_GENE_DISEASE_RE.search(context_slice):
        return True
    if _NO_DISEASE_LINK_RE.search(context_slice):
        return False
    return None


def build_pertinence_cache(
    variants: list[dict],
    kept_indices: list[int],
    context_slices: list[str],
) -> dict[int, bool | None]:
    """Per-variant phenotype pertinence, computed unconditionally for every
    kept variant (independent of the first_triage KEEP/DISCARD threshold,
    which conflates general relevance with phenotype-specificity and is
    skipped entirely when n <= TRIAGE_ENABLED_THRESHOLD)."""
    return {
        i: derive_pertinence(variants[i].get("Gene", "NA"), context_slices[i])
        for i in kept_indices
    }

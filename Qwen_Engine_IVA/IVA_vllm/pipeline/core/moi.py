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
    build_gene_mode_cache(variants, kept_indices, context_slices, group_by_gene, llm) ->
        (dict[str, str], dict[str, str])
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
MAX_NEW_TOKENS_MODE_CLASSIFICATION = 350

_MODE_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "moi_mode_classification.txt"

_MODE_TOKEN_RE = re.compile(
    r"Mode:\s*(AD_AR|XLD_XLR|AD|AR|XLR|XLD|XL|UNKNOWN)", re.IGNORECASE
)


def _load_mode_prompt() -> str:
    if _MODE_PROMPT_PATH.exists():
        return _MODE_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"moi_mode_classification prompt not found at {_MODE_PROMPT_PATH}.")

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
    "":        "Unknown — no inheritance signal in CSV field, CGD table, or retrieved evidence",
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


def _llm_classify_mode(
    gene: str,
    evidence_text: str,
    llm: "LLMClient",
    allow_x_linked: bool,
) -> tuple[str, str]:
    """LLM-reasoned replacement for the old regex-based tier-3 fallback.

    Reads the gene's pooled free-text evidence (literature, websearch,
    gnomAD constraint block) and reasons explicitly about AD/AR/X-linked
    mode, including whether a dominant-negative mechanism is described —
    which is genuinely AD regardless of gnomAD pLI/LOEUF LoF-tolerance,
    unlike haploinsufficiency-based AD. See prompts/moi_mode_classification.txt
    for the full reasoning rules (negation-safe, mechanism-aware).

    Returns (mode, reasoning_text). mode == "" means UNKNOWN/unresolved —
    same convention as classify_inheritance_mode(). allow_x_linked=False
    strips any X-linked mode the LLM returns, mirroring the hard chrX gate
    applied elsewhere in this module.
    """
    template = _load_mode_prompt()
    user_prompt = template.replace("{gene}", gene).replace("{evidence_text}", evidence_text)

    try:
        result = llm.generate(
            system=(
                "You are an expert clinical geneticist determining a gene's mode of "
                "inheritance from literature evidence. Limit your response to 150 words maximum."
            ),
            user=user_prompt,
            max_tokens=MAX_NEW_TOKENS_MODE_CLASSIFICATION,
        )
    except Exception as exc:
        logger.warning("[MOI] LLM mode classification failed for %s: %s", gene, exc)
        return "", ""

    m = _MODE_TOKEN_RE.search(result)
    if not m:
        logger.warning("[MOI] LLM mode classification unparseable for %s: %r", gene, result)
        return "", result.strip()

    mode = m.group(1).upper()
    if mode == "UNKNOWN":
        mode = ""
    if not allow_x_linked and mode in ("XLR", "XLD", "XLD_XLR", "XL"):
        logger.debug(
            "[MOI] Gene %s: LLM returned X-linked mode %s but chrX is excluded — discarding",
            gene, mode,
        )
        mode = ""

    reasoning = result.split("Mode:")[0].replace("Reasoning:", "", 1).strip()
    return mode, reasoning


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
    context_slices: list[str],
    group_by_gene: Callable[[list[dict]], dict[str, list[int]]],
    llm: "LLMClient",
) -> tuple[dict[str, str], dict[str, str]]:
    """Compute inheritance mode per gene, over genes with >=1 kept variant,
    via a 3-tier fallback: (1) CSV Inheritance/OMIM_inheritance text, (2)
    NHGRI CGD inheritance lookup, (3) an LLM call reasoning over the
    retrieved literature/websearch evidence text (replaces the old naive
    regex substring match — see _llm_classify_mode's docstring for why:
    negation-blind keyword matching was misclassifying genes on incidental
    "dominant"/"recessive" mentions, and had no way to reconcile a
    dominant-negative mechanism against a LoF-tolerant gnomAD pLI/LOEUF).
    Chromosome-based X-linked gating (gene_chromosome) is applied at every
    tier. "" means still unknown after all three tiers.

    Returns (gene_mode_cache, gene_mode_reasoning_cache) — the second dict
    holds the LLM's reasoning text for genes resolved at tier 3 only ("" for
    genes resolved at tier 1/2, which are unambiguous structured facts that
    don't need justification)."""
    from pipeline.tools.litvar2 import LitVar2SummaryTool

    kept_set = set(kept_indices)
    gene_mode_cache: dict[str, str] = {}
    gene_mode_reasoning_cache: dict[str, str] = {}
    allow_x_linked_cache: dict[str, bool] = {}
    tier3_targets: list[tuple[str, list[int], bool]] = []

    # ── Pass 1: tiers 1/2 (cheap, structured-field regex) ────────────────────
    for gene, idxs in group_by_gene(variants).items():
        kept_in_gene = [i for i in idxs if i in kept_set]
        if not kept_in_gene or gene == "NA":
            continue

        # Hard biological constraint: a gene not located on chrX cannot be
        # X-linked, no matter what any text tier below claims. Only disable
        # X-linked classification when the chromosome is positively known
        # and is NOT X — an unparseable/missing chromosome stays permissive
        # since absence of proof isn't proof of absence.
        gene_chrom = gene_chromosome(variants, kept_in_gene)
        allow_x_linked = not (gene_chrom and gene_chrom != "X")
        allow_x_linked_cache[gene] = allow_x_linked
        if gene_chrom and not allow_x_linked:
            logger.debug(
                "[MOI] Gene %s is on chr%s — X-linked classification disabled",
                gene, gene_chrom,
            )

        inheritance_texts = " ".join(
            str(variants[i].get("Inheritance", "")) + " " + str(variants[i].get("OMIM_inheritance", ""))
            for i in kept_in_gene
        )
        mode = classify_inheritance_mode(inheritance_texts, allow_x_linked)
        if not mode:
            # CSV had no usable inheritance field — fall back to the NHGRI CGD
            # inheritance lookup (same table already used for known-disease
            # literature search) rather than skipping the safeguard entirely.
            try:
                cgd_inheritance = LitVar2SummaryTool.get_cgd_inheritance(gene)
            except Exception as exc:
                logger.warning("[MOI] CGD inheritance lookup failed for %s: %s", gene, exc)
                cgd_inheritance = ""
            mode = classify_inheritance_mode(cgd_inheritance, allow_x_linked)
        if mode:
            gene_mode_cache[gene] = mode
            gene_mode_reasoning_cache[gene] = ""
        else:
            # CGD table lags newly characterized gene-disease links — defer to
            # tier 3 (LLM reasoning over retrieved literature/web-search
            # evidence), run concurrently below rather than serially here.
            tier3_targets.append((gene, kept_in_gene, allow_x_linked))

    # ── Pass 2: tier 3 (LLM reasoning), concurrent across unresolved genes ───
    if tier3_targets:
        def _resolve_one(item: tuple[str, list[int], bool]) -> tuple[str, str, str]:
            gene, kept_in_gene, allow_x_linked = item
            evidence_text = " ".join(context_slices[i] for i in kept_in_gene)
            mode, reasoning = _llm_classify_mode(gene, evidence_text, llm, allow_x_linked)
            return gene, mode, reasoning

        workers = min(MAX_WORKERS_MODE_CLASSIFICATION, len(tier3_targets))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_resolve_one, item): item[0] for item in tier3_targets}
            for future in as_completed(futures):
                gene, mode, reasoning = future.result()
                gene_mode_cache[gene] = mode  # "" means still unknown after all three tiers
                gene_mode_reasoning_cache[gene] = reasoning
                logger.info("[MOI] Gene %s: LLM-resolved mode=%r", gene, mode or "UNKNOWN")

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

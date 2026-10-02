"""
pipeline/stages/reasoning.py — Two-call SLM clinical reasoning stage.

Splitting the original single call into two focused calls reduces hallucination
by separating the evidence-grounded narrative from the inclusion decision:

  Call 1 — reasoning:  step-by-step clinical reasoning using ONLY cited evidence.
                       Prompt: prompts/reasoning.txt
  Call 2 — second_triage: structured INCLUDE/EXCLUDE decision with justification.
                           Prompt: prompts/second_triage.txt
                           Format: Include-case / Exclude-case / Decision (3 lines)

The two outputs are concatenated before returning so that the REASONING display
section in pipeline.py is self-contained.

Public API:
    run_one(variant_context, llm) -> str
    parse_inclusion_decision(reasoning_text) -> tuple[str, str]
        Returns ("INCLUDE"|"EXCLUDE", "Include: <...> | Exclude: <...>").
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_REASONING_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "reasoning.txt"
_GENE_PHENOTYPE_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "gene_phenotype_extraction.txt"
)
_PHENOTYPE_OVERLAP_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "phenotype_overlap.txt"
)
_SCORING_PROMPT_PATH   = Path(__file__).parent.parent.parent / "prompts" / "second_triage.txt"
_XLINKED_PROMPT_PATH   = Path(__file__).parent.parent.parent / "prompts" / "second_triage_xlinked.txt"
_SINGLE_HIT_AR_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "second_triage_single_hit_recessive.txt"
)
_COMPOUND_HET_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "second_triage_compound_het_exception.txt"
)
_LIT_QUALITY_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "second_triage_literature_evidence_quality.txt"
)

# Isolated gene-phenotype-extraction call — short, single-line output
MAX_NEW_TOKENS_GENE_PHENOTYPE = 300
# Isolated phenotype-overlap call — one short line per condition + verdict
MAX_NEW_TOKENS_PHENOTYPE_OVERLAP = 600
# Call 1: step-by-step narrative — no score, citation-only rules apply
MAX_NEW_TOKENS_REASONING = 1500
# Call 2: three-line Include-case / Exclude-case / Decision block
# 200 gives headroom for any residual preamble before the 3 structured lines.
MAX_NEW_TOKENS_SCORING   = 200

_INCLUDE_CASE_RE = re.compile(r"Include-case:\s*(.+)", re.IGNORECASE)
_EXCLUDE_CASE_RE = re.compile(r"Exclude-case:\s*(.+)", re.IGNORECASE)
_DECISION_RE     = re.compile(r"Decision:\s*(INCLUDE|EXCLUDE)", re.IGNORECASE)
_CLUSTER_MATCH_RE = re.compile(
    r"CLUSTER_PHENOTYPE:\s*(YES|PARTIAL|NONE|INCIDENTAL|EXCLUDED)", re.IGNORECASE
)
# Stage 1c per-condition call token: "- <condition>: MATCH | NO MATCH | EXCLUDED — ..."
_CONDITION_CALL_RE = re.compile(r":\s*\**\s*(NO MATCH|MATCH|EXCLUDED)\b", re.IGNORECASE)
_PHENOTYPE_TAG_RE = re.compile(
    r"^\s*PHENOTYPE:\s*(.+)$", re.IGNORECASE | re.MULTILINE
)


def parse_cluster_match(reasoning_text: str) -> str:
    """
    Extract Stage 1's own "CLUSTER_PHENOTYPE: YES|PARTIAL|NONE|INCIDENTAL"
    verdict from its Phenotype fit step (see prompts/reasoning.txt).

    This exists so the verdict can be re-stated to downstream stages
    (second_triage, conclusion) as a small labeled backend-determined fact —
    same pattern as _inheritance_mode_block/_phase_fact in pipeline.py — instead
    of leaving those stages to re-read and possibly reinterpret the full
    reasoning narrative themselves. A single computed-once verdict, restated
    verbatim, is far more reliably followed than the same judgment buried in
    several paragraphs of prose and asked to be located again each time.

    Returns "YES", "PARTIAL", "NONE", "INCIDENTAL", or "UNKNOWN" if the line is
    missing (e.g. the model dropped the label — treat as unknown, not as NONE).
    """
    m = _CLUSTER_MATCH_RE.search(reasoning_text)
    return m.group(1).upper() if m else "UNKNOWN"


def parse_phenotype_tag(reasoning_text: str) -> str:
    """
    Extract Stage 1's own "PHENOTYPE: <condition(s)>" tag — the gene's
    retrieved condition/disease name(s), independent of whether they overlap
    the patient (see prompts/reasoning.txt). Re-stated to downstream stages
    the same way parse_cluster_match's verdict is, and reused verbatim as the
    reported condition name when a variant is routed to Actionable/Incidental
    findings (an INCIDENTAL cluster-match verdict) — never re-derived or
    invented downstream.

    Returns the tag's raw text, or "NA" if the line is missing.
    """
    m = _PHENOTYPE_TAG_RE.search(reasoning_text)
    return m.group(1).strip() if m else "NA"


# ── Prompt loaders ────────────────────────────────────────────────────────────

def _load_reasoning_prompt() -> str:
    if _REASONING_PROMPT_PATH.exists():
        return _REASONING_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Reasoning prompt not found at {_REASONING_PROMPT_PATH}."
    )


def _load_gene_phenotype_prompt() -> str:
    if _GENE_PHENOTYPE_PROMPT_PATH.exists():
        return _GENE_PHENOTYPE_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Gene-phenotype-extraction prompt not found at {_GENE_PHENOTYPE_PROMPT_PATH}."
    )


def run_gene_phenotype_extraction(evidence_context: str, llm: "LLMClient") -> str:
    """
    Isolated SLM call: compiles this gene's complete retrieved condition list
    from tool evidence ONLY (GeneReviews/OMIM/CGD/LitVar2/web-search/ClinVar
    output already gathered for this variant) — patient_phenotype must never
    be part of `evidence_context` (pipeline.py's _evidence_only_context strips
    the PATIENT DATA header off the variant's context slice before calling
    this) and is never mentioned in this call's own prompt either, so the
    model has no way to conflate the two.

    Real observed failure this fixes: a single combined call asked to both
    compile a gene's condition list AND judge phenotype fit against the
    patient (1) wrote the patient's own presenting complaint into the
    condition-list tag instead of the gene's actual conditions, and
    separately (2) omitted a well-documented, unrelated condition it was
    actively discussing elsewhere in that same output — reliably, across
    multiple attempts at tightening the combined prompt's wording. Never
    showing this sub-task the patient at all removes the first failure
    structurally and removes any incentive to filter the list down to
    "what's relevant to this patient" for the second.

    Returns the parsed "PHENOTYPE: <list>" tag content (see
    parse_phenotype_tag), or "NA" if the model's output could not be parsed.
    """
    template = _load_gene_phenotype_prompt()
    user_prompt = template.replace("{gene_evidence}", evidence_context)
    result = llm.generate(
        system=(
            "You are compiling a gene's known condition list from the "
            "evidence provided. You have not been given any patient "
            "information and must not reference or assume any."
        ),
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_GENE_PHENOTYPE,
    )
    return parse_phenotype_tag(result)


def run_phenotype_overlap(
    patient_phenotype: str, phenotype_list: str, llm: "LLMClient",
) -> tuple[str | None, str]:
    """
    Isolated SLM call: compares the patient's phenotype against the gene's
    condition list (Stage 1b's PHENOTYPE tag) only, each condition annotated
    with its NCBI MedGen/HPO clinical features (tools/medgen_features.py).
    No literature or other evidence is shown.

    Real observed failures this replaces: the CLUSTER_PHENOTYPE verdict used
    to be one instruction inside the long Stage 1 reasoning call, which also
    reads all the retrieved literature. A gene whose condition list named an
    eye condition was judged NONE because the retrieved papers were all
    about the gene's other (non-eye) condition; and a gene whose list named
    an eponymous syndrome with retinal features was judged NONE because
    nothing in the evidence said what that syndrome involves.

    Returns (verdict, text): verdict is "YES", "PARTIAL", "INCIDENTAL" or
    "EXCLUDED", or None when the list is empty/NA or the output has no verdict
    line — the caller then keeps Stage 1's own verdict.

    EXCLUDED (the patient phenotype states findings the patient does NOT have)
    is decided per condition: the gene's verdict is EXCLUDED only when no
    condition is MATCH and at least one is EXCLUDED — recomputed here from the
    per-condition lines, so the verdict line cannot contradict them.
    """
    from pipeline.tools.medgen_features import split_condition_list, condition_features

    names = split_condition_list(phenotype_list)
    if not names:
        return None, ""
    lines = []
    for name in names:
        hit = condition_features(name)
        if hit:
            title, feats = hit
            lines.append(f"- {name} (MedGen: {title}) — features: {', '.join(feats)}")
        else:
            lines.append(f"- {name} — features: none listed")
    prompt = (_PHENOTYPE_OVERLAP_PROMPT_PATH.read_text(encoding="utf-8")
              .replace("{patient_phenotype}", patient_phenotype)
              .replace("{conditions_block}", "\n".join(lines)))
    result = llm.generate(
        system="You compare a patient's phenotype against a gene's known conditions.",
        user=prompt,
        max_tokens=MAX_NEW_TOKENS_PHENOTYPE_OVERLAP,
    )
    m = _CLUSTER_MATCH_RE.search(result)
    verdict = m.group(1).upper() if m else None
    if verdict == "NONE":  # the list is non-empty, so "no match" is INCIDENTAL
        verdict = "INCIDENTAL"
    calls = [
        m.group(1).upper()
        for line in result.splitlines() if line.lstrip().startswith("- ")
        for m in [_CONDITION_CALL_RE.search(line)] if m
    ]
    if "MATCH" not in calls and "EXCLUDED" in calls:
        verdict = "EXCLUDED"
    elif verdict == "EXCLUDED" and "MATCH" in calls:
        verdict = "PARTIAL"  # a clean MATCH condition keeps the gene
    return verdict, result


def _load_scoring_prompt() -> str:
    if _SCORING_PROMPT_PATH.exists():
        return _SCORING_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Second-triage prompt not found at {_SCORING_PROMPT_PATH}."
    )


def _load_xlinked_block() -> str:
    """
    Conditional sub-prompt (see prompts/second_triage_xlinked.txt), spliced
    into second_triage.txt's {xlinked_block} placeholder only when this
    variant's gene is actually on chrX. Split out of the main prompt so that
    the ~40-line X-linked-specific carve-out — biologically inapplicable to
    the large majority of variants, which are autosomal — does not compete
    for the model's attention on every single second-triage call. Conditional
    prompt assembly, not a smaller/weaker rule: the full text is used
    verbatim whenever it does apply.
    """
    if _XLINKED_PROMPT_PATH.exists():
        return _XLINKED_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"X-linked second-triage sub-prompt not found at {_XLINKED_PROMPT_PATH}."
    )


def _load_single_hit_recessive_block() -> str:
    """
    Conditional sub-prompt (see prompts/second_triage_single_hit_recessive.txt):
    the "single heterozygous variant in an AR/XLR-only gene is insufficient"
    EXCLUDE ground, spliced into {single_hit_recessive_block} only when it
    could actually apply — i.e. NOT when Zygosity is already confirmed
    homozygous/hemizygous (the rule's own text says it can never fire then
    anyway) and NOT when the gene's resolved mode is confirmed purely
    dominant (AD/XLD, no recessive component at all, so the concern is moot
    from the start). Included by default (see include_single_hit_recessive's
    default in run_second_triage) so an uninitialized/legacy call site never
    silently loses applicable content — unlike the X-linked block, which is
    relevant only to a small minority of variants, this recessive-insufficiency
    ground is relevant to most heterozygous variants in most genes, so the
    safe default is to include it rather than omit it.
    """
    if _SINGLE_HIT_AR_PROMPT_PATH.exists():
        return _SINGLE_HIT_AR_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Single-hit-recessive second-triage sub-prompt not found at {_SINGLE_HIT_AR_PROMPT_PATH}."
    )


def _load_compound_het_exception_block() -> str:
    """
    Conditional sub-prompt (see prompts/second_triage_compound_het_exception.txt):
    the compound-heterozygous EXCEPTION — spliced into
    {compound_het_exception_block} only when sibling_context_block is
    non-empty, i.e. a "SIBLING VARIANT" block actually exists for this
    variant (the pipeline already restricts that to AR/AD-AR genes with >=2
    kept variants — see pipeline.py's _sibling_block). Without a sibling
    variant this whole exception has nothing to apply to. Included by
    default so an un-updated call site never silently drops it.
    """
    if _COMPOUND_HET_PROMPT_PATH.exists():
        return _COMPOUND_HET_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Compound-het-exception second-triage sub-prompt not found at {_COMPOUND_HET_PROMPT_PATH}."
    )


def _load_literature_evidence_quality_block() -> str:
    """
    Conditional sub-prompt (see prompts/second_triage_literature_evidence_quality.txt):
    the ANIMAL-MODEL-ONLY / GWAS-ASSOCIATION-STUDY evidence-shape sub-cases,
    spliced into {literature_evidence_quality_block} only when literature
    evidence was actually retrieved for this gene (litvar2_raw_by_variant[i]
    is non-empty). With no literature retrieved at all, the outer "no
    literature, no OMIM, no ClinVar disease entries" EXCLUDE ground already
    covers the case on its own — these two sub-cases exist to judge the
    SHAPE of literature that does exist, and have nothing to act on
    otherwise. Included by default so an un-updated call site never
    silently drops it.
    """
    if _LIT_QUALITY_PROMPT_PATH.exists():
        return _LIT_QUALITY_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Literature-evidence-quality second-triage sub-prompt not found at {_LIT_QUALITY_PROMPT_PATH}."
    )


# ── Score parser ──────────────────────────────────────────────────────────────

def parse_inclusion_decision(reasoning_text: str) -> tuple[str, str]:
    """
    Parse the INCLUDE/EXCLUDE decision and justification from run_one output.

    Expected embedded format (after the "SECOND TRIAGE:" marker):
        Include-case: <strongest reason to include, ≤15 words>
        Exclude-case: <strongest reason to exclude, ≤15 words>
        Decision: INCLUDE or EXCLUDE

    Blank lines are stripped before parsing (Qwen without thinking sometimes
    inserts them between lines).

    Returns:
        ("INCLUDE" | "EXCLUDE", justification)
        justification = "Include: <include_case> | Exclude: <exclude_case>"
        Defaults to ("INCLUDE", "parse error — defaulting to INCLUDE") on failure.
    """
    cleaned = "\n".join(line for line in reasoning_text.splitlines() if line.strip())

    include_m  = _INCLUDE_CASE_RE.search(cleaned)
    exclude_m  = _EXCLUDE_CASE_RE.search(cleaned)
    decision_m = _DECISION_RE.search(cleaned)

    include_str = include_m.group(1).strip() if include_m else ""
    exclude_str = exclude_m.group(1).strip() if exclude_m else ""

    logger.debug(
        "[SecondTriage] Include-case: %r | Exclude-case: %r | raw Decision line: %r",
        include_str, exclude_str,
        decision_m.group(0) if decision_m else "(not found)",
    )

    if decision_m:
        decision = decision_m.group(1).upper()
        justification = (
            f"Include: {include_str} | Exclude: {exclude_str}"
            if (include_str or exclude_str)
            else decision
        )
        return (decision, justification)

    # Fallback: model may have ignored the three-line format; search the tail
    tail = reasoning_text[-150:]
    m2 = re.search(r"\b(INCLUDE|EXCLUDE)\b", tail, re.IGNORECASE)
    if m2:
        logger.warning(
            "[SecondTriage] Decision: prefix not found — matched bare word in tail: %r",
            m2.group(0),
        )
        return (m2.group(1).upper(), f"Include: {include_str} | Exclude: {exclude_str}")

    logger.warning(
        "[SecondTriage] Could not parse INCLUDE/EXCLUDE decision — defaulting to INCLUDE. "
        "Raw tail: %r",
        reasoning_text[-200:],
    )
    return ("INCLUDE", "parse error — defaulting to INCLUDE")


# ── Main entry points ──────────────────────────────────────────────────────────

def run_reasoning(
    variant_context: str,
    llm: "LLMClient",
    sibling_context_block: str = "",
    inheritance_mode_block: str = "",
    gene_phenotype_block: str = "",
) -> str:
    """
    Stage 2a — Call 1: step-by-step clinical reasoning grounded in cited evidence only.
    The model is instructed not to introduce external knowledge without a reference
    present in the provided context.

    Split out from the former combined run_one() so the pipeline can run reasoning
    for every kept variant in a gene group before any of them proceeds to
    second_triage — needed so second_triage can see sibling variants' reasoning
    (compound-het safeguard for AR/XLR genes).

    Args:
        variant_context:         Per-variant context string from Stage 1 (retrieval).
        llm:                     Shared LLMClient instance.
        sibling_context_block:   Optional block describing other kept variants in the
                                 same gene (only populated for AR/XLR genes with ≥2
                                 kept variants and this variant heterozygous) — built
                                 from raw variant evidence (not sibling reasoning,
                                 which doesn't exist yet at Call 1) so this first
                                 reasoning pass already weighs the variant jointly
                                 with its gene-mate instead of in isolation. Empty
                                 string when not applicable.
        inheritance_mode_block:  Backend-determined gene inheritance mode (AD/AR/
                                 XLD/XLR/unknown), stated as a fact so the model
                                 reports it in the summary table rather than
                                 re-deriving it from prose evidence.
        gene_phenotype_block:    Backend-extracted gene condition list (see
                                 run_gene_phenotype_extraction), computed by an
                                 isolated call that never saw the patient's
                                 phenotype — stated as a fact so this step only
                                 has to compare it against the patient's
                                 phenotype for the CLUSTER_PHENOTYPE verdict,
                                 never author or edit the list itself.

    Returns:
        Step-by-step reasoning text.
    """
    logger.info("[Reasoning] Call 1/2 — step-by-step reasoning...")
    reasoning_template = _load_reasoning_prompt()
    reasoning_user     = (
        reasoning_template
        .replace("{augmented_context}", variant_context)
        .replace("{sibling_context_block}", sibling_context_block)
        .replace("{inheritance_mode_block}", inheritance_mode_block)
        .replace("{gene_phenotype_block}", gene_phenotype_block)
    )

    return llm.generate(
        system="You are an expert clinical geneticist.",
        user=reasoning_user,
        max_tokens=MAX_NEW_TOKENS_REASONING,
    )


def run_second_triage(
    variant_context: str,
    reasoning_text: str,
    llm: "LLMClient",
    sibling_context_block: str = "",
    inheritance_mode_block: str = "",
    is_x_linked: bool = False,
    include_single_hit_recessive: bool = True,
    include_compound_het_exception: bool = True,
    include_literature_evidence_quality: bool = True,
) -> str:
    """
    Stage 2b — Call 2: structured INCLUDE/EXCLUDE decision with justification.

    Args:
        variant_context:        Per-variant context string from Stage 1 (retrieval).
        reasoning_text:          Output of run_reasoning() for this variant.
        llm:                     Shared LLMClient instance.
        sibling_context_block:   Optional block describing other kept variants in
                                 the same gene (only populated for AR/AD-AR genes
                                 with ≥2 kept variants) — lets second_triage apply
                                 the compound-het safeguard instead of judging this
                                 variant in isolation. Empty string when not applicable.
        inheritance_mode_block:  Backend-determined gene inheritance mode (AD/AR/
                                 XLD/XLR/AD_AR/XLD_XLR/XL/UNDEFINED), same fact
                                 string already handed to run_reasoning() (Call 1)
                                 as `inheritance_mode_block` — re-passed here so
                                 Call 2 sees it directly instead of only through
                                 Call 1's own prose restatement of it. Without
                                 this, a gene's actual backend-resolved mode (or
                                 its UNDEFINED/unresolved status) was only visible
                                 to second_triage secondhand, filtered through
                                 whatever Call 1 chose to say about it — and a
                                 conflicting inheritance-mode claim surfacing
                                 elsewhere in the evidence text (e.g. a websearch
                                 or literature summary asserting a different mode
                                 than the backend CSV/CGD/LLM-resolved one) could
                                 win out in Call 2's own reasoning, since Call 2
                                 had no independent, authoritative fact to check
                                 it against. See prompts/second_triage.txt for how
                                 this block is used as the sole authoritative
                                 mode source in the zygosity-vs-inheritance rule.
        is_x_linked:             True when this variant's gene resolved to chrX
                                 (moi.gene_chromosome(variants, [i]) == "X") —
                                 splices in the X-LINKED GENES SPECIFICALLY
                                 sub-prompt (prompts/second_triage_xlinked.txt)
                                 only for genes it can actually apply to; a
                                 hard biological fact, never a judgment call,
                                 so this is conditional prompt assembly, not a
                                 weakened rule for the autosomal majority.
        include_single_hit_recessive: False only when Zygosity is already
                                 confirmed homozygous/hemizygous, or the
                                 gene's resolved mode is confirmed purely
                                 dominant (AD/XLD) — both cases where the
                                 "single het insufficient under a recessive
                                 model" ground is structurally inapplicable.
                                 Defaults True (include) so an un-updated or
                                 future call site never silently drops a
                                 ground that applies to most heterozygous
                                 variants, unlike is_x_linked's minority-case
                                 default of False.
        include_compound_het_exception: False only when sibling_context_block
                                 is empty (no sibling variant exists for this
                                 gene) — the exception has nothing to apply to
                                 in that case. Defaults True.
        include_literature_evidence_quality: False only when no literature
                                 evidence was retrieved at all for this gene —
                                 the outer "gene-phenotype link absent"
                                 EXCLUDE ground already covers that case, and
                                 these evidence-SHAPE sub-cases have nothing
                                 to characterize without literature to look
                                 at. Defaults True.

    Returns:
        Combined text: step-by-step reasoning followed by the inclusion decision
        on its own line. Compatible with parse_inclusion_decision() and with the
        REASONING display section in pipeline.py.
    """
    logger.info("[Reasoning] Call 2/2 — second_triage inclusion decision (INCLUDE/EXCLUDE)...")
    scoring_template   = _load_scoring_prompt()
    xlinked_block      = _load_xlinked_block() if is_x_linked else ""
    single_hit_block   = _load_single_hit_recessive_block() if include_single_hit_recessive else ""
    compound_het_block = _load_compound_het_exception_block() if include_compound_het_exception else ""
    lit_quality_block  = _load_literature_evidence_quality_block() if include_literature_evidence_quality else ""
    scoring_user     = (
        scoring_template
        .replace("{augmented_context}", variant_context)
        .replace("{reasoning}", reasoning_text)
        .replace("{sibling_context_block}", sibling_context_block)
        .replace("{inheritance_mode_block}", inheritance_mode_block)
        .replace("{compound_het_exception_block}", compound_het_block)
        .replace("{literature_evidence_quality_block}", lit_quality_block)
        .replace("{single_hit_recessive_block}", single_hit_block)
        .replace("{xlinked_block}", xlinked_block)
    )

    decision_text = llm.generate(
        system="You are an expert clinical geneticist.",
        user=scoring_user,
        max_tokens=MAX_NEW_TOKENS_SCORING,
    )

    logger.debug("[SecondTriage] Full model output:\n%s", decision_text.strip())

    # parse_inclusion_decision() searches for the Decision: line within the
    # "SECOND TRIAGE:" block.  The full three-line output is preserved so the
    # REASONING display section shows Include-case / Exclude-case / Decision.
    combined = reasoning_text.rstrip() + "\n\nSECOND TRIAGE:\n" + decision_text.strip()
    return combined

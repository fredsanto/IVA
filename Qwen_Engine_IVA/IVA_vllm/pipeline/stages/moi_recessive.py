"""
pipeline/stages/moi_recessive.py — MOI Layer 5: recessive / compound-het analysis.

For gene groups with >=2 kept variants and a recessive-relevant inheritance
mode (AR, XLR, AD_AR, XLD_XLR), where the TRANS phase has already been
confirmed (or left unknown-but-permitted) by the deterministic Python gate in
pipeline.py BEFORE this stage ever runs — a CIS pair never reaches here. Adds
PM3 (in-trans with a pathogenic/likely-pathogenic variant) on top of each
variant's Layer 2 base ACMG score — never re-scores base criteria.

Prompt loaded from prompts/moi_recessive.txt.

Public API:
    run_pair(gene, variant_a_label, variant_a_context, variant_a_base_conclusion,
              variant_b_label, variant_b_context, variant_b_base_conclusion,
              phase, cross_analysis, llm) -> str
    run_solo(gene, variant_label, variant_context, variant_base_conclusion,
              cross_analysis, llm) -> str
        For a CONFIRMED HOMOZYGOUS variant in a recessive-relevant gene with no
        compound-het partner — homozygosity alone satisfies the biallelic
        requirement, so this variant belongs in the Recessive layer even
        though it never goes through run_pair()'s >=2-variant TRANS gate.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.core.citations import validate_citations
from pipeline.core.clinvar_reference import append_clinvar_reference, append_clinvar_references
from pipeline.core.acmg_points import extract_base_acmg, resync_moi_total, splice_base_and_total
from pipeline.core.acmg_bs2_recessive import validate_bs2_homozygous_unaffected_parent
from pipeline.stages.conclusion import _own_identity

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "moi_recessive.txt"
_SOLO_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "moi_recessive_homozygous.txt"

MAX_NEW_TOKENS_RECESSIVE = 1800
MAX_NEW_TOKENS_RECESSIVE_SOLO = 600

# A compound-het pair is only a confirmed P/LP biallelic diagnosis if BOTH
# variants independently clear the Likely Pathogenic threshold — one strong
# variant does not rescue a weak/VUS partner (the partner's own base score
# already reflects that it does not, on its own, explain a recessive disease).
# Computed deterministically here (not left to the LLM) because the SLM
# repeatedly let a single >=6 partner carry the whole pair as "causative"
# during final synthesis, ignoring the weak partner entirely.
_CAUSATIVE_THRESHOLD = 6.0

_TOTAL_POINTS_RE = re.compile(
    r"\*\*Total ACMG points:\*\*\s*([-+]?\d+(?:\.\d+)?)\s*→"
)

# End of one variant's section: the next "## Variant" header, or the
# "**Comment:**" line that follows both variants — whichever comes first.
_SECTION_END_RE = re.compile(r"\n##\s*Variant|\n\*\*Comment:\*\*")


def _inject_base_and_total(pair_block: str, marker: str, base_conclusion: str) -> str:
    """
    Runs acmg_points.splice_base_and_total on this variant's own section of
    the pair block (from its "## Variant A/B" marker to the next variant
    header or the joint "**Comment:**") — the same base splice + Total every
    MOI layer uses. No-op (with a warning) if the marker isn't found.
    """
    start = pair_block.find(marker)
    if start == -1:
        logger.warning(
            "[MOIRecessive] Marker %r not found — could not inject base ACMG block.",
            marker,
        )
        return pair_block
    end_m = _SECTION_END_RE.search(pair_block, start + len(marker))
    section_end = end_m.start() if end_m else len(pair_block)
    section = splice_base_and_total(pair_block[start:section_end], base_conclusion)
    return pair_block[:start] + section + pair_block[section_end:]


def _force_pair_pm3(pair_block: str, marker: str, partner_base_conclusion: str, phase: str) -> str:
    """
    Sets this variant's compound-het PM3 deterministically instead of leaving
    it to the SLM, which wrote +1 under confirmed trans with an LP partner and
    +2 with a VUS partner, despite the prompt's rule. PM3 [Moderate, +2] when
    the partner's own Stage-4 base score is >= the Likely Pathogenic
    threshold — firm under TRANS confirmed, "hypothesis" under unknown phase
    (resync_moi_total then labels a P/LP total "Potential"). Partner below
    the threshold: +0. Uses the partner's BASE (not its layer Total) so the
    two variants' PM3 never depend on each other. No-op, with a warning, if
    the section or its criteria/delta lines aren't found.
    """
    start = pair_block.find(marker)
    if start == -1:
        logger.warning("[MOIRecessive] Marker %r not found — PM3 weight not enforced.", marker)
        return pair_block
    end_m = _SECTION_END_RE.search(pair_block, start + len(marker))
    section_end = end_m.start() if end_m else len(pair_block)
    section = pair_block[start:section_end]
    if not _SOLO_CRITERIA_LINE_RE.search(section) or not _SOLO_DELTA_LINE_RE.search(section):
        logger.warning("[MOIRecessive] %s PM3/delta lines not found — PM3 weight not enforced.", marker)
        return pair_block

    extracted = extract_base_acmg(partner_base_conclusion)
    partner_base = extracted[1] if extracted else None
    if partner_base is None or partner_base < _CAUSATIVE_THRESHOLD:
        shown = "unavailable" if partner_base is None else f"{partner_base:g} pts"
        criteria = (f"**Recessive criteria applied:** None — partner does not meet Likely "
                    f"Pathogenic threshold (partner base score: {shown}).")
        delta = "**Recessive delta:** +0"
    elif phase == "trans":
        criteria = (f"**Recessive criteria applied:** PM3 [Moderate, +2] — detected in trans "
                    f"(different parents) with a Likely Pathogenic or Pathogenic partner "
                    f"(partner base score: {partner_base:g} pts).")
        delta = "**Recessive delta:** +2 confirmed"
    elif phase == "denovo":
        criteria = (f"**Recessive criteria applied:** PM3 [Moderate, +2] — compound heterozygous "
                    f"assumed with a Likely Pathogenic or Pathogenic partner (partner base score: "
                    f"{partner_base:g} pts); one allele is de novo, so trans is assumed, not "
                    f"shown — PHASE MUST BE CHECKED (parental testing cannot phase a de novo "
                    f"allele; long-read sequencing or cloning needed).")
        delta = "**Recessive delta:** +2 (trans assumed — phase must be checked)"
    else:
        criteria = (f"**Recessive criteria applied:** PM3 (hypothesis — trans phase not "
                    f"confirmed) [Moderate, +2] — partner is Likely Pathogenic or Pathogenic "
                    f"(partner base score: {partner_base:g} pts); parental testing needed to "
                    f"confirm trans.")
        delta = "**Recessive delta:** +2 hypothesis"
    section = _SOLO_CRITERIA_LINE_RE.sub(lambda _m: criteria, section, count=1)
    section = _SOLO_DELTA_LINE_RE.sub(lambda _m: delta, section, count=1)
    return pair_block[:start] + section + pair_block[section_end:]


def _joint_compound_het_status(pair_block: str) -> str | None:
    """
    Parse both variants' own "**Total ACMG points:** N →" lines (in Variant A,
    Variant B order, per the fixed template below) and deterministically decide
    whether this pair is a confirmed compound-het P/LP diagnosis or a compound
    VUS. Returns None (no-op) if the expected two totals can't be found, so a
    malformed LLM response degrades gracefully instead of raising.
    """
    matches = _TOTAL_POINTS_RE.findall(pair_block)
    if len(matches) < 2:
        logger.warning(
            "[MOIRecessive] Could not find two 'Total ACMG points' lines in pair "
            "block — skipping joint compound-het classification."
        )
        return None
    try:
        points_a, points_b = float(matches[0]), float(matches[1])
    except ValueError:
        return None

    if points_a >= _CAUSATIVE_THRESHOLD and points_b >= _CAUSATIVE_THRESHOLD:
        return (
            f"CAUSATIVE — compound heterozygous Pathogenic/Likely Pathogenic. "
            f"Both alleles independently reach the Likely Pathogenic threshold "
            f"(Variant A: {points_a:g} pts, Variant B: {points_b:g} pts; both >= "
            f"{_CAUSATIVE_THRESHOLD:g})."
        )
    return (
        f"COMPOUND VUS — NOT a confirmed causative biallelic diagnosis. "
        f"At least one allele is below the Likely Pathogenic threshold "
        f"(Variant A: {points_a:g} pts, Variant B: {points_b:g} pts; both must "
        f"independently reach >= {_CAUSATIVE_THRESHOLD:g}). Report this gene as "
        f"compound VUS, not as Pathogenic/Likely Pathogenic, even though one "
        f"partner individually clears the threshold — a single strong allele "
        f"does not establish a biallelic recessive diagnosis on its own."
    )


def _inject_joint_status(pair_block: str, joint_status: str | None) -> str:
    if joint_status is None:
        return pair_block
    marker = "\n## Variant A"
    if marker not in pair_block:
        logger.warning(
            "[MOIRecessive] '## Variant A' marker not found — could not inject "
            "joint compound-het classification line."
        )
        return pair_block
    return pair_block.replace(
        marker,
        f"\n**Joint compound-het classification:** {joint_status}\n{marker}",
        1,
    )

_PHASE_LABELS = {
    "trans":   "TRANS confirmed (different parents) — compound heterozygous model applies.",
    "denovo":  "ASSUMED TRANS (one variant is de novo; parental testing cannot phase it) — "
               "compound heterozygous model applies, but state that phase must be checked.",
    "unknown": "UNKNOWN (phase not determinable from available allelic-balance data) — "
               "compound heterozygous model may still apply per biallelic evidence, "
               "but phase uncertainty must be noted explicitly.",
}

# classify_segregation() outputs (pipeline.core.segregation), relabelled for the
# homozygous-solo PM3 decision. "homozygous_parent" is the hard negative gate —
# every other outcome is treated as "not contradicted" (PM3 may still apply at
# reduced weight; see prompts/moi_recessive_homozygous.txt for the point logic).
_SEGREGATION_LABELS = {
    "both_carriers": "Both parents confirmed heterozygous carriers by allelic balance "
                      "(~0.5 each), proband homozygous (~1.0) — classic autosomal "
                      "recessive segregation pattern. Not contradicted; PM3 may apply "
                      "per the rules below.",
    "homozygous_parent": "At least one parent's allelic balance is itself ~1.0 "
                      "(homozygous) for this variant — an \"unaffected\" carrier "
                      "parent should be heterozygous, not homozygous. Segregation is "
                      "DISCORDANT with simple biallelic transmission from two carrier "
                      "parents — do NOT apply PM3 (see rule 2a).",
    "insufficient_data": "No parental allelic-balance data available — segregation not "
                      "verified. Not contradicted; PM3 may still apply at reduced weight "
                      "per the rules below (parental carrier status assumed but unconfirmed).",
    "uncertain": "Parental allelic-balance data present but does not match a clean "
                      "carrier pattern — segregation ambiguous. Not contradicted; PM3 "
                      "may still apply at reduced weight per the rules below.",
}
_SEGREGATION_DEFAULT_LABEL = (
    "Segregation pattern does not match the expected two-carrier-parents pattern for "
    "a homozygous proband — treat as unconfirmed. Not contradicted; PM3 may still "
    "apply at reduced weight per the rules below."
)


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"moi_recessive prompt not found at {_PROMPT_PATH}.")


def _load_solo_prompt() -> str:
    if _SOLO_PROMPT_PATH.exists():
        return _SOLO_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"moi_recessive_homozygous prompt not found at {_SOLO_PROMPT_PATH}.")


def run_pair(
    gene: str,
    variant_a_label: str,
    variant_a_context: str,
    variant_a_base_conclusion: str,
    variant_b_label: str,
    variant_b_context: str,
    variant_b_base_conclusion: str,
    phase: str,
    cross_analysis: str | None,
    llm: "LLMClient",
) -> str:
    """
    MOI Layer 5 — recessive/compound-het analysis for one gene's variant pair.

    Args:
        gene:                          Gene symbol.
        variant_a_label, variant_b_label: Display labels (e.g. "Variant 3 — GENE_X (p.Lys34Thr)").
        variant_a_context, variant_b_context: Per-variant context strings from retrieval.
        variant_a_base_conclusion, variant_b_base_conclusion: Layer 2's full structured
                          output for each variant — ground truth this stage adds to.
        phase:            classify_phase() result — "trans", "denovo" or "unknown"; callers
                           must never call this with "cis" (that pair should have been
                           excluded before reaching this stage).
        cross_analysis:    Gene-level cross_analysis.run() text, or None if unavailable.
        llm:               Shared LLMClient instance.

    Returns:
        Structured recessive layer block covering both variants, each with its own
        "Total ACMG points: [base+delta] → [classification]".
    """
    if phase not in _PHASE_LABELS:
        raise ValueError(
            f"moi_recessive.run_pair called with phase={phase!r} for gene={gene} — "
            "only 'trans', 'denovo' or 'unknown' may reach this stage; 'cis' must be excluded upstream."
        )

    logger.info("[MOIRecessive] Generating recessive analysis for gene %s (phase=%s)...", gene, phase)

    template = _load_prompt()
    cross_analysis_block = (
        f"GENE-LEVEL CROSS-ANALYSIS:\n{cross_analysis}\n" if cross_analysis is not None else ""
    )

    user_prompt = (template
        .replace("{gene}", gene)
        .replace("{variant_a_label}", variant_a_label)
        .replace("{variant_a_base_conclusion}", variant_a_base_conclusion)
        .replace("{variant_b_label}", variant_b_label)
        .replace("{variant_b_base_conclusion}", variant_b_base_conclusion)
        .replace("{phase_block}", _PHASE_LABELS[phase])
        .replace("{cross_analysis_block}", cross_analysis_block))

    result = llm.generate(
        system="You are an expert clinical geneticist evaluating compound heterozygous/biallelic recessive inheritance. Limit your response to 500 words maximum.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_RECESSIVE,
    )
    # Splice in the deterministic, mechanically-copied Base ACMG block +
    # Total for each variant (see acmg_points.splice_base_and_total) — the LLM
    # is not trusted to transcribe or re-derive these numbers itself.
    # PM3 weight is set first so the splice's Total uses the enforced delta.
    result = _force_pair_pm3(result, "## Variant A", variant_b_base_conclusion, phase)
    result = _force_pair_pm3(result, "## Variant B", variant_a_base_conclusion, phase)
    result = _inject_base_and_total(result, "## Variant A", variant_a_base_conclusion)
    result = _inject_base_and_total(result, "## Variant B", variant_b_base_conclusion)
    full_context = variant_a_context + "\n" + variant_b_context + "\n" + variant_a_base_conclusion + "\n" + variant_b_base_conclusion
    result = validate_citations(result, full_context)
    joint_status = _joint_compound_het_status(result)
    logger.info("[MOIRecessive] Joint compound-het status for %s: %s", gene, joint_status)
    result = _inject_joint_status(result, joint_status)
    return append_clinvar_references(result, [
        (variant_a_label, variant_a_context + "\n" + variant_a_base_conclusion),
        (variant_b_label, variant_b_context + "\n" + variant_b_base_conclusion),
    ])


_SOLO_HEADER_RE = re.compile(r"(?m)^#\s*Recessive Analysis\b.*$")


def _force_solo_header_identity(result: str, gene: str, variant_context: str) -> str:
    """
    Rewrites the homozygous-solo "# Recessive Analysis — ..." header with this
    variant's own Gene/HGVS read from the input CSV (variant_context's Gene=/
    HGVS= fields), ending in the same "[GENE] ([HGVS])" shape as every other
    MOI layer's header; the "(confirmed homozygous, biallelic)" kind label
    downstream prompts key on is kept before the dash. The template header used to name only the gene, so the block
    carried no identity of its own — the only protein change inside it was
    the PM5/PS1 comparator quoted in the copied criteria, and final synthesis
    reported THAT comparator as the patient's variant. Same override as
    conclusion._force_variant_header_identity(), which only covers the Stage-4
    header and never reaches this layer block. Kept in the "GENE (detail)"
    shape final_conclusion._HEADER_LINE_RE parses.
    """
    true_gene, true_hgvs = _own_identity(variant_context)
    header = (f"# Recessive Analysis (confirmed homozygous, biallelic) — "
              f"{true_gene or gene} ({true_hgvs or 'HGVS unavailable'})")
    if not _SOLO_HEADER_RE.search(result):
        logger.warning("[MOIRecessive] Solo header not found for %s — prepending forced header.", gene)
        return header + "\n" + result
    return _SOLO_HEADER_RE.sub(lambda _m: header, result, count=1)


_SOLO_CRITERIA_LINE_RE = re.compile(r"(?m)^\*\*Recessive criteria applied:\*\*.*$")
_SOLO_DELTA_LINE_RE = re.compile(r"(?m)^\*\*Recessive delta:\*\*.*$")


def _force_homozygous_pm3(result: str, segregation: str) -> str:
    """
    Sets the homozygous PM3 weight deterministically instead of leaving it to
    the SLM. ClinGen SVI counts one homozygous affected proband as 0.5 PM3
    case points = PM3_Supporting, which on this pipeline's Tavtigian scale
    (Supporting = 1) is +1 — the old "+0.5" put the case-count value straight
    into the total. Consanguinity does not lower it: when unconfirmed it is
    assumed, and SVI's 0.5-per-homozygote weight already allows for
    identity-by-descent. The only exception is the backend's hard segregation
    gate — a parent who is themselves homozygous — which gives +0.
    resync_moi_total() then computes the Total from the delta line.
    """
    if segregation == "homozygous_parent":
        criteria = ("**Recessive criteria applied:** None — a parent is themselves "
                    "homozygous for this variant; segregation discordant with simple "
                    "biallelic transmission from two carrier parents.")
        delta = "**Recessive delta:** +0"
    else:
        criteria = ("**Recessive criteria applied:** PM3 [Supporting, +1] — homozygous "
                    "occurrence in an affected proband (ClinGen SVI: 0.5 PM3 points = "
                    "PM3_Supporting); consanguinity unconfirmed, assumed.")
        delta = "**Recessive delta:** +1"
    if not _SOLO_CRITERIA_LINE_RE.search(result) or not _SOLO_DELTA_LINE_RE.search(result):
        logger.warning("[MOIRecessive] Homozygous PM3/delta lines not found — PM3 weight not enforced.")
        return result
    result = _SOLO_CRITERIA_LINE_RE.sub(lambda _m: criteria, result, count=1)
    return _SOLO_DELTA_LINE_RE.sub(lambda _m: delta, result, count=1)


def run_solo(
    gene: str,
    variant_label: str,
    variant_context: str,
    variant_base_conclusion: str,
    segregation: str,
    cross_analysis: str | None,
    llm: "LLMClient",
) -> str:
    """
    MOI Layer 5 (solo path) — confirmed-homozygous recessive analysis for one
    variant with no compound-het partner.

    Args:
        gene:                     Gene symbol.
        variant_label:            Display label (e.g. "Variant 3 — GENE_X (p.Lys34Thr)").
        variant_context:          Per-variant context string from retrieval.
        variant_base_conclusion:  Layer 2's full structured output — ground truth this
                                   stage documents against (does not re-score it).
        segregation:              pipeline.core.segregation.classify_segregation() result
                                   for this variant's trio AB data — a hard Python gate on
                                   PM3 eligibility (see _SEGREGATION_LABELS): a parent who
                                   is themselves homozygous blocks PM3 outright.
        cross_analysis:           Gene-level cross_analysis.run() text, or None if unavailable.
        llm:                      Shared LLMClient instance.

    Returns:
        Structured recessive layer block for a single confirmed-homozygous variant,
        with "Total ACMG points: [base + PM3 delta] → [classification]" where the PM3
        delta is +0, +0.5 (consanguinity not excluded), or +1 (consanguinity explicitly
        excluded in evidence) per ClinGen SVI's homozygous-PM3 guidance.
    """
    logger.info("[MOIRecessive] Generating homozygous-solo recessive analysis for gene %s...", gene)

    template = _load_solo_prompt()
    cross_analysis_block = (
        f"GENE-LEVEL CROSS-ANALYSIS:\n{cross_analysis}\n" if cross_analysis is not None else ""
    )
    segregation_block = _SEGREGATION_LABELS.get(segregation, _SEGREGATION_DEFAULT_LABEL)

    user_prompt = (template
        .replace("{gene}", gene)
        .replace("{variant_label}", variant_label)
        .replace("{variant_base_conclusion}", variant_base_conclusion)
        .replace("{segregation_block}", segregation_block)
        .replace("{cross_analysis_block}", cross_analysis_block))

    result = llm.generate(
        system="You are an expert clinical geneticist evaluating a confirmed homozygous recessive variant. Limit your response to 400 words maximum.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_RECESSIVE_SOLO,
    )
    # Mechanical DEFAULT-UNAFFECTED-POLICY BS2 check, before resync so the
    # inserted bullet gets folded into the Base/Total point lines below — the
    # prompt already neutralizes PM3 for a homozygous-parent discordance but
    # never applies the BS2 penalty that discordance itself is evidence for.
    # See validate_bs2_homozygous_unaffected_parent's docstring.
    result = _force_solo_header_identity(result, gene, variant_context)
    result = _force_homozygous_pm3(result, segregation)
    # Same as every MOI layer: base block spliced from Stage 4 by code, then
    # (after BS2) Total = base + delta.
    result = splice_base_and_total(result, variant_base_conclusion)
    result = validate_bs2_homozygous_unaffected_parent(result, variant_base_conclusion, segregation)
    result = resync_moi_total(result)
    full_context = variant_context + "\n" + variant_base_conclusion
    result = validate_citations(result, full_context)
    return append_clinvar_reference(result, full_context)

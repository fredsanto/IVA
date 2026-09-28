"""
pipeline/stages/conclusion.py — SLM structured report stage.

Ported from run_conclusion() in server_main.py.

Single SLM call per variant. Takes one variant's context string, its reasoning
output, and an optional gene-level cross-analysis text, then produces the
structured clinical block for that variant.

The Clinical Conclusion paragraph (overall synthesis across all variants) is
generated separately by stages/final_conclusion.py.

Prompt loaded from prompts/conclusion.txt.

Public API:
    run_one(variant_context, reasoning, cross_analysis, llm) -> str
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.core.errors import SLMError
from pipeline.core.citations import validate_citations
from pipeline.core.acmg_pp3 import validate_pp3
from pipeline.core.acmg_pp2_bp1 import validate_pp2_bp1
from pipeline.core.acmg_ps1_pm5 import validate_ps1_pm5, strip_ungrounded_ps1_pm5
from pipeline.core.acmg_ps3 import validate_ps3, block_ps3_under_pvs1
from pipeline.core.acmg_pm1 import validate_pm1
from pipeline.core.acmg_pm2 import validate_pm2
from pipeline.core.acmg_bp6 import validate_bp6
from pipeline.core.acmg_pp4 import validate_pp4, validate_pp4_full_coverage
from pipeline.core.acmg_pvs1 import validate_pvs1
from pipeline.core.acmg_points import relabel_all_points_lines, recompute_and_fix_totals

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "conclusion.txt"

MAX_NEW_TOKENS_REPORT = 1500

# Scoped to the Gene=/HGVS= fields of variant_context specifically (the
# key=value line retrieval.py's _variant_dict_to_str builds from the input
# CSV's own normalized columns), not the whole context blob — which also
# contains OTHER variants' HGVS (CLINVAR RESIDUE-LEVEL SEARCH comparators,
# other genes' evidence blocks). Same field-scoping lesson as
# acmg_ps1_pm5.py's _HGVS_FIELD_RE/_own_cdnas(), which this mirrors.
_GENE_FIELD_RE = re.compile(r"Gene=(.*?)(?:,\s*OMIM_phenotype=|\n)")
_HGVS_FIELD_RE = re.compile(r"HGVS=(.*?)(?:,\s*Zygosity=|\n)")

# The literal "# Variant [N] — [GENE] ([HGVS])" header format prompts/conclusion.txt
# instructs the SLM to write (see the format spec below run_one's return in
# its own docstring). Tolerant of an em-dash or hyphen separator. The HGVS
# group is greedy (not `[^)]*`) because the HGVS itself nests parens (e.g.
# "p.(Lys34Thr)") — a non-greedy/exclusion capture would stop at that
# inner ')' and leave the real outer ')' dangling in the untouched tail.
# Greedy + line-anchored (`.` excludes '\n') naturally lands on the last
# ')' on the header line, which is the real outer closing paren.
_VARIANT_HEADER_RE = re.compile(r"(?m)^(#\s*Variant\s+\d+\s*[—-]\s*)(\S+)(\s*\((.*)\))?$")


def _own_identity(variant_context: str) -> tuple[str | None, str | None]:
    """Ground-truth Gene and HGVS for this variant, read straight from the
    input CSV via variant_context's own Gene=/HGVS= fields — never from
    anything the SLM wrote."""
    gene_m = _GENE_FIELD_RE.search(variant_context)
    hgvs_m = _HGVS_FIELD_RE.search(variant_context)
    gene = gene_m.group(1).strip() if gene_m else None
    hgvs = hgvs_m.group(1).strip() if hgvs_m else None
    return gene, hgvs


def _force_variant_header_identity(result: str, variant_context: str) -> str:
    """
    Force-substitutes this variant's own Gene/HGVS (ground truth from the
    input CSV) into the "# Variant [N] — [GENE] ([HGVS])" header the SLM
    free-writes, instead of trusting whatever it wrote there.

    Real observed failure: the SLM's PS1/PM5 evidence-gathering surfaces a
    same-amino-acid/different-nucleotide ClinVar comparator variant (exactly
    the evidence PS1 needs) into the same context the header is drawn from,
    and the header ends up citing THAT comparator's nucleotide change
    instead of the submitted variant's own — gene and protein change
    correct, nucleotide wrong (e.g. a submitted c.100A>C reported back as
    c.100A>G; another submitted c.200G>A reported back as an unrelated
    c.300C>T). Every downstream MOI layer (moi_denovo.py, moi_xlinked.py,
    moi_recessive.py, moi_dominant.py) takes this header's block as its
    frozen "base_conclusion" ground truth, and prompts/clinical_conclusion.txt
    is explicitly told to copy it "character-for-character" into the final
    report — so an error here propagates to "2) Causative variant(s):"
    unchanged. This is the one point where the true value is unambiguous
    (variant_context's own Gene=/HGVS= fields), so it is forced here rather
    than trusted from the SLM, the same mechanical-override pattern
    acmg_points.extract_base_acmg() and moi_recessive.py's
    _inject_base_and_total() already use for other frozen facts.

    No-op (with a warning) if the header pattern can't be found — degrades
    to whatever the SLM produced rather than corrupting the block further.
    """
    true_gene, true_hgvs = _own_identity(variant_context)
    if true_gene is None and true_hgvs is None:
        return result

    m = _VARIANT_HEADER_RE.search(result)
    if not m:
        logger.warning(
            "[Conclusion] Could not locate '# Variant N — GENE (HGVS)' header "
            "to enforce true variant identity — leaving SLM output as-is."
        )
        return result

    stated_gene = m.group(2)
    stated_hgvs = (m.group(4) or "").strip()
    if true_gene and stated_gene != true_gene:
        logger.warning(
            "[Conclusion] Header gene mismatch: SLM wrote %r, input CSV says %r — correcting.",
            stated_gene, true_gene,
        )
    if true_hgvs and stated_hgvs != true_hgvs:
        logger.warning(
            "[Conclusion] Header HGVS mismatch: SLM wrote %r, input CSV says %r — correcting.",
            stated_hgvs, true_hgvs,
        )

    new_gene = true_gene or stated_gene
    new_hgvs = true_hgvs if true_hgvs else stated_hgvs
    new_header = f"{m.group(1)}{new_gene} ({new_hgvs})" if new_hgvs else f"{m.group(1)}{new_gene}"
    return result[: m.start()] + new_header + result[m.end() :]


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Conclusion prompt not found at {_PROMPT_PATH}. "
        "Run step 19 to extract prompts from server_main.py."
    )


def _has_acmg_criteria_section(text: str) -> bool:
    """False when the response is missing its "ACMG criteria:" heading —
    the prompt (see prompts/conclusion.txt) always requires this heading,
    even for a variant with zero applicable criteria ("None apply" is still
    written under it). Its absence means the SLM response was truncated or
    otherwise malformed before reaching that section, not that the model
    legitimately found nothing — a real observed failure: a large,
    uncompressed GeneReviews chapter for the variant's gene bloated the
    prompt enough that MAX_NEW_TOKENS_REPORT ran out mid-response, silently
    dropping the ACMG criteria (and total-points) lines. Downstream MOI
    layers and acmg_points.recompute_and_fix_totals() have no criteria
    bullets to sum in that case and silently leave the SLM's own broken
    "[Base score + N] -> [...]" placeholder text untouched — the variant
    then vanishes from the final Clinical Conclusion with no error anywhere.
    Failing loudly here, before that silent chain starts, is the fix."""
    return "ACMG criteria" in text


def run_one(
    variant_context: str,
    reasoning: str,
    cross_analysis: str | None,
    llm: "LLMClient",
) -> str:
    """
    Stage 4 — Structured clinical report for one variant.

    Args:
        variant_context: Per-variant context string from Stage 1 (retrieval).
                         Contains PATIENT DATA header + one VARIANT block with tool outputs.
        reasoning:       Output from reasoning.run_one() for this variant.
        cross_analysis:  Gene-level cross-analysis text from cross_analysis.run(), or None
                         if this variant's gene has fewer than two variants in this run.
        llm:             Shared LLMClient instance.

    Returns:
        Structured report block for this variant:
            # Variant [N] — [GENE] ([HGVS])
            **Molecular mechanism:** ...
            **Phenotype fit:** ...
            **Inheritance check:** ...
            **Evidence strength:** ...
            **ACMG criteria**: ...
            **Comment:** ...
    """
    logger.info("[Conclusion] Generating structured report for variant...")

    template = _load_prompt()

    cross_analysis_block = (
        f"GENE-LEVEL CROSS-ANALYSIS:\n{cross_analysis}\n"
        if cross_analysis is not None else ""
    )

    user_prompt = (template
        .replace("{augmented_context}", variant_context)
        .replace("{reasoning_output}", reasoning)
        .replace("{cross_analysis_block}", cross_analysis_block))

    result = llm.generate(
        system="You are an expert clinical geneticist. Limit your response to 1000 words maximum.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_REPORT,
    )
    if not _has_acmg_criteria_section(result):
        raise SLMError(
            "Conclusion response missing required 'ACMG criteria' section "
            f"(likely truncated — response ended with: ...{result[-200:]!r})"
        )
    result = _force_variant_header_identity(result, variant_context)
    result = validate_citations(result, variant_context)
    result = validate_pp3(result, variant_context)
    result = validate_pp2_bp1(result, variant_context)
    # Strip any PS1/PM5 not grounded in a qualifying residue-search hit
    # (circular own-record PS1, uncited, non-missense variant or hit)
    # BEFORE the adder below runs, so a genuinely-missing-but-warranted
    # PS1/PM5 can still be filled in from a real hit rather than being
    # blocked by the invalid one already sitting in the text.
    result = strip_ungrounded_ps1_pm5(result, variant_context)
    result = validate_ps1_pm5(result, variant_context)
    # PM1: kept/added only on a MET ClinVar hotspot window verdict.
    result = validate_pm1(result, variant_context)
    # PM2: added (never stripped) when gnomAD/CSV allele frequency < 1e-4.
    result = validate_pm2(result, variant_context)
    # PS3: kept/added only when citing a tagged PMID (ClinVar submitter
    # functional work with a reference, or LitVar2 variant-level functional
    # evidence); any other PS3 is stripped. Independent of whether PS1 applies.
    result = validate_ps3(result, variant_context)
    result = validate_bp6(result)
    result = validate_pp4(result)
    # reasoning (not reasoning_output/variant_context) carries the
    # backend-injected "PHENOTYPE CLUSTER MATCH" verdict block pipeline.py's
    # _cluster_match_block() appended — the authoritative source for PP4's
    # FULL COVERAGE condition, checked independently of whatever the model's
    # own prose in `result` claims about phenotype fit.
    result = validate_pp4_full_coverage(result, reasoning)
    result = validate_pvs1(result, variant_context)
    # PS3 never stacks on PVS1 (functional loss is what PVS1 already scores).
    result = block_ps3_under_pvs1(result)
    # Unconditional final pass: the validators above only adjust the stated
    # total when THEY strip a criterion. They never check whether the SLM's
    # own original total already matched its own criteria list — a real,
    # recurring observed failure: a list totalling PS3(+4)+PM2(+2)+PP4(+1) =
    # 7 stated as "ACMG points: 11", with nothing here to catch it since no
    # criterion needed stripping. Re-sum every points/classification line
    # from its own preceding criteria bullets and correct the stated total
    # (and label) whenever it disagrees.
    result = recompute_and_fix_totals(result)
    # Final safety net: re-derive any remaining line's label from its own
    # stated N (e.g. "4.5 -> Likely Pathogenic" when 4.5 is in the 0-5 VUS
    # band, not 6-9) — covers the case recompute_and_fix_totals() left
    # untouched (no criteria bullets found to check against).
    return relabel_all_points_lines(result)

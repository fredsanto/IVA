"""
pipeline/core/build_detection.py — auto-detect genome build (hg19 vs hg38)
from a sample of input variants' resolvable genomic coordinates.

Real-world CSV uploads mix hg19- and hg38-coordinated data interchangeably,
and DEFAULT_GENOME_BUILD ("hg19") is only ever a static fallback — trusting
it blindly silently corrupts every coordinate-dependent tool that has no
per-request build fallback of its own (SpliceAI: queries the wrong genome's
API endpoint with right-looking coordinates, returning wrong/garbage results
without erroring). ClinGenAlleleTool already tries both builds per variant
and picks whichever resolves cleanly — this module runs that same check once
(over a small sample) at pipeline start so the DETECTED build can be used as
the single source of truth for the whole run, not just within ClinGen calls.
"""

import logging
from collections import Counter

from pipeline.config import DEFAULT_GENOME_BUILD
from pipeline.tools.autopvs1 import parse_variant_coords
from pipeline.tools.clingen_allele import (
    ClinGenAlleleTool,
    _HG19_CHROM_TO_NC,
    _HG38_CHROM_TO_NC,
)

logger = logging.getLogger(__name__)

_SAMPLE_SIZE = 5


def _clingen_genomic_match(data: dict | None, expected_hgvs: str) -> bool:
    """True when ClinGen response contains the expected genomic HGVS string
    in its genomicAlleles list — meaning the variant was recognized at those
    coordinates for that reference genome (transcript-level resolution).

    This replaces the old _resolved_cdna + cDNA-matching approach which failed
    when the variant's HGVS used a non-canonical transcript (different cDNA
    numbering) than the MANE transcript ClinGen resolves to — causing false
    negatives even when the build was correct.
    """
    if not data:
        return False
    expected = expected_hgvs.strip().lower()
    for ga in data.get("genomicAlleles", []):
        for hgvs in ga.get("hgvs", []):
            if hgvs.strip().lower() == expected:
                return True
    return False


def detect_genome_build(variants: list[dict]) -> str:
    """
    Returns "hg19" or "hg38" — majority vote across up to _SAMPLE_SIZE
    variants that have a resolvable single-base SNV coordinate.

    A build "wins" a given variant's vote when ClinGen's `genomicAlleles`
    list contains the exact genomic HGVS query string for that build —
    confirming the variant was recognized at those coordinates in that
    reference genome. This is deliberately NOT based on whether the ClinGen
    query merely succeeds vs. errors — the Allele Registry auto-registers
    any syntactically valid HGVS as a "novel" allele rather than cleanly
    rejecting a wrong-build query, so a wrong-build query for a gene spanning
    a large genomic footprint can still resolve "successfully" to some OTHER
    (irrelevant, e.g. deep-intronic) position within the same gene — a real
    observed case, not hypothetical. Checking that the exact query string
    appears in `genomicAlleles` is immune to this: a wrong-build query at a
    position whose reference happens to match the declared ref (triggering
    auto-register) would still not contain the EXACT same query string in its
    `genomicAlleles` list — ClinGen generates a fresh entry with its own
    coordinates, not the queried ones. This also avoids the previous approach's
    false-negative failure when the variant's HGVS uses a non-canonical
    transcript (different cDNA numbering) than the MANE transcript ClinGen
    resolves to.

    Falls back to DEFAULT_GENOME_BUILD (with a logged warning) when no
    sampled variant yields a clear signal — e.g. an indel-only CSV with no
    resolvable single-base SNVs, a gene absent from ClinGen's index, or a
    transient API issue on both queries.
    """
    tool = ClinGenAlleleTool()
    votes: Counter = Counter()
    sampled = 0

    for variant in variants:
        if sampled >= _SAMPLE_SIZE:
            break

        try:
            coords = parse_variant_coords(
                variant_str=variant.get("Variant", ""),
                chrom_field=variant.get("Chromosome", ""),
                pos_field=variant.get("Position", ""),
                ref_field=variant.get("Ref_seq", ""),
                alt_field=variant.get("Var_seq", ""),
            )
        except (ValueError, AttributeError, TypeError):
            # Sampling loop — unparseable candidates are expected (indels,
            # missing fields, or a non-string Variant field slipping through
            # as a bare float NaN) and simply don't count toward the vote.
            # A real observed failure: an unmapped "Variant" field reaching
            # here as a float (not "NA") raised AttributeError on .strip(),
            # which escaped this loop uncaught and silently killed the whole
            # background pipeline task (see server_qwen.py's fire-and-forget
            # asyncio.create_task(_run_direct(...)) — nothing awaited it, so
            # the job just hung forever with no error ever surfaced).
            continue
        chrom, pos, ref, alt = coords
        if len(ref) != 1 or len(alt) != 1 or ref in ("-", "") or alt in ("-", ""):
            continue

        chrom_key = chrom.strip().upper().removeprefix("CHR")
        nc19 = _HG19_CHROM_TO_NC.get(chrom_key)
        nc38 = _HG38_CHROM_TO_NC.get(chrom_key)
        if not nc19 or not nc38:
            continue

        sampled += 1
        query19 = f"{nc19}:g.{pos}{ref.upper()}>{alt.upper()}"
        query38 = f"{nc38}:g.{pos}{ref.upper()}>{alt.upper()}"
        data19, _ = tool._fetch(query19)
        data38, _ = tool._fetch(query38)

        match19 = _clingen_genomic_match(data19, query19)
        match38 = _clingen_genomic_match(data38, query38)

        if match38 and not match19:
            votes["hg38"] += 1
        elif match19 and not match38:
            votes["hg19"] += 1
        # Both match, neither matches, or one/both unresolvable: uninformative
        # — no vote cast, sample doesn't count toward the decision either way.

    if not votes:
        logger.warning(
            "[build_detection] No clear build signal from %d sampled variant(s) — "
            "falling back to DEFAULT_GENOME_BUILD=%r",
            sampled, DEFAULT_GENOME_BUILD,
        )
        return DEFAULT_GENOME_BUILD

    detected, n_votes = votes.most_common(1)[0]
    logger.info(
        "[build_detection] Detected genome_build=%r (%d/%d sampled variant(s) agreed; votes=%s)",
        detected, n_votes, sampled, dict(votes),
    )
    return detected

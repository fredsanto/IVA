"""
pipeline/core/acmg_bs2_recessive.py — mechanically enforces BS2's
DEFAULT-UNAFFECTED POLICY (prompts/conclusion.txt) for the confirmed-
homozygous recessive solo path, mirroring acmg_bs2_dominant.py's fix for
the equivalent dominant-layer gap.

moi_recessive.run_solo()'s own prompt (moi_recessive_homozygous.txt) already
detects the discordant case where a parent is THEMSELVES homozygous for the
variant — an "unaffected" carrier parent of a recessive condition should be
heterozygous, not homozygous — but only neutralizes PM3 for it (Recessive
delta: +0). A parent homozygous for a variant that is supposedly a fully
penetrant recessive cause, and not flagged anywhere in the pipeline as
clinically affected, is itself Strong Benign evidence (BS2) exactly like the
dominant-layer unaffected-carrier case — the pipeline has no channel to mark
a parent as affected in this recessive-solo path (segregation here is
allelic-balance-only), so "homozygous_parent" always means an unaffected
homozygous carrier by construction, no additional affected-parent check
needed before applying BS2.
"""

import re

_BS2_ALREADY_PRESENT_RE = re.compile(r"^-\s*BS2\b", re.MULTILINE)

# The base criteria list spliced into this layer's block by
# acmg_points.splice_base_and_total — a contiguous run of
# "- CRITERION [..]: ..." bullet lines directly following its header.
_BASE_CRITERIA_BLOCK_RE = re.compile(r"(\*\*Base ACMG criteria\b[^\n]*\*\*\n(?:-[^\n]*\n?)+)")


def validate_bs2_homozygous_unaffected_parent(
    result: str, base_conclusion: str, segregation: str
) -> str:
    """
    Appends a BS2 [Strong, -4] bullet to this layer's own reproduced
    "Base ACMG criteria" list — as the last bullet in that contiguous run,
    so pipeline.core.acmg_points.resync_moi_total() (called right
    after this, in moi_recessive.run_solo()) re-sums it into the Base and
    Total point lines automatically — when ALL of:

      (1) segregation is "homozygous_parent" (a parent is themselves
          homozygous for this variant, backend-determined from allelic
          balance — the discordant-with-simple-biallelic-transmission case
          moi_recessive_homozygous.txt already gates PM3 on);
      (2) BS2 is not already present in the base conclusion (upstream
          already handled it, whatever it decided — never double-apply).

    No-op (returns `result` unchanged) if either condition fails, or if the
    expected "Base ACMG criteria" block isn't found in the expected format
    (leave untouched rather than guess where to insert).
    """
    if segregation != "homozygous_parent":
        return result
    if _BS2_ALREADY_PRESENT_RE.search(base_conclusion):
        return result

    m = _BASE_CRITERIA_BLOCK_RE.search(result)
    if not m:
        return result

    bs2_line = (
        "- BS2 [Strong, -4]: Observed in a parent, homozygous for this exact "
        "variant, who is not documented as affected (no evidence tracked for "
        "this parent's clinical status — DEFAULT-UNAFFECTED POLICY applies), "
        "at a genotype expected to cause a fully penetrant recessive disorder "
        "if this variant were truly pathogenic.\n"
    )
    insertion_point = m.end()
    return result[:insertion_point] + bs2_line + result[insertion_point:]

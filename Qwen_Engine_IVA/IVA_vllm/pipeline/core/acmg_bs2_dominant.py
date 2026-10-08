"""
pipeline/core/acmg_bs2_dominant.py — mechanically enforces BS2's
DEFAULT-UNAFFECTED POLICY (prompts/conclusion.txt) for a dominant-relevant
variant genuinely inherited from one identified parent.

Real observed failure: a paternally-inherited PVS1 loss-of-function variant
in a dominant-mode gene was scored Pathogenic (PVS1+PM2+PP4) with BS2 never
applied or even considered, despite no evidence the father is affected.
prompts/conclusion.txt already instructs the base conclusion
stage to apply BS2 by default for exactly this pattern — a heterozygous
(or homozygous) unaffected carrier parent of a fully-penetrant dominant
variant is itself Strong Benign evidence — but that instruction is not
reliably followed. moi_dominant.py's own layer stage is the safest place to
catch the omission mechanically: by the time this layer runs, SEGREGATION
is already backend-determined (not inferred from text) as "maternal" or
"paternal" (genuinely inherited, not de novo), and this layer's own
cosegregation check has already decided whether PP1-qualifying evidence
(the transmitting parent affirmatively stated as affected) exists.
"""

import re

_BS2_ALREADY_PRESENT_RE = re.compile(r"^-\s*BS2\b", re.MULTILINE)
_PP1_APPLIED_RE = re.compile(r"^-\s*PP1\b", re.MULTILINE)

# The base criteria list spliced into this layer's block by
# acmg_points.splice_base_and_total — a contiguous run of
# "- CRITERION [..]: ..." bullet lines directly following its header.
_BASE_CRITERIA_BLOCK_RE = re.compile(r"(\*\*Base ACMG criteria\b[^\n]*\*\*\n(?:-[^\n]*\n?)+)")


def validate_bs2_unaffected_dominant_carrier(
    result: str, base_conclusion: str, segregation: str, disorder: str = "dominant"
) -> str:
    """
    Appends a BS2 [Strong, -4] bullet to this layer's own reproduced
    "Base ACMG criteria" list — as the last bullet in that contiguous run,
    so pipeline.core.acmg_points.resync_moi_total() (called right
    after this, in moi_dominant.py's run_one) re-sums it into the Base and
    Total point lines automatically — when ALL of:

      (1) segregation is "maternal" or "paternal" (genuinely inherited from
          one identified parent, backend-determined, not de novo), or
          "homozygous_parent" (a parent carries two copies of the variant);
      (2) BS2 is not already present in the base conclusion (upstream
          already handled it, whatever it decided — never double-apply);
      (3) this layer's own result did NOT apply PP1 (no cosegregation
          evidence found) — meaning, per the DEFAULT-UNAFFECTED POLICY, the
          transmitting parent is treated as an unaffected carrier by
          default. A confirmed-affected transmitting parent (PP1 applied)
          is not a clean unaffected-carrier case and BS2 does not apply.

    `disorder` names the inheritance in the BS2 text ("dominant", or
    "X-linked dominant" / "X-linked recessive" from moi_xlinked.py).

    No-op (returns `result` unchanged) if any condition fails, or if the
    expected "Base ACMG criteria" block isn't found in the expected format
    (leave untouched rather than guess where to insert).
    """
    if segregation not in ("maternal", "paternal", "homozygous_parent"):
        return result
    if _BS2_ALREADY_PRESENT_RE.search(base_conclusion):
        return result
    if _PP1_APPLIED_RE.search(result):
        return result

    m = _BASE_CRITERIA_BLOCK_RE.search(result)
    if not m:
        return result

    if segregation == "homozygous_parent":
        carrier = ("a parent homozygous (or, for a chrX father, hemizygous) for the variant, clinically unaffected"
                   if disorder.startswith("X-linked") else
                   "a parent homozygous for the variant, clinically unaffected")
    else:
        carrier = f"the {'mother' if segregation == 'maternal' else 'father'}, a clinically unaffected transmitting parent"
    bs2_line = (
        f"- BS2 [Strong, -4]: Observed in {carrier} (no family-history evidence "
        "supports this parent being affected — DEFAULT-UNAFFECTED POLICY "
        f"applies), at a genotype expected to cause a fully penetrant "
        f"{disorder} disorder if this variant were truly pathogenic.\n"
    )
    insertion_point = m.end()
    return result[:insertion_point] + bs2_line + result[insertion_point:]

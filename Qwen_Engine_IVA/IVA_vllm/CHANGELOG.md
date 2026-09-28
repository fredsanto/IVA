# Changelog

## 2026-09-13 — De Novo delta-regex fix + suspected-parent PP1 policy

Commit: `d621cb1` — "Fix ACMG delta-line regex blind spot and add suspected-parent PP1 handling"

### Context

A 16-trio round (`refix2`, 09-13) showed a real regression vs. prior runs. Investigation
traced it to a false-positive/false-negative arithmetic bug hitting the De Novo layer
across most of the 16 cases, plus a separate policy gap on how a "suspected but
unconfirmed" affected parent is scored.

### Fixed

1. **`pipeline/core/acmg_points.py` — `_DELTA_LINE_RE` regex.**
   Character class `[A-Za-z-]+` excluded spaces, so `"**De novo delta:**"` (two words)
   never matched — only single-word/hyphenated layer names (`Recessive delta`,
   `X-linked delta`) did. Consequence: `recompute_and_fix_totals()`'s Base+delta vs.
   Total reconciliation pass silently no-op'd for every De Novo layer block — the most
   common layer — letting stale/invented SLM totals through uncorrected (observed:
   stated `Base: 3, delta: +0, Total: 11` on an RYR1 variant in LGE_14; correct value
   is 3). Fix: widened to `[A-Za-z][A-Za-z -]*` to allow internal spaces.
   Verified against synthetic repro (3+0 now correctly resolves to 3, was 11).

2. **`prompts/moi_dominant.txt` — suspected/unconfirmed parental phenotype handling.**
   A transmitting parent described as having a "suspected"/"possible"/unconfirmed
   diagnosis matching the patient's condition (e.g. "father with suspected X") was
   previously defaulting to BS2 (DEFAULT-UNAFFECTED POLICY) since the diagnosis wasn't
   *confirmed* — cancelling out otherwise-strong evidence. New instruction: treat this
   the same direction as an affected parent (PP1-eligible, BS2 withheld), while
   mandating an explicit report caveat that the finding is "pending clinical
   confirmation of the parent's phenotype." Observed case: LGE_5/FGFR3, father
   "suspicion hypochondroplasie" per source clinical CSV.

### Verified post-fix (server restarted after commit `d621cb1`)

- **LGE_5 (FGFR3)**: rerun 20:17-20:37. Was 4pts VUS (BS2 applied against a
  wrongly-defaulted-unaffected father). Now: PP1 applied, Base 8 + Dominant delta
  +1 = Total 9 → Likely Pathogenic, correctly in section 2, caveat sentence present
  verbatim. **Confirmed fixed end-to-end.**
- **LGE_14 (RYR1 compound-het + BRCA1 SF)**: rerun 20:17-21:04. RYR1 compound-het
  now correctly lands in section 2 as `CAUSATIVE` (Variant A 9pts, Variant B stated
  9pts — see residual note below) — no longer needs the mechanical FYI-addendum
  fallback. BRCA1 correctly in section 3 (SF). A new unrelated finding (CACNA1E,
  14pts) also appeared this run — not investigated, may be legitimate or run
  variance. **Confirmed fixed.**
- **LGE_15 (SUPT16H)**: rerun 20:59-21:46. PS2 now survives into the final section
  2: PM2(2)+PS2(4)+PM5(2)=8pts → Likely Pathogenic, arithmetic consistent.
  **Confirmed fixed.** The MAP-stage-drops-a-criterion theory floated during
  investigation (see below) did not hold up — once the raw block's own Total was
  no longer self-contradictory (two different numbers for "the" score), the MAP
  condensation pass carried PS2 through correctly. Downgraded from "known bug" to
  "theory ruled out, but worth re-testing if it recurs on a still-broken block."

- **Residual, not yet fixed**: LGE_14 Variant B still shows internally inconsistent
  math — PVS1(4)+PM2(2)+PM5(2)+PM3(2)=10, stated as "9 pts total." Smaller than the
  original bug (was 11) but not zero. The compound-het/recessive joint-scoring path
  (`moi_recessive.py`) apparently has its own total-line shape not covered by the
  `_DELTA_LINE_RE` fix — same bug family, different code path. Doesn't cross a
  classification band here (still Likely Pathogenic), not chased further this
  session.

## 2026-09-13 (cont'd) — `homozygous_parent` misread as de novo evidence

### Problem

Systematic AB audit across all 16 trios' causative calls (cross-referencing each
variant's `Allelic_balance_proband`/`Allelic_Balance_mother`/`Allelic_Balance_father`
CSV columns, plus the CSV's own `denovo` ground-truth flag, against what each report
claimed) found **3 confirmed instances** of a parent's AB≈1.0 (homozygous for the
variant — the single strongest possible evidence AGAINST de novo) being used to
*support* a de novo / PS2 call instead of correctly blocking it:

- **LGE_8 / ARHGEF6**: father AB=1.00. Report: *"father is homozygous reference
  (AB=1.00), confirming a de novo origin"* — directionally inverted (AB≈1.0 = alt-heavy,
  not reference).
- **LGE_8 / PHF8**: same pattern, father AB=1.00.
- **LGE_4 / TET3**: mother AB=1.00, CSV `denovo=FALSE`. Report: *"Confirmed by parental
  allelic balance showing absence of the variant in both mother and father"* — not
  even a misread, a flat fabrication contradicting the actual AB=1.00 reading.

### Root cause

`pipeline/core/segregation.py`'s `classify_segregation()` computes correctly in all
three cases — hand-traced through its branch logic, confirms `"homozygous_parent"` is
the correct return value each time (verified: mother/father `classify_ab_ratio`
returns `"hom"` for AB>0.8, which unconditionally triggers the
`mother_class=="hom" or father_class=="hom"` branch before any de-novo branch is
reached). The bug is downstream, in `prompts/moi_denovo.txt` rule 4: it names only 3
of the 7 possible non-`"de_novo"` segregation values as examples (`"maternal"`,
`"paternal"`, `"uncertain"`) and never defines `"homozygous_parent"` anywhere. Facing
an enum token it was never shown a worked example for, the model invented its own
(wrong) gloss instead of applying the general rule ("not `de_novo`" → withhold PS2)
mechanically.

Cases with the same `"homozygous_parent"` backend value that did NOT trigger the bug
(LGE_8 BCOR, TSPAN7 — both chrX, father AB=1.00) were unaffected only because their
causative score never depended on PS2/PM6 in the first place (base PVS1+PM2 alone was
sufficient) — same underlying misread was likely still present internally but
harmless to the final number.

### Fixed

**`prompts/moi_denovo.txt` rule 4** — rewritten to spell out all 6 non-`de_novo`
segregation values explicitly (`maternal`/`paternal`/`both_carriers`/
`homozygous_parent`/`uncertain`/`insufficient_data`), with `homozygous_parent`
explicitly defined as "the STRONGEST possible evidence AGAINST de novo, never a
signal that supports it," and the exact TET3/ARHGEF6-style failure named inline so
the model can't reconstruct the same wrong inference. `prompts/moi_dominant.txt`
already handled this value correctly (explicitly names and excludes
`homozygous_parent`) — no change needed there.

### Full AB audit table (all 16 trios' causative calls)

Everywhere else the backend segregation value and the report's PS2/PM6 usage were
consistent — including one genuinely ambiguous case (LGE_13 PHKA1, proband AB=0.24
sits in `classify_ab_ratio`'s ambiguous 0.2-0.3 gap → `"uncertain"`, CSV's own
`denovo` flag says `TRUE` but the model correctly did not apply PS2 regardless,
landing on base-score-only Likely Pathogenic — harmless disagreement, not a bug).

### Not yet verified

Fix is prompt-only (`moi_denovo.txt` is read fresh per-request, no code change, no
restart needed) — not yet retested against a live rerun of LGE_4/LGE_8 at time of
writing. Queued: full 16-trio rerun split across 3 servers (dnagpu002:8002/8003 +
new 8004) to verify this fix plus re-confirm everything above in one pass.

### Not in scope / explicitly ruled out during investigation

- BRCA1 in LGE_14 is a correct ACMG secondary (actionable) finding — SF genes bypass
  phenotype-match filtering by design; the PVS1-counted-despite-"not applicable"
  wrinkle on that block did not actually cause a reporting miss (BRCA1 landed
  correctly in section 3 regardless).
- LGE_5 FGFR3's father ("suspicion hypochondroplasie") is a distinct, already-fixed
  case (see PP1 policy fix above) — not the same bug as this section.

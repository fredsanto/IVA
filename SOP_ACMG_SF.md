# SOP — ACMG Secondary Findings (Actionable Variants) Reporting

**Scope:** clinical/lab staff reviewing ServerQwen variant analysis reports.
**System component:** `Qwen_Engine_IVA/IVA_vllm` pipeline,
`ACTIONABLE VARIANTS (ACMG SF)` report section.

---

## 1. Purpose

Define the procedure for identifying, reviewing, and acting on secondary
findings (SF) per the American College of Medical Genetics and Genomics (ACMG)
recommendations: pathogenic or likely pathogenic (P/LP) variants in a fixed
list of genes with actionable health implications, discovered incidentally
during analysis of a patient's primary indication, **irrespective of whether
the finding is related to the primary phenotype or in which family member
(proband or parent) it was observed.**

---

## 2. Background

The ACMG SF list (current version: **v3.2, 81 genes**, source:
[ncbi.nlm.nih.gov/clinvar/docs/acmg](https://www.ncbi.nlm.nih.gov/clinvar/docs/acmg/))
covers conditions where early detection and intervention change clinical
management — hereditary cancer syndromes (e.g. `BRCA1`/`BRCA2`, Lynch
syndrome genes), inherited cardiac disease (cardiomyopathies, arrhythmia
syndromes), and selected metabolic/other conditions (Wilson disease,
malignant hyperthermia, etc.).

Unlike the rest of the pipeline's output — which is driven by fit to the
patient's presenting phenotype — SF reporting is **not** phenotype-gated.
A variant is reportable purely on gene identity + pathogenicity, regardless
of how unrelated the associated condition is to why the patient was
sequenced, and regardless of whether it was found in the proband's own
variant or reflects a variant carried by a parent.

---

## 3. Automated detection (system behavior)

For every variant in the input file, the pipeline:

1. Checks whether `Gene` is in the ACMG SF v3.2 list
   (`pipeline/core/acmg_sf.py::ACMG_SF_GENES`).
2. If so, determines P/LP status:
   - Primary: the `ClinVar_class` column, if `Pathogenic` or `Likely pathogenic`.
   - Fallback (only when `ClinVar_class` is `NA`/missing): an SLM judgment of
     whether the variant's LitVar2/PubMed literature evidence explicitly
     reports a clinically classified P/LP variant for that gene.
3. Flagged variants are guaranteed to survive triage and appear in the final
   report — both in their normal MOI/ACMG scoring section **and** in a
   dedicated `ACTIONABLE VARIANTS (ACMG SF)` section with a plain-language
   summary of the gene, condition, and basis for the P/LP call.

This is an automated screen, not a final clinical determination — see
Section 5.

---

## 4. Reviewer procedure

When a report contains an `ACTIONABLE VARIANTS (ACMG SF)` section:

1. **Do not dismiss the finding based on phenotype mismatch.** Its presence
   in this section means the system has already determined it is unrelated
   to (or independent of) the primary indication — that is expected, not an
   error.
2. **Confirm the P/LP classification independently** against current ClinVar
   and, where available, a curated internal database — the system's call is
   ClinVar-field- or literature-evidence-derived and is not a substitute for
   variant-level clinical review.
3. **Confirm gene-list currency.** The ACMG SF list is revised periodically
   (v3.0 → v3.1 → v3.2 → ...). Check `pipeline/core/acmg_sf.py` against the
   latest ACMG publication before relying on an absence of findings; a gene
   added in a newer SF version will not be flagged until the list is updated
   (see Section 6).
4. **Determine origin (proband vs. parent)** from the report's SEGREGATION
   ANALYSIS / parental allelic-balance data, where trio data is available.
   Origin does **not** change reportability, but affects who receives
   genetic counseling.
5. **Route for genetic counseling** per the condition category:
   - Cancer predisposition genes → oncogenetics referral.
   - Cardiac genes → cardiogenetics referral / cardiology.
   - Metabolic/other → relevant specialty referral.
6. **Document the finding and counseling outcome** in the patient's record
   per standard secondary-findings consent and disclosure policy (patients
   must have consented to SF reporting prior to sequencing, per local policy
   — this pipeline does not track or enforce consent status).
7. **If the variant is a VUS or Likely Benign/Benign** and the SLM fallback
   path flagged it in error (no `ClinVar_class` was available and the
   literature fallback misfired), correct/backfill the `ClinVar_class` field
   in the source data for future runs and disregard the finding after
   independent confirmation.

---

## 5. Limitations

- The literature-evidence fallback (used only when `ClinVar_class` is `NA`)
  is a single cheap SLM call over already-fetched LitVar2/PubMed text — it
  is a triage aid, not a diagnostic classification. Any SF finding reached
  via this path should be prioritized for manual ClinVar/variant-database
  confirmation before disclosure.
- The gene list is a static snapshot (`pipeline/core/acmg_sf.py`) captured
  from the ACMG SF v3.2 page at implementation time. It is not fetched live
  and will not reflect list updates until manually refreshed.
- This SOP covers only the automated actionable-findings screen. It does not
  replace institutional policy on secondary findings consent, disclosure, or
  opt-out handling.

---

## 6. Updating the gene list

When ACMG publishes a new SF version:

1. Fetch the updated table from
   [ncbi.nlm.nih.gov/clinvar/docs/acmg](https://www.ncbi.nlm.nih.gov/clinvar/docs/acmg/).
2. Update `ACMG_SF_CONDITIONS` in
   `Qwen_Engine_IVA/IVA_vllm/pipeline/core/acmg_sf.py`
   (gene → condition mapping; `ACMG_SF_GENES` derives from its keys).
3. Update the version number referenced in this SOP and in
   `Qwen_Engine_IVA/IVA_vllm/README.md`.
4. Re-run existing test cases to confirm the new/removed genes are picked
   up correctly.

---

## 7. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-07-14 | Initial SOP — ACMG SF v3.2 (81 genes) actionable-variants detection introduced. |

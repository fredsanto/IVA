"""
pipeline/core/acmg_sf.py — ACMG Secondary Findings (SF v3.2) actionable-gene detection.

Source: https://www.ncbi.nlm.nih.gov/clinvar/docs/acmg/ (ACMG SF v3.2, 81 genes).

Secondary findings are P/LP variants in a fixed list of genes with actionable
implications (cancer predisposition, cardiac disease, metabolic disease, etc.)
that must be reported regardless of relevance to the patient's presenting
phenotype and regardless of which sample the variant was observed in — the
recommendation is gene+classification driven, not phenotype- or
proband/parent-driven, so detection here is deliberately independent of the
MOI/phenotype-fit machinery elsewhere in the pipeline.

Public API:
    ACMG_SF_GENES                       — frozenset of gene symbols
    ACMG_SF_CONDITIONS                  — {gene: condition string}
    build_actionable_set(variants, litvar2_raw_by_variant, llm)
        -> (set[int] actionable_indices, dict[int, str] reasons,
            dict[int, str] classifications)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pipeline.core.clinvar_reference import resolve_variant_clinvar_status

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

# ACMG SF v3.2 (81 genes) — gene: associated actionable condition
ACMG_SF_CONDITIONS: dict[str, str] = {
    "APC":     "Adenomatous polyposis coli",
    "MYH11":   "Familial thoracic aortic aneurysm 4",
    "ACTA2":   "Familial thoracic aortic aneurysm 6",
    "TMEM43":  "Arrhythmogenic right ventricular cardiomyopathy 5",
    "DSP":     "ARVC type 8; dilated cardiomyopathy",
    "PKP2":    "ARVC type 9",
    "DSG2":    "ARVC type 10",
    "DSC2":    "ARVC type 11",
    "BTD":     "Biotinidase deficiency",
    "BRCA1":   "Breast-ovarian cancer, familial 1",
    "BRCA2":   "Breast-ovarian cancer, familial 2",
    "SCN5A":   "Brugada syndrome; dilated cardiomyopathy; Long QT",
    "RYR2":    "Catecholaminergic polymorphic ventricular tachycardia 1",
    "CASQ2":   "CPVT 2",
    "CALM1":   "CPVT 4; Long QT syndrome 14",
    "TRDN":    "CPVT 5; Long QT syndrome",
    "FLNC":    "Dilated cardiomyopathy; myofibrillar myopathy 5",
    "LMNA":    "Dilated cardiomyopathy 1A",
    "TNNT2":   "DCM 1D; familial hypertrophic cardiomyopathy 2",
    "DES":     "DCM 1I; myofibrillar myopathy 1",
    "MYH7":    "DCM 1S; familial hypertrophic cardiomyopathy 1",
    "TNNC1":   "DCM 1Z",
    "RBM20":   "DCM 1DD",
    "BAG3":    "DCM 1HH; myofibrillar myopathy 6",
    "TTN":     "Dilated cardiomyopathy (truncating variants only)",
    "COL3A1":  "Ehlers-Danlos syndrome, type 4",
    "GLA":     "Fabry disease",
    "LDLR":    "Familial hypercholesterolemia 1",
    "APOB":    "Familial hypercholesterolemia 2",
    "TPM1":    "Familial hypertrophic cardiomyopathy 3",
    "MYBPC3":  "Familial hypertrophic cardiomyopathy 4",
    "PRKAG2":  "FHC 6",
    "TNNI3":   "FHC 7",
    "MYL3":    "FHC 8",
    "MYL2":    "FHC 10",
    "ACTC1":   "FHC 11",
    "RET":     "Familial medullary thyroid carcinoma; MEN 2a; MEN 2b",
    "PALB2":   "Hereditary breast cancer",
    "HFE":     "Hereditary hemochromatosis (c.845G>A homozygotes)",
    "ENG":     "Hereditary hemorrhagic telangiectasia type 1",
    "ACVRL1":  "HHT type 2",
    "SDHD":    "Hereditary paraganglioma-pheochromocytoma syndrome",
    "SDHB":    "Hereditary paraganglioma-pheochromocytoma syndrome",
    "TTR":     "Hereditary transthyretin-related amyloidosis",
    "PCSK9":   "Hypercholesterolemia, autosomal dominant 3",
    "BMPR1A":  "Juvenile polyposis syndrome",
    "SMAD4":   "Juvenile polyposis; JPS/HHT syndrome",
    "TP53":    "Li-Fraumeni syndrome 1",
    "TGFBR1":  "Loeys-Dietz syndrome type 1A",
    "TGFBR2":  "Loeys-Dietz syndrome type 1B",
    "SMAD3":   "Loeys-Dietz syndrome type 3",
    "KCNQ1":   "Long QT syndrome 1",
    "KCNH2":   "Long QT syndrome 2",
    "CALM2":   "Long QT syndrome 15; CPVT",
    "CALM3":   "Long QT syndrome 16; CPVT",
    "MSH2":    "Lynch syndrome 1",
    "MLH1":    "Lynch syndrome 2",
    "PMS2":    "Lynch syndrome 4",
    "MSH6":    "Lynch syndrome 5",
    "RYR1":    "Malignant hyperthermia",
    "CACNA1S": "Malignant hyperthermia",
    "FBN1":    "Marfan syndrome",
    "HNF1A":   "Maturity-Onset Diabetes of the Young",
    "MEN1":    "Multiple endocrine neoplasia, type 1",
    "MUTYH":   "MYH-associated polyposis",
    "NF2":     "Neurofibromatosis, type 2",
    "OTC":     "Ornithine carbamoyltransferase deficiency",
    "SDHAF2":  "Paragangliomas 2",
    "SDHC":    "Paragangliomas 3",
    "STK11":   "Peutz-Jeghers syndrome",
    "MAX":     "Pheochromocytoma",
    "TMEM127": "Pheochromocytoma",
    "GAA":     "Pompe disease",
    "PTEN":    "PTEN hamartoma tumor syndrome",
    "RB1":     "Retinoblastoma",
    "RPE65":   "RPE65-related retinopathy",
    "TSC1":    "Tuberous sclerosis 1",
    "TSC2":    "Tuberous sclerosis 2",
    "VHL":     "Von Hippel-Lindau syndrome",
    "WT1":     "Wilms tumor",
    "ATP7B":   "Wilson disease",
}

ACMG_SF_GENES: frozenset[str] = frozenset(ACMG_SF_CONDITIONS)

# ClinVar_class substrings that rule out a pathogenic call even though they
# may contain the substring "pathogenic" (e.g. "Conflicting interpretations
# of pathogenicity").  Checked before the "pathogenic" substring match.
_NON_PATHOGENIC_MARKERS = ("benign", "uncertain", "conflicting", "not provided", "risk factor")


def is_pathogenic_clinvar(value: str | None) -> bool:
    """True if a ClinVar_class field value reads as Pathogenic or Likely pathogenic."""
    if not value:
        return False
    v = value.strip().lower()
    if v in ("", "na"):
        return False
    if any(marker in v for marker in _NON_PATHOGENIC_MARKERS):
        return False
    return "pathogenic" in v


def _slm_check_clingen_pathogenic(gene: str, hgvs: str, clingen_text: str, llm: "LLMClient") -> bool:
    """
    Cheap fallback SLM call, parallel to _slm_check_litvar2_pathogenic above,
    for the ClinGen Allele Registry cross-reference fetched by
    tools/clingen_allele.py — used when ClinVar_class is NA/missing AND the
    litvar2 fallback above didn't already flag the variant.

    IMPORTANT ASYMMETRY vs. the litvar2 check: clingen_allele's own output
    (see its run() method) never states a clinical significance at all — it
    only reports a CAid, and, if present, a bare "ClinVar variation ID: N
    (RCV: ...)" cross-reference line. That line proves a ClinVar record
    EXISTS for this exact variant; it says nothing about whether that record
    is Pathogenic or Benign. The prompt below is written to fail closed on
    that ambiguity — it must answer NO whenever the text stops at an ID/RCV
    with no significance wording, and must not treat "has a ClinVar entry"
    as a proxy for "is a P/LP entry".
    """
    system = (
        "You are a clinical genetics literature screener. "
        "Answer with exactly one word: YES or NO."
    )
    user = (
        f"Gene: {gene}\n"
        f"Variant under review: {hgvs}\n\n"
        f"ClinGen Allele Registry cross-reference:\n{clingen_text[:3000]}\n\n"
        "Does this text EXPLICITLY state that this variant's ClinVar record "
        "has a clinical significance of Pathogenic or Likely pathogenic? "
        "A bare 'ClinVar variation ID' or 'RCV' cross-reference with no "
        "stated clinical significance does NOT count — that only proves a "
        "ClinVar record exists, not what it says. Answer YES only if a "
        "significance of Pathogenic/Likely pathogenic is stated in the text "
        "itself. Answer YES or NO."
    )
    try:
        raw = llm.generate(system=system, user=user, max_tokens=5, temperature=0.0).strip().upper()
    except Exception as exc:
        logger.warning("[ACMG-SF] clingen P/LP fallback check failed for gene %s: %s", gene, exc)
        return False
    return raw.startswith("Y")


def _slm_check_litvar2_pathogenic(gene: str, hgvs: str, litvar2_text: str, llm: "LLMClient") -> bool:
    """
    Cheap fallback SLM call used only when ClinVar_class is unavailable/NA for a
    variant in an ACMG SF gene: does the already-fetched litvar2 evidence
    explicitly report THIS SPECIFIC VARIANT (matched by HGVS/position) as
    clinically classified P/LP — not merely that the gene is disease-
    associated, and not some OTHER classified variant in the same gene.

    Gene-level literature for a well-studied ACMG SF gene will almost always
    mention *some* known pathogenic variant as background context (e.g. APOB's
    literature discusses the well-known R3500Q variant even when the variant
    actually under review is an unrelated deep-intronic change) — without
    pinning the check to this variant's own identity, that background mention
    alone was enough to flag an unrelated variant as an incidental finding.
    Same failure shape as citing a different variant's ClinVar record for PS1.
    """
    system = (
        "You are a clinical genetics literature screener. "
        "Answer with exactly one word: YES or NO."
    )
    user = (
        f"Gene: {gene}\n"
        f"Variant under review: {hgvs}\n\n"
        f"Literature evidence:\n{litvar2_text[:6000]}\n\n"
        "Does this evidence explicitly report THIS SPECIFIC variant "
        f"({hgvs}) — not a different variant in the same gene, and not the "
        "gene in general — as clinically classified Pathogenic or Likely "
        "pathogenic? A mention of some other named pathogenic variant in "
        "this gene (e.g. a well-known founder/hotspot variant used as "
        "background context) does NOT count unless it is this exact variant. "
        "Answer YES only if the evidence ties a P/LP classification to this "
        "variant's own position/HGVS/rsID. Answer YES or NO."
    )
    try:
        raw = llm.generate(system=system, user=user, max_tokens=5, temperature=0.0).strip().upper()
    except Exception as exc:
        logger.warning("[ACMG-SF] litvar2 P/LP fallback check failed for gene %s: %s", gene, exc)
        return False
    return raw.startswith("Y")


def build_actionable_set(
    variants: list[dict],
    litvar2_raw_by_variant: dict[int, str | None],
    llm: "LLMClient",
    clingen_raw_by_variant: dict[int, str | None] | None = None,
    clinvar_tally_raw_by_variant: dict[int, str | None] | None = None,
) -> tuple[set[int], dict[int, str]]:
    """
    Identify variants that are ACMG SF actionable findings: gene is in the
    ACMG SF v3.2 list AND the variant is P/LP, checked in this order:
      1. ClinVar_class field on the variant dict (primary — but this is an
         SLM-mapped column from the input CSV and can be missing or mapped
         from the wrong raw column; see the BRCA1 case below).
      2. ClinVarGeneStatsTool's own already-fetched, deterministic
         per-submission ClinVar tally (resolve_variant_clinvar_status,
         pipeline.core.clinvar_reference) — real ClinVar data straight from
         NCBI, entirely independent of the input CSV's own columns. A real
         observed failure this fixes: a CSV's actual ClinVar-classification
         column had a dated, non-standard header
         ("ClinVar.20230813..Stars..Conflict_details.") that the SLM header
         interpreter left unmapped while mapping an unrelated empty column
         to ClinVar_class instead — silently losing a 46/46-submitter
         Pathogenic BRCA1 call that ClinVarGeneStatsTool had already fetched
         correctly and was sitting unused in the pipeline's own evidence.
      3. litvar2 literature evidence (SLM judgment call) when 1 and 2 both
         come back unresolved/NA.
      4. clingen_allele's ClinVar cross-reference (SLM judgment call) as a
         last resort — see _slm_check_clingen_pathogenic's docstring for why
         this fires only rarely: clingen_allele's own output proves a
         ClinVar record exists, not its significance, so the check fails
         closed rather than false-positiving on a bare variation ID.

    Returns (actionable_indices, reasons, classifications). reasons[i] is a
    short string describing why variant i was flagged, for use in
    triage-exemption justifications and the Actionable Variants report
    section. classifications[i] is this variant's OWN, phenotype-independent
    P/LP classification (e.g. "Pathogenic", "Likely pathogenic") resolved
    from the exact same evidence that triggered the flag above — this MUST
    be used verbatim as this finding's reported classification, never
    re-derived from that variant's phenotype-scored base-conclusion ACMG
    label elsewhere in the pipeline. A real observed failure this fixes: a
    BRCA1 frameshift variant was correctly flagged actionable here from its
    own ClinVar record (46/46 submitters, Pathogenic) — the report's
    Actionable Findings section then displayed its classification as
    "Uncertain Significance (VUS)" anyway, because the caller pulled that
    label from the variant's Layer 2 base conclusion, which (correctly, for
    ITS OWN purpose) scores PP4/PVS1 against this patient's presenting
    phenotype and found no gene-disease link to it. ACMG SF reporting is
    gene+classification driven, independent of phenotype, by design (see
    module docstring) — a secondary-finding gene's own real-world
    pathogenicity must never be diluted by a phenotype mismatch that is
    completely beside the point for a mandatory secondary finding.
    """
    actionable_indices: set[int] = set()
    reasons: dict[int, str] = {}
    classifications: dict[int, str] = {}
    clingen_raw_by_variant = clingen_raw_by_variant or {}
    clinvar_tally_raw_by_variant = clinvar_tally_raw_by_variant or {}

    for i, variant in enumerate(variants):
        gene = (variant.get("Gene") or "NA").strip()
        if gene == "NA" or gene not in ACMG_SF_GENES:
            continue

        clinvar_class = variant.get("ClinVar_class", "NA")
        if is_pathogenic_clinvar(clinvar_class):
            actionable_indices.add(i)
            reasons[i] = f"ClinVar_class={clinvar_class}"
            classifications[i] = str(clinvar_class).strip()
            logger.info(
                "[ACMG-SF] Variant %d (%s) flagged actionable via ClinVar_class=%s",
                i + 1, gene, clinvar_class,
            )
            continue

        resolved_status = resolve_variant_clinvar_status(clinvar_tally_raw_by_variant.get(i))
        if is_pathogenic_clinvar(resolved_status):
            actionable_indices.add(i)
            reasons[i] = f"ClinVar submission tally (variant-level, deterministic): {resolved_status}"
            classifications[i] = str(resolved_status).strip()
            logger.info(
                "[ACMG-SF] Variant %d (%s) flagged actionable via ClinVarGeneStatsTool "
                "submission tally (CSV ClinVar_class was unusable): %s",
                i + 1, gene, resolved_status,
            )
            continue

        hgvs = variant.get("HGVS") or variant.get("Variant") or "this variant"

        litvar2_text = litvar2_raw_by_variant.get(i)
        if litvar2_text and _slm_check_litvar2_pathogenic(gene, hgvs, litvar2_text, llm):
            actionable_indices.add(i)
            reasons[i] = "P/LP variant reported for this gene in literature (LitVar2/PubMed evidence)"
            classifications[i] = "Pathogenic/Likely pathogenic (per literature evidence)"
            logger.info(
                "[ACMG-SF] Variant %d (%s) flagged actionable via litvar2 evidence fallback",
                i + 1, gene,
            )
            continue

        clingen_text = clingen_raw_by_variant.get(i)
        if clingen_text and _slm_check_clingen_pathogenic(gene, hgvs, clingen_text, llm):
            actionable_indices.add(i)
            reasons[i] = "P/LP variant per ClinGen Allele Registry ClinVar cross-reference"
            classifications[i] = "Pathogenic/Likely pathogenic (per ClinGen/ClinVar cross-reference)"
            logger.info(
                "[ACMG-SF] Variant %d (%s) flagged actionable via clingen_allele evidence fallback",
                i + 1, gene,
            )

    return actionable_indices, reasons, classifications

"""
pipeline/pipeline.py — Top-level orchestrator.

Wires together:
  - LLM client (vLLM in production, HF in dev)
  - Tool instances
  - Manifest loader
  - Executor
  - Stage functions (retrieval → first_triage → reasoning → cross_analysis → conclusion → final_conclusion)

Design rule: server.py calls pipeline.run() and knows nothing about variants.
All logic lives here and in the stages/tools/core modules.

Public API:
    Pipeline(model_name, llm_kwargs)   — create and configure
    Pipeline.run(variants, phenotype)  — execute the full pipeline, return report string

Per-variant flow:

  retrieval:         variants → list[str]  (one context slice per variant)
  first_triage:      per slice → KEEP/DISCARD  (skipped if n <= TRIAGE_ENABLED_THRESHOLD)
  reasoning:         kept slices → dict[int, str]  (one reasoning block per kept variant)
  cross_analysis:    per gene with ≥2 tier-1 variants → dict[gene, str]
  conclusion:        tier-1 slices + reasoning + cross_analysis → dict[int, str]
  final_conclusion:  tier-1 conclusions → str  (the # Clinical Conclusion paragraph)
"""

from __future__ import annotations

import re
import logging
import itertools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

MAX_WORKERS          = 64   # first_triage (vLLM calls, runs across all variants) —
                             # raised from 32 to test whether this all-variant stage,
                             # not reasoning/conclusion's smaller kept-variant pool, is
                             # the actual bottleneck
_CLUSTER_LINE_RE = re.compile(r"(CLUSTER_PHENOTYPE:\s*)(YES|PARTIAL|NONE|INCIDENTAL)", re.IGNORECASE)
MAX_WORKERS_LLM      = 16   # reasoning/conclusion (vLLM calls, GPU-bound)

from pipeline.config          import (
    DEFAULT_GENOME_BUILD,
    TRIAGE_ENABLED_THRESHOLD,
)
from pipeline.core.manifest  import ManifestLoader
from pipeline.core.executor  import Executor
from pipeline.core.context   import ToolContext
from pipeline.core.normalizer import TARGET_COLUMNS
from pipeline.core            import moi
from pipeline.core            import segregation
from pipeline.core            import acmg_sf
from pipeline.llm.registry   import get_client
from pipeline.tools          import (
    AutoPVS1Tool, LitVar2SummaryTool, SpliceAITool, WebSearchAgentTool,
    GnomadConstraintTool, GnomadFrequencyTool, ClinVarGeneStatsTool,
    ClinVarResidueSearchTool, ClinVarHotspotTool, UniProtDomainTool, ClinGenAlleleTool, GeneReviewsTool,
)
from pipeline.tools.clinvar_gene_stats import classify_consequence_counts
from pipeline.tools.gnomad_constraint  import (
    classify_pli, classify_loeuf, _BUILD_TO_REFERENCE_GENOME,
)
from pipeline.tools.gnomad_frequency   import fetch_variant_frequency
from pipeline.tools.autopvs1           import (
    parse_variant_coords, _CLEAR_LOF_TYPES,
)
from pipeline.core.errors              import ToolFetchError, ToolParseError
from pipeline.core.clinvar_reference   import (
    append_clinvar_reference, add_clinvar_status_to_findings, clinvar_status_from_context,
)
from pipeline.core.acmg_ps3            import surface_ps3_citations
from pipeline.core.acmg_case_ref       import apply_case_reference
from pipeline.stages         import (
    retrieval, reasoning, conclusion, cross_analysis, final_conclusion, first_triage,
    moi_denovo, moi_dominant, moi_recessive, moi_xlinked, actionable,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANT STRING → DICT CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_variant_str(variant_str: str) -> dict:
    """
    Parse a canonical key=value variant string into a dict.

    The normalizer produces strings of the form:
        "Variant=chr1:1000000 T>C, Chromosome=chr1, Position=1000000, ..."

    We extract each TARGET_COLUMN by matching:
        <ColumnName>=<value>
    where value runs up to the next ", <NextColumnName>=" or end of string.

    This is more robust than a naive split on ", " because field values may
    themselves contain commas (e.g. OMIM_phenotype).
    """
    result: dict[str, str] = {}
    for col in TARGET_COLUMNS:
        # Pattern: "Col=<capture>" stopped at ", NextCol=" or end of string
        m = re.search(
            rf"(?:^|,\s*){re.escape(col)}=(.*?)(?=,\s*(?:{'|'.join(re.escape(c) for c in TARGET_COLUMNS)})=|$)",
            variant_str,
        )
        result[col] = m.group(1).strip() if m else "NA"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# Matches the shared PATIENT DATA … VARIANT DATA:\n\n header within each slice
_SLICE_HEADER_RE = re.compile(r"^(PATIENT DATA:.*?VARIANT DATA:\n\n)", re.DOTALL)

# Matches the classification named on the conclusion's "ACMG points: N → Classification" line.
_ACMG_POINTS_LINE_RE = re.compile(r"ACMG points:?\*{0,2}\s*[^\n→]*→\s*([A-Za-z /()]+)", re.IGNORECASE)

# Matches the numeric point value on that same line (e.g. "ACMG points: 8 →" -> 8).
# Used to pull the base ACMG score out as a number so downstream MOI layers can
# compute base+delta without re-parsing prose or trusting the LLM's own arithmetic
# across two separate documents (base conclusion + layer-specific addition).
_ACMG_POINTS_VALUE_RE = re.compile(r"ACMG points:?\*{0,2}\s*(-?\d+)", re.IGNORECASE)


def _extract_acmg_points_value(conclusion_text: str) -> int | None:
    """Numeric base ACMG point total from a conclusion block, or None if the
    line is missing/unparseable (caller should treat that as unknown, not 0)."""
    m = _ACMG_POINTS_VALUE_RE.search(conclusion_text)
    if not m:
        return None
    return int(m.group(1))

# Checked most-specific-first so "likely pathogenic"/"likely benign" don't match
# the "pathogenic"/"benign" substring check meant for the standalone terms.
_ACMG_CLASSIFICATION_RANK = [
    ("likely pathogenic", 1),
    ("pathogenic", 0),
    ("uncertain significance", 2),
    ("vus", 2),
    ("likely benign", 3),
    ("benign", 4),
]


def _classification_rank(conclusion_text: str) -> int:
    """P=0, LP=1, VUS=2, LB=3, B=4; unrecognized/missing sorts last."""
    m = _ACMG_POINTS_LINE_RE.search(conclusion_text)
    if not m:
        return len(_ACMG_CLASSIFICATION_RANK) + 1
    label = m.group(1).strip().lower()
    for key, rank in _ACMG_CLASSIFICATION_RANK:
        if key in label:
            return rank
    return len(_ACMG_CLASSIFICATION_RANK) + 1


_LOF_HGVS_CLUES = ("fs", "ter", "del", "dup", "ins", "ext*")


def _is_lof_variant(variant: dict) -> bool:
    """
    Fast, deterministic LoF check — reuses the exact same rule AutoPVS1's own
    gate uses (autopvs1.py's _CLEAR_LOF_TYPES) so "LoF" means the same thing
    here as it does for PVS1 eligibility elsewhere in the pipeline.
    """
    type_val = (variant.get("Type") or "").lower()
    if any(t in type_val for t in _CLEAR_LOF_TYPES):
        return True
    hgvs_val = (variant.get("HGVS") or "").lower()
    if any(c in hgvs_val for c in _LOF_HGVS_CLUES) and "=" not in hgvs_val:
        return True
    return False


def _gene_phenotype_relevant(context_slice: str) -> bool:
    """
    True unless the retrieval evidence explicitly recorded no gene-phenotype
    literature link (litvar2.py emits the literal string "NO DISEASE LINK"
    in both of its no-link cases) — i.e. relevance is assumed by default,
    only overridden by an explicit negative signal.
    """
    return "NO DISEASE LINK" not in context_slice


def _group_by_gene(variants: list[dict]) -> dict[str, list[int]]:
    """
    Return a mapping of gene symbol → list of 0-based variant indices.

    Variants with Gene == "NA" or missing Gene are excluded (they cannot
    participate in gene-level cross-analysis).
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, variant in enumerate(variants):
        gene = variant.get("Gene", "NA")
        if gene and gene != "NA":
            groups[gene].append(i)
    return dict(groups)


def _reconstruct_full_context(context_slices: list[str]) -> str:
    """
    Reconstruct the full augmented context string from per-variant slices.

    Used only for display in the AUGMENTED CONTEXT section of the final output.
    Each slice contains a PATIENT DATA header + one VARIANT block; this helper
    emits the header once, then concatenates the VARIANT blocks from all slices.
    """
    if not context_slices:
        return ""

    m = _SLICE_HEADER_RE.match(context_slices[0])
    if not m:
        # Fallback: concatenate slices separated by a blank line
        return "\n\n".join(s.rstrip() for s in context_slices) + "\n"

    header     = m.group(1)
    header_len = len(header)

    variant_sections = [s[header_len:].rstrip() for s in context_slices]
    return header + "\n\n".join(variant_sections) + "\n"


def _build_gene_evidence_table(
    variants: list[dict],
    genome_build: str,
    gene_mode_cache: dict[str, str] | None = None,
    gene_mode_reasoning_cache: dict[str, str] | None = None,
) -> str:
    """
    One block per gene in the variant set: ClinVar P/LP missense:nonsense/FS
    ratio (the same counts that gate BP1/PP2), gnomAD pLI/LOEUF, this gene's
    inheritance-mode determination (with reasoning, if the mode was resolved
    by the LLM tier — see moi.build_gene_mode_cache), and this gene's
    representative variant's own gnomAD allele frequency + homozygote/
    heterozygote carrier counts (never present in the input CSV, regardless
    of whether a plain AF was already supplied there).

    pLI/LOEUF and the ClinVar counts are read from the tools' own class-level
    caches (populated during retrieval, one fetch per gene for the whole run)
    rather than re-fetched — cheap and guaranteed consistent with whatever
    the reasoning/conclusion stages actually saw. The variant-level gnomAD
    frequency is NOT cached anywhere else (it's variant-scoped, not
    gene-scoped) so it's fetched fresh here, once per gene's first variant.
    """
    gene_mode_cache = gene_mode_cache or {}
    gene_mode_reasoning_cache = gene_mode_reasoning_cache or {}
    reference_genome = _BUILD_TO_REFERENCE_GENOME.get(genome_build, "GRCh37")

    # First variant seen per gene — used both to key the gene and as the
    # representative variant for the live gnomAD frequency lookup below.
    gene_variants: dict[str, dict] = {}
    gene_order: list[str] = []
    for variant in variants:
        gene = variant.get("Gene", "NA")
        if gene and gene != "NA" and gene not in gene_variants:
            gene_variants[gene] = variant
            gene_order.append(gene)
    if not gene_order:
        return ""

    lines = []
    for gene in gene_order:
        variant    = gene_variants[gene]
        stats      = ClinVarGeneStatsTool._stats_cache.get(gene.upper())
        constraint = GnomadConstraintTool._constraint_cache.get(f"{gene.upper()}:{reference_genome}")

        lines.append(f"{gene}:")

        if stats:
            missense, nonsense = stats["missense"], stats["nonsense"]
            total = missense + nonsense
            fraction = f"{missense / total:.1%} missense" if total else "n/a"
            verdict = classify_consequence_counts(missense, nonsense)
            lines.append(
                f"  ClinVar P/LP missense:nonsense/FS = {missense}:{nonsense} ({fraction}) -> {verdict}"
            )
        else:
            lines.append("  ClinVar gene-level counts: not fetched")

        if constraint and (constraint["pli"] is not None or constraint["loeuf"] is not None):
            pli, loeuf = constraint["pli"], constraint["loeuf"]
            pli_str   = f"{pli:.3f} ({classify_pli(pli)})" if pli is not None else "not available"
            loeuf_str = f"{loeuf:.3f} ({classify_loeuf(loeuf)})" if loeuf is not None else "not available"
            lines.append(f"  gnomAD pLI = {pli_str}  |  LOEUF = {loeuf_str}")
        else:
            lines.append("  gnomAD constraint: not available")

        mode      = gene_mode_cache.get(gene, "")
        reasoning = gene_mode_reasoning_cache.get(gene, "")
        mode_label = moi.MODE_LABELS.get(mode, moi.MODE_LABELS[""])
        if reasoning:
            # Only resolved via the LLM tier (tiers 1/2 are unambiguous
            # structured facts and carry no reasoning text) — surface the
            # justification so the mode call is auditable, not a silent label.
            lines.append(f"  Inheritance mode (literature-reasoned): {mode_label} — {reasoning}")
        elif mode:
            lines.append(f"  Inheritance mode (CSV/CGD): {mode_label}")

        try:
            coords = parse_variant_coords(
                variant_str=variant.get("Variant", ""),
                chrom_field=variant.get("Chromosome", ""),
                pos_field=variant.get("Position", ""),
                ref_field=variant.get("Ref_seq", ""),
                alt_field=variant.get("Var_seq", ""),
            )
        except ValueError:
            coords = None
        if coords:
            try:
                freq = fetch_variant_frequency(*coords, genome_build)
            except (ToolFetchError, ToolParseError) as e:
                lines.append(f"  gnomAD variant AF ({variant.get('HGVS', '?')}): fetch failed — {e}")
                freq = None
            if freq:
                if freq["error"]:
                    lines.append(f"  gnomAD variant AF ({variant.get('HGVS', '?')}): not found — {freq['error']}")
                else:
                    for key, label in (("genome", "Genome"), ("exome", "Exome")):
                        sub = freq.get(key)
                        if sub:
                            hom = sub["homozygote_count"]
                            het = sub["heterozygote_count"]
                            lines.append(
                                f"  gnomAD {label} AF = {sub['af']:.6g} "
                                f"(homozygotes={hom if hom is not None else 'n/a'}, "
                                f"heterozygotes={het if het is not None else 'n/a'})"
                            )
        else:
            lines.append("  gnomAD variant AF: coordinates not resolvable for this variant")

        lines.append("")

    return "\n".join(lines).rstrip()


def _build_process_details(
    process_log: list[dict],
    variants: list[dict],
) -> str:
    """
    Build the PROCESS DETAILS section from the executor's process log.
    One block per variant, one entry per tool.
    """
    SEP_VARIANT = "─" * 56
    lines = []

    by_variant: dict[int, list[dict]] = defaultdict(list)
    for entry in process_log:
        by_variant[entry["variant_index"]].append(entry)

    for i, variant in enumerate(variants):
        gene  = variant.get("Gene", "?")
        hgvs  = variant.get("HGVS", variant.get("Variant", "?"))
        lines.append(f"\n{SEP_VARIANT}")
        lines.append(f"VARIANT {i + 1} — {gene}  {hgvs}")
        lines.append(SEP_VARIANT)

        entries = by_variant.get(i, [])
        if not entries:
            lines.append("  (no tool entries recorded)")
            continue

        for entry in entries:
            tool_name = entry["tool_name"].upper()
            gate      = entry["gate"]

            if gate == "SKIP":
                reason = entry.get("gate_reason") or ""
                lines.append(f"\n[{tool_name}]  GATE: SKIPPED  ({reason})")
                continue

            lines.append(f"\n[{tool_name}]  GATE: PASS")

            metadata = entry.get("metadata") or {}

            # ── LitVar2 metadata ────────────────────────────────────────
            if "query" in metadata:
                query   = metadata.get("query") or "?"
                n_found = metadata.get("n_found") or 0
                kept    = metadata.get("kept") or []
                lines.append(f"  Query     : {query}")
                lines.append(f"  Articles  : {len(kept)} kept from {n_found} found")
                if kept:
                    lines.append("  Kept articles:")
                    for art in kept:
                        lines.append(f"    [PMID:{art['pmid']}] {art['title']}")

            # ── WebSearch trace ──────────────────────────────────────────
            if "trace" in metadata:
                trace = metadata.get("trace") or []
                if trace:
                    lines.append("  ReAct trace:")
                    for step in trace:
                        lines.append(f"    {step}")

            # ── Raw output ───────────────────────────────────────────────
            raw = entry.get("raw_output")
            if raw:
                lines.append("  Raw output:")
                for raw_line in raw.splitlines():
                    lines.append(f"    {raw_line}")
            else:
                lines.append("  Output: (none)")

    return "\n".join(lines)




# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class Pipeline:
    """
    Full variant analysis pipeline.

    Instantiate once at server startup; call run() per request.

    Args:
        model_name: Key in pipeline.llm.registry.MODELS (default "qwen3.5-9b").
        llm_kwargs: Forwarded to get_client() — e.g. base_url for VLLMClient.
    """

    def __init__(
        self,
        model_name: str = "qwen3.5-9b",
        llm_kwargs: Optional[dict] = None,
        llm=None,                              # pre-built LLMClient (overrides model_name)
    ):
        self._model_name = model_name
        self._llm_kwargs = llm_kwargs or {}

        # ── LLM client (loaded once at startup) ──────────────────────────────
        if llm is not None:
            logger.info("[Pipeline] Using pre-built LLM client: %s", type(llm).__name__)
            self._llm = llm
        else:
            logger.info("[Pipeline] Loading LLM client: %s", model_name)
            self._llm = get_client(model_name, **self._llm_kwargs)

        # ── Manifests ─────────────────────────────────────────────────────────
        loader = ManifestLoader()
        self._manifests = loader.load_all()
        logger.info(
            "[Pipeline] Loaded %d manifests: %s",
            len(self._manifests),
            [m["name"] for m in self._manifests],
        )

        # ── Tool instances (one per pipeline tool) ────────────────────────────
        self._tools = [
            LitVar2SummaryTool(),
            SpliceAITool(),
            AutoPVS1Tool(),
            WebSearchAgentTool(),
            GnomadConstraintTool(),
            GnomadFrequencyTool(),
            ClinVarGeneStatsTool(),
            ClinVarResidueSearchTool(),
            ClinVarHotspotTool(),
            UniProtDomainTool(),
            ClinGenAlleleTool(),
            GeneReviewsTool(),
        ]
        # Direct reference for the phenotype-agnostic condition-inventory search
        # (Stage 1b, pipeline.run()) — called outside the manifest/gate system
        # since it isn't per-variant-gated the way the manifest-driven litvar2
        # call is, but reusing the same instance keeps its class-level CGD
        # cache shared.
        self._litvar2_tool = next(t for t in self._tools if isinstance(t, LitVar2SummaryTool))

        # ── Executor ──────────────────────────────────────────────────────────
        self._executor = Executor(
            llm=self._llm,
            tools=self._tools,
            manifests=self._manifests,
        )
        logger.info("[Pipeline] Executor ready with %d tools.", len(self._executor._paired))

    # ── public ────────────────────────────────────────────────────────────────

    async def run(
        self,
        variants: list[dict],
        patient_phenotype: str,
        genome_build: str = DEFAULT_GENOME_BUILD,
        raw_rows: list[dict] | None = None,
        parental_ab: list[dict] | None = None,
        header_mapping_summary: str | None = None,
    ) -> str:
        """
        Execute the full per-variant pipeline and return the combined output.

        Pipeline flow:
            Stage 1   — retrieval:        variants → list[str] (one context slice per variant)
            Triage    — (if n > threshold) kept/discarded per variant
            Stage 2   — reasoning:        kept slices → dict[int, str]
            Tier select — split kept variants into included/excluded via inclusion_decisions
            Stage 3.5 — cross_analysis:   per gene with ≥2 tier-1 variants → dict[gene, str]
            Stage 4   — conclusion:        tier-1 variants → dict[int, str]
            Stage 5   — final_conclusion:  tier-1 conclusions → str

        When n <= TRIAGE_ENABLED_THRESHOLD the pipeline behaves identically to
        the no-triage path: all variants go through all stages, and the
        Tier 2/Tier 3 appendix sections are omitted from the output.

        Args:
            variants:          List of normalized variant dicts from normalize_upload().
            patient_phenotype: Free-text patient phenotype string from the request.
            genome_build:      Reference genome build for this run ("hg19" or "hg38").
            raw_rows:          Full original CSV rows, one dict per variant.
            parental_ab:       Per-variant allelic balance dicts from normalizer,
                               or None if no parental data available.
            header_mapping_summary: Human-readable text from normalizer.normalize_upload()
                               showing what the SLM understood the CSV header to mean —
                               surfaced in the final output as its own section, or None.

        Returns:
            Combined output string with PROCESS DETAILS, AUGMENTED CONTEXT,
            REASONING, and FINAL REPORT sections separated by === banners.

        Raises:
            ValueError if variants is empty.
        """
        if not variants:
            raise ValueError("variants list is empty — nothing to analyze.")

        n = len(variants)
        logger.info("[Pipeline] Processing %d variants.", n)

        SEP         = "=" * 60
        SEP_VARIANT = "─" * 56

        # ── Stage 1: Retrieval ────────────────────────────────────────────────
        context_slices: list[str] = retrieval.run(
            variants=variants,
            patient_phenotype=patient_phenotype,
            executor=self._executor,
            genome_build=genome_build,
            raw_rows=raw_rows,
            parental_ab=parental_ab,
        )

        process_details = _build_process_details(self._executor.process_log, variants)

        # ── Proband allelic-balance artifact gate (deterministic, pre-triage) ───
        # A variant whose own Zygosity field claims a called genotype (het/hom/
        # hemi/...) but whose own proband Allelic_balance is anomalously low
        # (<0.1, segregation.classify_ab_ratio's "absent" bucket) has essentially
        # zero alt-allele read support in THIS patient — that is a sequencing
        # artifact or miscall, not a real variant, regardless of how strong its
        # ACMG evidence looks. This must be excluded before any SLM call, not
        # left as a prose caveat for a later synthesis step to reason past.
        # A real past failure: a compound-het pair's second variant had
        # Allelic_balance_proband=0.00 (Qual_comment=Poor in the source CSV,
        # itself never mapped/surfaced downstream) yet was still carried
        # through to the final report as a causative co-finding — the artifact
        # caveat reached the SLM only as prose ("AB=0.00... suggests artifact"),
        # and the synthesis step reasoned past it anyway ("the joint
        # classification overrides this for the purpose of the diagnosis").
        # Removing the variant here leaves nothing left to reason past.
        artifact_indices: set[int] = set()
        for i, v in enumerate(variants):
            zygosity = str(v.get("Zygosity", "") or "").strip().lower()
            if zygosity in ("", "na", "none"):
                continue
            if segregation.classify_ab_ratio(v.get("Allelic_balance")) == "absent":
                artifact_indices.add(i)
                logger.info(
                    "[Pipeline] Proband AB artifact: variant %d (%s) Zygosity=%s "
                    "but Allelic_balance=%s (<0.1) — excluding as sequencing "
                    "artifact/miscall, not a real finding in this patient.",
                    i + 1, v.get("Gene", "?"), v.get("Zygosity", "?"),
                    v.get("Allelic_balance", "?"),
                )

        # ── ACMG Secondary Findings (SF v3.2) actionable-gene detection ─────────
        # Computed on ALL variants (not just kept ones) right after retrieval, so
        # it can force-exempt flagged variants from triage/second-triage discard
        # below — a P/LP variant in an SF gene must be reported regardless of
        # phenotype fit, so it must survive both filtering stages irrespective of
        # what the phenotype-driven KEEP/INCLUDE machinery would otherwise decide.
        litvar2_raw_by_variant: dict[int, str | None] = {
            entry["variant_index"]: entry.get("raw_output")
            for entry in self._executor.process_log
            if entry["tool_name"] == "litvar2_summary" and entry["gate"] == "PASS"
        }
        # clingen_allele already resolved this variant against the ClinGen
        # Allele Registry earlier in retrieval (CAid + ClinVar variation
        # ID/RCV cross-reference) — reuse that existing fetch as a second
        # fallback source alongside litvar2 rather than re-fetching anything,
        # for the case where the CSV's own ClinVar_class column is missing or
        # was mismapped by the header interpreter (real observed case: a
        # dated column name like "ClinVar.20230813..Stars..Conflict_details."
        # went unmapped while an unrelated empty column was mapped to
        # ClinVar_class instead, silently losing a Pathogenic ACMG SF gene call).
        clingen_raw_by_variant: dict[int, str | None] = {
            entry["variant_index"]: entry.get("raw_output")
            for entry in self._executor.process_log
            if entry["tool_name"] == "clingen_allele" and entry["gate"] == "PASS"
        }
        # clinvar_gene_stats already fetched this variant's own real
        # per-submission ClinVar classification tally (NCBI efetch VCV
        # record) earlier in retrieval — deterministic ground truth,
        # independent of the CSV's own (SLM-mapped, sometimes wrong)
        # ClinVar_class column. See acmg_sf.build_actionable_set's docstring
        # for the case this was added for.
        clinvar_tally_raw_by_variant: dict[int, str | None] = {
            entry["variant_index"]: entry.get("raw_output")
            for entry in self._executor.process_log
            if entry["tool_name"] == "clinvar_gene_stats" and entry["gate"] == "PASS"
        }
        actionable_indices, actionable_reasons, actionable_classifications = acmg_sf.build_actionable_set(
            variants=variants,
            litvar2_raw_by_variant=litvar2_raw_by_variant,
            clingen_raw_by_variant=clingen_raw_by_variant,
            clinvar_tally_raw_by_variant=clinvar_tally_raw_by_variant,
            llm=self._llm,
        )
        if actionable_indices:
            logger.info(
                "[Pipeline] ACMG SF actionable variant(s) flagged: %s",
                {i + 1: variants[i].get("Gene", "?") for i in actionable_indices},
            )

        # ── Triage ────────────────────────────────────────────────────────────
        # triage_results maps original variant index → ("KEEP"|"DISCARD", justification)
        triage_results: dict[int, tuple[str, str]] = {}
        triage_ran = n > TRIAGE_ENABLED_THRESHOLD

        if triage_ran:
            # Identify multi-variant genes (compound het candidates) — exempt from discard
            gene_groups_all = _group_by_gene(variants)
            multi_gene_indices: set[int] = set()
            for gene, idxs in gene_groups_all.items():
                if len(idxs) >= 2:
                    multi_gene_indices.update(idxs)

            def _triage_one(i: int) -> tuple[int, str, str]:
                decision, justification = first_triage.run_one(
                    variant_context=context_slices[i],
                    patient_phenotype=patient_phenotype,
                    llm=self._llm,
                )
                return i, decision, justification

            workers = min(MAX_WORKERS, n)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_triage_one, i): i for i in range(n)}
                for future in as_completed(futures):
                    i, decision, justification = future.result()
                    gene = variants[i].get("Gene", "?")
                    if i in artifact_indices:
                        # Unconditional — takes precedence over every KEEP
                        # override below (ACMG SF, compound-het, LoF-in-gene):
                        # a variant not actually present in the proband cannot
                        # be a real actionable/causative/compound-het finding
                        # no matter what those overrides would otherwise argue.
                        justification = (
                            f"[proband AB artifact — Allelic_balance="
                            f"{variants[i].get('Allelic_balance', '?')} contradicts "
                            f"Zygosity={variants[i].get('Zygosity', '?')}; excluded "
                            f"as sequencing artifact/miscall, not a real finding in "
                            f"this patient]"
                        )
                        decision = "DISCARD"
                    elif decision == "DISCARD" and i in actionable_indices:
                        logger.info(
                            "[Pipeline] Triage override: variant %d (%s) is an ACMG SF "
                            "actionable finding — forced KEEP", i + 1, gene,
                        )
                        justification = (
                            f"[ACMG SF actionable exempt — original triage: DISCARD: {justification}]"
                        )
                        decision = "KEEP"
                    elif decision == "DISCARD" and i in multi_gene_indices:
                        logger.info(
                            "[Pipeline] Triage override: variant %d is compound-het "
                            "candidate in gene %s — forced KEEP", i + 1, gene,
                        )
                        justification = (
                            f"[compound-het exempt — original triage: DISCARD: {justification}]"
                        )
                        decision = "KEEP"
                    elif (
                        decision == "DISCARD"
                        and _is_lof_variant(variants[i])
                        and _gene_phenotype_relevant(context_slices[i])
                    ):
                        logger.info(
                            "[Pipeline] Triage override: variant %d (%s) is a LoF variant "
                            "in a phenotype-relevant gene — forced KEEP", i + 1, gene,
                        )
                        justification = (
                            f"[LoF-in-phenotype-gene exempt — original triage: DISCARD: {justification}]"
                        )
                        decision = "KEEP"
                    triage_results[i] = (decision, justification)
        else:
            # No triage — all variants kept, EXCEPT proband-AB artifacts, which
            # are excluded unconditionally regardless of the triage threshold.
            for i in range(n):
                if i in artifact_indices:
                    triage_results[i] = (
                        "DISCARD",
                        f"[proband AB artifact — Allelic_balance="
                        f"{variants[i].get('Allelic_balance', '?')} contradicts "
                        f"Zygosity={variants[i].get('Zygosity', '?')}; excluded "
                        f"as sequencing artifact/miscall, not a real finding in "
                        f"this patient]",
                    )
                else:
                    triage_results[i] = ("KEEP", "")

        kept_indices     = [i for i, (dec, _) in triage_results.items() if dec == "KEEP"]
        discarded_indices = [i for i, (dec, _) in triage_results.items() if dec == "DISCARD"]
        logger.info(
            "[Pipeline] Triage complete: %d kept, %d discarded.",
            len(kept_indices), len(discarded_indices),
        )

        # ── Compound-het gene grouping (AR / AD-AR genes with ≥2 kept variants) ─
        # Computed before reasoning (Stage 2a) so that sibling variant evidence
        # can be injected into each variant's FIRST reasoning call, not just
        # second_triage — Call 1 should already weigh the variant jointly with
        # its gene-mate instead of reasoning about it in isolation.
        # Scoped to recessive-relevant inheritance only — a second variant in a
        # purely dominant gene doesn't change the first variant's standing.
        # Inheritance-mode classification and compound-het gene grouping live in
        # pipeline/core/moi.py (relocated so the MOI-layer stage modules can
        # import this logic directly) — same behavior, just no longer inline here.
        kept_set = set(kept_indices)
        gene_mode_cache, gene_mode_reasoning_cache = moi.build_gene_mode_cache(
            variants, kept_indices, context_slices, _group_by_gene, self._llm,
        )
        recessive_gene_groups = moi.build_recessive_gene_groups(
            variants, gene_mode_cache, kept_indices, _group_by_gene,
        )

        # Layer 1 (MOI-layer restructuring): per-variant phenotype pertinence,
        # sourced from the LitVar2 evidence already in each context slice — not
        # yet consumed by any scoring stage (that wiring lands with the Layer 2
        # base-ACMG stage, where PVS1's phenotype-fit gate will read this value
        # as a backend-determined fact instead of re-deriving it from prose).
        pertinence_cache = moi.build_pertinence_cache(variants, kept_indices, context_slices)
        logger.debug(
            "[Pipeline] Pertinence cache: %s",
            {variants[i].get("Gene", "?"): v for i, v in pertinence_cache.items()},
        )

        # Segregation classification, computed once per KEPT variant (not just
        # included) — needed up front so the deterministic CIS/TRANS phase
        # check below can run BEFORE reasoning/second-triage, not just for the
        # later MOI Layers 3/4/5 (which only ever saw included pairs). Scoped
        # to kept_indices ⊇ include_indices, so this single cache serves both;
        # classify_segregation() only depends on parental_ab[i], not on
        # inclusion status, so computing it early for the wider set is safe.
        segregation_cache: dict[int, str] = {}
        for i in kept_indices:
            ab_entry = parental_ab[i] if parental_ab else {}
            segregation_cache[i] = segregation.classify_segregation(
                ab_entry.get("proband"), ab_entry.get("mother"), ab_entry.get("father"),
            )

        _PHASE_FACT_LABELS = {
            "cis": (
                "CIS (same parent as this variant) — the two variants sit on the "
                "SAME parental allele; the other parental allele is wild-type at "
                "both positions. This is NOT compound heterozygous, no matter how "
                "strong either variant's individual evidence looks. Do not describe "
                "this pair as compound heterozygous or apply a biallelic/recessive "
                "model to it."
            ),
            "trans": (
                "TRANS (different parents) — consistent with compound heterozygous "
                "biallelic inheritance."
            ),
            "unknown": (
                "UNKNOWN — phase not determinable from available parental "
                "allelic-balance data. A compound-heterozygous model may still apply "
                "per biallelic evidence, but phase uncertainty must be noted "
                "explicitly, not asserted as confirmed trans."
            ),
        }

        def _phase_fact(i: int, j: int) -> str:
            """Backend-determined CIS/TRANS/UNKNOWN phase between variants i and j,
            from classify_phase() over the already-computed segregation_cache —
            handed to the model as a stated fact (same pattern as
            _inheritance_mode_block) instead of letting it re-derive phase itself
            from raw allelic-balance numbers, which can disagree between two
            separate LLM calls reasoning about the same pair from either side."""
            phase = segregation.classify_phase(segregation_cache.get(i, ""), segregation_cache.get(j, ""))
            return f"Phase vs this variant (backend-determined): {_PHASE_FACT_LABELS[phase]}"

        def _inheritance_mode_block(i: int) -> str:
            """Backend-determined gene inheritance mode for variant i, handed to the
            reasoning prompt as a stated fact so the model reports it rather than
            re-deriving (and possibly mis-deriving) it from prose evidence."""
            gene  = variants[i].get("Gene", "NA")
            mode  = gene_mode_cache.get(gene, "")
            label = moi.MODE_LABELS.get(mode, moi.MODE_LABELS[""])
            return f"\n--- GENE INHERITANCE MODE (backend-determined) ---\nGene: {gene} | Mode: {label}\n"

        def _pertinence_block(i: int) -> str:
            """Backend-determined gene-phenotype literature pertinence for
            variant i, appended to the reasoning text handed into the ACMG
            base-scoring stage (conclusion.py) as a stated fact — a hard
            negative constraint on Phenotype fit/PVS1 rather than letting the
            model claim "match" from prose alone when the backend LitVar2
            search found no gene-phenotype link at all. Injected via the
            existing `reasoning` argument (not a new run_one() parameter) so
            server_qwen.py's direct-mode monkeypatch of conclusion.run_one,
            which wraps the exact (variant_context, reasoning, cross_analysis,
            llm) signature for SSE progress ticks, keeps working unmodified."""
            gene  = variants[i].get("Gene", "NA")
            value = pertinence_cache.get(i)
            label = moi.PERTINENCE_LABELS[value]
            return f"\n--- GENE-PHENOTYPE LITERATURE PERTINENCE (backend-determined) ---\n{label}\n"

        def _zygosity_note_block(i: int) -> str:
            """Backend-determined caveat for apparent homozygosity on a chrX
            gene. The canonical schema has no patient-sex field, so a
            "Homozygous" Zygosity call on chrX is systematically ambiguous:
            it's frequently a hemizygous male call (single X allele, ~100%
            VAF) that the upstream variant caller reports as "Homozygous"
            because only one allele is observed — indistinguishable from
            true biallelic homozygosity without sex information. Layer 6
            (moi_xlinked.py) already treats hemizygous/homozygous as
            equally consistent with XLR via AB-derived pattern, but Layers
            1-2 (reasoning/second_triage/conclusion) run first and see the
            raw Zygosity field with no such caveat — without this note the
            model can treat "true" homozygosity as improbable for a rare
            X-linked variant (e.g. absent consanguinity) and downweight or
            exclude the variant on that basis alone. Handed to the prompts
            as a stated fact, same pattern as _inheritance_mode_block."""
            zyg = str(variants[i].get("Zygosity", "") or "").strip().lower()
            if "hom" not in zyg:
                return ""
            if moi.gene_chromosome(variants, [i]) != "X":
                return ""
            gene = variants[i].get("Gene", "NA")
            return (
                "\n--- ZYGOSITY CAVEAT (backend-determined) ---\n"
                f"Gene {gene} is on chrX and Zygosity reads Homozygous, but patient "
                "sex is not part of this dataset. A 'Homozygous' call on chrX is "
                "commonly a hemizygous male variant (single X allele, ~100% VAF) "
                "reported as 'Homozygous' because only one allele is seen — "
                "indistinguishable from true homozygosity without sex information. "
                "Treat this as a plausible hemizygous-male hit, not as evidence "
                "against the variant. Do NOT discard, exclude, or downweight this "
                "variant on the grounds that true homozygosity would be improbable.\n"
            )

        def _evidence_only_context(i: int) -> str:
            """Strip the PATIENT DATA header off context_slices[i], leaving only
            this variant's own retrieved tool evidence — same stripping
            _sibling_evidence_block below already relies on (there, applied to
            a sibling's slice; here, applied to the variant's own slice), used
            for the isolated gene-phenotype-extraction call below, which must
            never see the patient's own phenotype text."""
            m = _SLICE_HEADER_RE.match(context_slices[i])
            return context_slices[i][m.end():].rstrip() if m else context_slices[i].rstrip()

        def _gene_phenotype_block(i: int) -> str:
            """Backend-extracted gene condition list for variant i (isolated
            call, evidence-only — see phenotype_list_cache below), handed to
            the reasoning prompt as a stated fact so Step 1 only has to judge
            CLUSTER_PHENOTYPE by comparing the patient's phenotype against this
            pre-compiled list, never re-derive or fill it in itself. A single
            combined call asked to do both reliably conflated the two (wrote
            the patient's own presenting complaint into the condition list, or
            silently dropped an unrelated-but-real condition) — see
            reasoning.run_gene_phenotype_extraction's docstring."""
            return (
                "\n--- GENE PHENOTYPE LIST (backend-extracted from retrieved "
                "evidence only, computed before this patient's phenotype was "
                "ever shown to that step — authoritative, do not re-derive or "
                "add to it) ---\n"
                f"PHENOTYPE: {phenotype_list_cache.get(i, 'NA')}\n"
                + _overlap_block(i)
            )

        def _per_condition_lines(i: int) -> str:
            return "\n".join(
                l for l in overlap_text_cache.get(i, "").splitlines()
                if l.lstrip().startswith("- ")
            )

        def _overlap_block(i: int) -> str:
            """Stage 1c's result, as an input to Stage 1 reasoning."""
            if i not in overlap_verdict_cache:
                return ""
            return (
                "\n--- PHENOTYPE OVERLAP (backend: this patient's phenotype vs. each "
                "listed condition and its NCBI MedGen/HPO clinical features) ---\n"
                f"{_per_condition_lines(i)}\n"
                f"OVERLAP VERDICT: {overlap_verdict_cache[i]}\n"
            )

        def _sibling_evidence_block(i: int) -> str:
            """Sibling block built from raw retrieval evidence (Call 1 — no reasoning
            exists yet). Skipped only when this variant is a CONFIRMED homozygous
            call in an AR/XLR gene — that already explains the disease on its own
            and doesn't need a compound-het partner. Zygosity is otherwise left to
            the model (it sees the ALLELIC BALANCE block and is instructed to
            derive zygosity from it when Zygosity is NA)."""
            gene = variants[i].get("Gene", "NA")
            if moi.zygosity_is_confirmed_hom(variants[i].get("Zygosity", "")):
                return ""
            siblings = [j for j in recessive_gene_groups.get(gene, []) if j != i]
            if not siblings:
                return ""
            parts = ["\n--- SIBLING VARIANT(S) IN SAME GENE (autosomal recessive / X-linked recessive) ---"]
            for j in siblings:
                sib = variants[j]
                m = _SLICE_HEADER_RE.match(context_slices[j])
                sib_evidence = context_slices[j][m.end():].rstrip() if m else context_slices[j].rstrip()
                parts.append(
                    f"\nSibling Variant {j + 1} — {sib.get('Gene', '?')} "
                    f"{sib.get('HGVS', sib.get('Variant', '?'))} "
                    f"(zygosity: {sib.get('Zygosity', 'NA')})\n"
                    f"{_phase_fact(i, j)}\n"
                    f"Sibling evidence:\n{sib_evidence}"
                )
            return "\n".join(parts) + "\n"

        # ── Stage 1b: Gene-phenotype-list extraction (isolated, evidence-only) ──
        # Runs before reasoning so its result can be handed to Step 1 as a
        # stated fact (_gene_phenotype_block). Deliberately a separate SLM call
        # per kept variant, given _evidence_only_context(i) — never
        # context_slices[i] itself, which still carries the PATIENT DATA header
        # — so this call has no patient phenotype text to conflate with the
        # gene's own retrieved condition list. See
        # reasoning.run_gene_phenotype_extraction's docstring for the failure
        # this fixes.
        #
        # Also runs the phenotype-agnostic condition-inventory search
        # (litvar2.run_condition_inventory) and appends it to the evidence —
        # _evidence_only_context(i) alone still carries whatever the manifest-
        # driven litvar2_summary tool retrieved during Stage 1 Retrieval, and
        # that tool's own relevance selection is anchored to the patient's
        # phenotype (see run_condition_inventory's docstring) — a real,
        # well-documented condition for this gene that OMIM/CGD doesn't
        # separately catalog and that doesn't overlap the patient's phenotype
        # would otherwise never reach this call at all, regardless of how
        # this call's own prompt is worded. Passed a ToolContext with
        # patient_phenotype="" (not the real value) as defense in depth —
        # this call path has no legitimate use for it.
        phenotype_list_cache: dict[int, str] = {}

        def _extract_phenotype_one(i: int) -> tuple[int, str]:
            gene = variants[i].get("Gene", "NA")
            evidence = _evidence_only_context(i)
            if gene != "NA":
                inventory_context = ToolContext(
                    variant=variants[i],
                    patient_phenotype="",
                    all_outputs={},
                    variant_report="",
                    variant_index=i,
                    total_variants=n,
                    llm=self._llm,
                )
                try:
                    condition_inventory = self._litvar2_tool.run_condition_inventory(
                        gene, inventory_context,
                    )
                except Exception as exc:
                    logger.warning(
                        "[Pipeline] Condition-inventory search failed for variant %d (%s): %s",
                        i + 1, gene, exc,
                    )
                    condition_inventory = None
                if condition_inventory:
                    evidence = evidence + "\n\n" + condition_inventory
            return i, reasoning.run_gene_phenotype_extraction(
                evidence_context=evidence, llm=self._llm,
            )

        workers = min(MAX_WORKERS_LLM, len(kept_indices))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_extract_phenotype_one, i): i for i in kept_indices}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    i, phenotype_list = future.result()
                except Exception as exc:
                    logger.error(
                        "[Pipeline] Gene-phenotype extraction failed for variant %d: %s",
                        i + 1, exc,
                    )
                    phenotype_list = "NA"
                phenotype_list_cache[i] = phenotype_list

        # ── Stage 1c: isolated phenotype overlap (before reasoning) ─────────
        # Patient phenotype vs. the gene's condition list (Stage 1b) with each
        # condition's MedGen/HPO clinical features — nothing else. Its per-
        # condition matches and verdict are an INPUT to Stage 1 reasoning
        # (appended to _gene_phenotype_block), which weighs them with all the
        # other evidence. Real observed failure: judged only inside Stage 1,
        # literature about one of a gene's conditions overrode a matching
        # condition on its own list, and an eponymous syndrome's retinal
        # features were never known at all.
        overlap_verdict_cache: dict[int, str] = {}
        overlap_text_cache: dict[int, str] = {}

        def _overlap_one(i: int) -> tuple[int, str | None, str]:
            verdict, text = reasoning.run_phenotype_overlap(
                patient_phenotype, phenotype_list_cache.get(i, "NA"), self._llm,
            )
            return i, verdict, text

        workers = min(MAX_WORKERS_LLM, len(kept_indices))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_overlap_one, i): i for i in kept_indices}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    i, verdict, text = future.result()
                except Exception as exc:
                    logger.warning(
                        "[Pipeline] Phenotype-overlap call failed for variant %d: %s", i + 1, exc,
                    )
                    continue
                if verdict is not None:
                    overlap_verdict_cache[i] = verdict
                    overlap_text_cache[i] = text

        # ── Stage 2a: Reasoning (only on kept variants) ───────────────────────
        # Split into its own wave (rather than one combined reasoning+second_triage
        # call per variant) so that, for compound-het candidate genes, every kept
        # variant's reasoning is available before any of them proceeds to
        # second_triage — second_triage needs to see sibling variants' evidence.
        reasoning_only: dict[int, str] = {}
        reasoning_failed: set[int] = set()

        def _reason_one(i: int) -> tuple[int, str]:
            sib_block  = _sibling_evidence_block(i)
            mode_block = _inheritance_mode_block(i) + _zygosity_note_block(i)
            return i, reasoning.run_reasoning(
                variant_context=context_slices[i], llm=self._llm,
                sibling_context_block=sib_block,
                inheritance_mode_block=mode_block,
                gene_phenotype_block=_gene_phenotype_block(i),
            )

        workers = min(MAX_WORKERS_LLM, len(kept_indices))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_reason_one, i): i for i in kept_indices}
            for future in as_completed(futures):
                i = futures[future]
                gene = variants[i].get("Gene", "?")
                try:
                    i, r = future.result()
                except Exception as exc:
                    logger.error(
                        "[Pipeline] Reasoning failed for variant %d (%s): %s — marking EXCLUDE",
                        i + 1, gene, exc,
                    )
                    reasoning_only[i] = f"[REASONING FAILED: {exc}]"
                    reasoning_failed.add(i)
                    continue
                reasoning_only[i] = r

        # Backend-extracted CLUSTER_PHENOTYPE verdict (YES/PARTIAL/NONE/INCIDENTAL/
        # UNKNOWN), parsed once from each variant's own Stage 1 reasoning output
        # (its paired PHENOTYPE tag comes from phenotype_list_cache instead —
        # Stage 1b, an earlier and separate isolated call, see above). Computed
        # here (not re-derived by second_triage/conclusion) for the same reason
        # phase and gene inheritance mode are backend-determined facts: a
        # judgment made once and restated verbatim is followed far more
        # reliably than the same judgment left buried in prose and re-derived
        # by each downstream stage on its own — which in practice produced
        # inconsistent verdicts call to call for the same variant.
        cluster_match_cache: dict[int, str] = {
            i: reasoning.parse_cluster_match(reasoning_only[i])
            for i in kept_indices if i not in reasoning_failed
        }
        # Stage 1 was given the overlap verdict as an input and told to use it;
        # enforce it, so no later stage ever sees two different verdicts.
        for i in list(cluster_match_cache):
            ov = overlap_verdict_cache.get(i)
            if ov is None or ov == cluster_match_cache[i]:
                continue
            logger.info(
                "[Pipeline] Variant %d (%s): Stage 1 wrote CLUSTER_PHENOTYPE %s, "
                "backend overlap is %s — corrected", i + 1,
                variants[i].get("Gene", "?"), cluster_match_cache[i], ov,
            )
            reasoning_only[i] = _CLUSTER_LINE_RE.sub(
                lambda m: f"{m.group(1)}{ov}", reasoning_only[i],
            )
            cluster_match_cache[i] = ov
        # phenotype_list_cache (Stage 1b, computed above, before reasoning ever
        # ran) is the authoritative PHENOTYPE source — it was extracted by an
        # isolated call that never saw the patient's phenotype, so it cannot
        # be conflated with it. Not re-parsed from reasoning_only[i] here.

        _CLUSTER_MATCH_LABELS = {
            "YES": (
                "YES — the phenotype assessment affirmatively linked this gene to the "
                "patient's phenotype essentially as a whole."
            ),
            "PARTIAL": (
                "PARTIAL — the phenotype assessment affirmatively linked this gene to AT "
                "LEAST ONE distinct cluster of the patient's phenotype; it is "
                "silent on (not contradicting) at least one other cluster. This is "
                "a POSITIVE, sufficient gene-phenotype link — treat it the same as "
                "YES for inclusion/PVS1/Phenotype-fit purposes ONLY. Do NOT write "
                "'no gene-phenotype link', 'mismatch', or similar language for this "
                "variant on phenotype grounds. EXCEPTION — PP4 specifically: PARTIAL "
                "is NOT sufficient for PP4 (see conclusion.txt's PP4 condition (3), "
                "'Full coverage') — PP4 requires a YES verdict. A gene that only "
                "explains one cluster of a multi-cluster phenotype is not 'highly "
                "specific' for what this patient actually has, even though it is "
                "still a real enough link to keep the variant under consideration."
            ),
            "NONE": (
                "NONE — the phenotype assessment found no cluster of the patient's "
                "phenotype that this gene's evidence overlaps, and no other "
                "well-established condition for this gene either."
            ),
            "INCIDENTAL": (
                "INCIDENTAL — the phenotype assessment found this gene affirmatively "
                "linked to a real, well-established, named condition that has "
                "ZERO overlap with any cluster of the patient's presenting "
                "phenotype — a different disease entirely (see the PHENOTYPE "
                "tag below for its name). This variant is NEVER causative for "
                "this patient regardless of its ACMG score — it belongs only "
                "under Actionable/Incidental findings (section 3), never in "
                "the causative section (section 2). Do NOT let a high "
                "PVS1/PM2/PS2 score override this verdict, and do NOT re-derive "
                "or second-guess it from the reasoning narrative."
            ),
            "UNKNOWN": (
                "UNKNOWN — Stage 1 reasoning did not produce a clear verdict; judge "
                "phenotype fit from the reasoning narrative below as a fallback."
            ),
        }

        def _cluster_match_block(i: int) -> str:
            """Backend-extracted Stage 1 CLUSTER_PHENOTYPE verdict + PHENOTYPE
            tag for variant i, handed to second_triage, conclusion, and every
            MOI-layer stage as a stated, closed fact — same pattern as
            _inheritance_mode_block/_phase_fact — instead of letting those
            stages re-read and re-judge the phenotype-fit narrative themselves
            each time. The short machine-parseable "CLUSTER_PHENOTYPE: X" /
            "PHENOTYPE: Y" lines are included verbatim (not just the long
            prose label) so a later stage (final_conclusion.py's INCIDENTAL
            cap/routing) can regex-match this fact directly out of whichever
            downstream block reproduces this injected text, the same
            reproduction pattern already observed for the prose label."""
            verdict = cluster_match_cache.get(i, "UNKNOWN")
            phenotype_tag = phenotype_list_cache.get(i, "NA")
            label = _CLUSTER_MATCH_LABELS[verdict]
            per_condition = _per_condition_lines(i)
            return (
                "\n--- CLUSTER_PHENOTYPE (backend-determined: patient phenotype vs. "
                "the gene's condition list and MedGen clinical features — "
                "authoritative, do not re-derive) ---\n"
                f"CLUSTER_PHENOTYPE: {verdict}\n"
                f"{label}\n"
                f"PHENOTYPE: {phenotype_tag}\n"
                + (f"Per-condition match:\n{per_condition}\n" if per_condition else "")
            )

        def _sibling_block(i: int) -> str:
            """Sibling block built from sibling reasoning text (Call 2 — post Call 1).
            Same confirmed-hom-only gate as _sibling_evidence_block."""
            gene = variants[i].get("Gene", "NA")
            if moi.zygosity_is_confirmed_hom(variants[i].get("Zygosity", "")):
                return ""
            siblings = [
                j for j in recessive_gene_groups.get(gene, [])
                if j != i and j not in reasoning_failed
            ]
            if not siblings:
                return ""
            parts = ["\n--- SIBLING VARIANT(S) IN SAME GENE (autosomal recessive / X-linked recessive) ---"]
            for j in siblings:
                sib = variants[j]
                parts.append(
                    f"\nSibling Variant {j + 1} — {sib.get('Gene', '?')} "
                    f"{sib.get('HGVS', sib.get('Variant', '?'))} "
                    f"(zygosity: {sib.get('Zygosity', 'NA')})\n"
                    f"{_phase_fact(i, j)}\n"
                    f"Sibling reasoning:\n{reasoning_only[j]}"
                )
            return "\n".join(parts) + "\n"

        # ── Stage 2b: Second triage (INCLUDE/EXCLUDE, with sibling context) ────
        reasonings: dict[int, str] = {}
        inclusion_decisions: dict[int, str] = {}
        second_triage_justifications: dict[int, str] = {}

        for i in reasoning_failed:
            reasonings[i] = reasoning_only[i]
            inclusion_decisions[i] = "EXCLUDE"
            second_triage_justifications[i] = "reasoning error"

        triage_targets = [i for i in kept_indices if i not in reasoning_failed]

        def _second_triage_one(i: int) -> tuple[int, str, str, str]:
            sib_block = _sibling_block(i)
            gene_i = variants[i].get("Gene", "NA")
            zyg_i = str(variants[i].get("Zygosity") or "").strip().lower()
            is_hom_or_hemi = "hom" in zyg_i or "hemi" in zyg_i
            is_purely_dominant_mode = gene_mode_cache.get(gene_i, "") in ("AD", "XLD")
            combined = reasoning.run_second_triage(
                variant_context=context_slices[i],
                reasoning_text=reasoning_only[i] + _zygosity_note_block(i) + _cluster_match_block(i),
                llm=self._llm,
                sibling_context_block=sib_block,
                inheritance_mode_block=_inheritance_mode_block(i),
                is_x_linked=moi.gene_chromosome(variants, [i]) == "X",
                include_single_hit_recessive=not (is_hom_or_hemi or is_purely_dominant_mode),
                include_compound_het_exception=bool(sib_block),
                include_literature_evidence_quality=bool(litvar2_raw_by_variant.get(i)),
            )
            decision, justification = reasoning.parse_inclusion_decision(combined)
            return i, combined, decision, justification

        workers = min(MAX_WORKERS_LLM, len(triage_targets))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_second_triage_one, i): i for i in triage_targets}
            for future in as_completed(futures):
                i = futures[future]
                gene = variants[i].get("Gene", "?")
                try:
                    i, combined, decision, justification = future.result()
                except Exception as exc:
                    logger.error(
                        "[Pipeline] Second triage failed for variant %d (%s): %s — marking EXCLUDE",
                        i + 1, gene, exc,
                    )
                    reasonings[i] = reasoning_only[i] + f"\n\n[SECOND TRIAGE FAILED: {exc}]"
                    inclusion_decisions[i] = "EXCLUDE"
                    second_triage_justifications[i] = f"second triage error: {exc}"
                    continue
                reasonings[i] = combined
                inclusion_decisions[i] = decision
                second_triage_justifications[i] = justification
                logger.info(
                    "[Pipeline] Variant %d (%s) second triage → %s", i + 1, gene, decision
                )

        # ── ACMG SF override: force INCLUDE for actionable variants that reached
        #     a second-triage decision (reasoning_failed ones are left EXCLUDEd —
        #     there is no reasoning text to build a conclusion/report block from). ─
        for i in actionable_indices:
            if i in reasoning_failed or i not in kept_indices:
                continue
            if inclusion_decisions.get(i) != "INCLUDE":
                logger.info(
                    "[Pipeline] Second-triage override: variant %d (%s) is an ACMG SF "
                    "actionable finding — forced INCLUDE", i + 1, variants[i].get("Gene", "?"),
                )
                second_triage_justifications[i] = (
                    f"[ACMG SF actionable exempt — original: {inclusion_decisions.get(i)}: "
                    f"{second_triage_justifications.get(i, '')}]"
                )
                inclusion_decisions[i] = "INCLUDE"

        # ── Compound-het second-triage override: a variant EXCLUDEd by second_triage
        #     that sits in a recessive-relevant gene (AR/XLR/AD_AR/XLD_XLR) with >=2
        #     kept variants must not be dropped on the SLM's own free-text cis/trans
        #     call — that determination belongs to the deterministic backend
        #     classify_phase() (segregation_cache, computed earlier from real
        #     parental allelic-balance data and handed to the model as a stated fact
        #     via _phase_fact). The SLM can still assert "cis" in its Exclude-case
        #     prose even when the backend fact it was given says phase is UNKNOWN
        #     (no parental AB data at all) — that is a hallucinated phase call, not
        #     a grounded one, and must not be trusted to drop a real compound-het
        #     candidate. Mirrors the first_triage compound-het override above, one
        #     stage later. Only let an EXCLUDE stand when the backend phase against
        #     EVERY kept sibling in the group is deterministically "cis"; any
        #     "trans" or "unknown" sibling relationship forces INCLUDE so the
        #     AR module (moi_recessive.py) — which requires phase == "trans" or
        #     "unknown" and hard-blocks "cis" — is the one to make the final call.
        #
        #     PHENOTYPE GATE (mandatory, checked before the phase check above):
        #     this override exists to rescue a real compound-het candidate from a
        #     hallucinated/premature cis call — it is not license to rescue a gene
        #     Stage 1 has already determined has NO overlap with the patient's
        #     phenotype at all. cluster_match_cache[i] is that same backend-parsed,
        #     authoritative Stage 1 verdict already used to gate PP4/PVS1 at the
        #     conclusion stage (see prompts/conclusion.txt's PHENOTYPE PERTINENCE
        #     CHECK) — a real observed failure: a myopathy gene was
        #     correctly EXCLUDEd by second_triage with cluster verdict NONE ("no
        #     cluster of the patient's phenotype overlaps"), but the compound-het
        #     override forced it back to INCLUDE anyway because phase alone wasn't
        #     confirmed cis, and it went on to be scored Pathogenic and reported as
        #     causative for a phenotype its own gene-disease evidence explicitly
        #     does not cover. Being a real biallelic pair is necessary but not
        #     sufficient — the gene must also plausibly explain the phenotype.
        #     A pair with NONE (or INCIDENTAL) on both variants gets no
        #     phase-based rescue at all.
        for gene, idxs in recessive_gene_groups.items():
            for i in idxs:
                if i in reasoning_failed or inclusion_decisions.get(i) != "EXCLUDE":
                    continue
                if cluster_match_cache.get(i) in ("NONE", "INCIDENTAL"):
                    logger.info(
                        "[Pipeline] Second-triage override withheld: variant %d (%s) "
                        "has CLUSTER_PHENOTYPE: %s — compound-het/phase status "
                        "does not override a confirmed phenotype mismatch; EXCLUDE stands.",
                        i + 1, gene, cluster_match_cache.get(i),
                    )
                    continue
                siblings = [j for j in idxs if j != i]
                phases = {
                    segregation.classify_phase(segregation_cache.get(i, ""), segregation_cache.get(j, ""))
                    for j in siblings
                }
                if phases == {"cis"}:
                    continue  # backend-confirmed cis against every sibling — EXCLUDE stands
                logger.info(
                    "[Pipeline] Second-triage override: variant %d (%s) is a compound-het "
                    "candidate and backend phase vs sibling(s) is not confirmed cis — "
                    "forced INCLUDE", i + 1, gene,
                )
                second_triage_justifications[i] = (
                    f"[compound-het exempt — backend phase not confirmed cis — original: "
                    f"EXCLUDE: {second_triage_justifications.get(i, '')}]"
                )
                inclusion_decisions[i] = "INCLUDE"

        # ── Inclusion split (direct from second_triage decisions) ────────────
        include_indices = [i for i in kept_indices if inclusion_decisions.get(i) == "INCLUDE"]
        exclude_indices = [i for i in kept_indices if inclusion_decisions.get(i) == "EXCLUDE"]

        logger.info(
            "[Pipeline] Second triage: %d included, %d excluded.",
            len(include_indices), len(exclude_indices),
        )

        # ── Stage 3.5: Gene-level cross-analysis (included variants only) ──────
        gene_groups_included: dict[str, list[int]] = defaultdict(list)
        for i in include_indices:
            gene = variants[i].get("Gene", "NA")
            if gene and gene != "NA":
                gene_groups_included[gene].append(i)

        cross_analyses: dict[str, str] = {}
        for gene, indices in gene_groups_included.items():
            if len(indices) >= 2:
                logger.info(
                    "[Pipeline] Cross-analysis for gene %s (%d variants)...",
                    gene, len(indices),
                )
                cross_analyses[gene] = cross_analysis.run(
                    gene=gene,
                    variant_contexts=[context_slices[j] for j in indices],
                    reasonings=[reasonings[j] for j in indices],
                    llm=self._llm,
                )

        # ── Stage 4: Per-variant conclusion (ALL kept variants — ACMG score is
        #     forced for every evaluated variant, not just INCLUDEd ones, so
        #     excluded variants still carry a documented ACMG classification in
        #     the report appendix for audit purposes) ──────────────────────────
        conclusion_targets = [i for i in kept_indices if i not in reasoning_failed]
        conclusions: dict[int, str] = {}
        base_acmg_points: dict[int, int | None] = {}
        conclusion_failed: set[int] = set()
        def _conclude_one(i: int) -> tuple[int, str]:
            gene = variants[i].get("Gene", "NA")
            ca   = cross_analyses.get(gene) if gene != "NA" else None
            # Pertinence and cluster-match appended to the reasoning text (not
            # a new run_one() param) — see _pertinence_block's docstring for why.
            reasoning_with_pertinence = (
                reasonings[i] + _pertinence_block(i) + _zygosity_note_block(i)
                + _cluster_match_block(i)
            )
            conc = conclusion.run_one(
                variant_context=context_slices[i],
                reasoning=reasoning_with_pertinence,
                cross_analysis=ca,
                llm=self._llm,
            )
            # ClinVar "P/LP with case reference" → PM3/PS4_Supporting, the code
            # chosen from this gene's inheritance mode (acmg_case_ref.py). Done
            # here, not inside conclusion.run_one, because gene_mode_cache lives
            # here and run_one's signature is wrapped by server_qwen.py.
            return i, apply_case_reference(conc, context_slices[i], gene_mode_cache.get(gene, ""))

        workers = min(MAX_WORKERS_LLM, len(conclusion_targets))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_conclude_one, i): i for i in conclusion_targets}
            for future in as_completed(futures):
                i = futures[future]
                gene = variants[i].get("Gene", "?")
                try:
                    i, conc = future.result()
                except Exception as exc:
                    # Same fail-loud-but-keep-going shape as reasoning_failed
                    # above: a malformed conclusion (e.g. SLMError from
                    # conclusion.py's ACMG-criteria-section check) must not
                    # silently disappear — record it as a visible failure note
                    # (surfaced in the Unclassified appendix below, since
                    # conclusion_failed indices are excluded from every MOI
                    # layer target list that reads conclusions[i]) instead of
                    # feeding broken/incomplete text into moi_denovo etc.,
                    # which can only copy criteria verbatim, never invent them.
                    logger.error(
                        "[Pipeline] Conclusion failed for variant %d (%s): %s",
                        i + 1, gene, exc,
                    )
                    conclusions[i] = f"[CONCLUSION FAILED: {exc}]"
                    base_acmg_points[i] = None
                    conclusion_failed.add(i)
                    continue
                conclusions[i] = conc
                base_acmg_points[i] = _extract_acmg_points_value(conc)
                logger.info(
                    "[Pipeline] Conclusion done for variant %d (%s), base ACMG points=%s.",
                    i + 1, gene, base_acmg_points[i],
                )

        # ── Stage 4.5: MOI-specific layers (3: de novo, 4: dominant-inherited,
        #     5: recessive/compound-het, 6: X-linked) — each ADDS criteria on
        #     top of the Layer 2 base score above; scoped to include_indices
        #     (Tier 1), the same population the flat report used to cover. A
        #     variant can appear in more than one layer if its gene's MOI
        #     qualifies for more than one (e.g. AD_AR genes appear in both the
        #     dominant and recessive layers) — deliberately not deduplicated. ──
        include_set = set(include_indices)

        # segregation_cache was already computed above (over kept_indices, a
        # superset of include_indices) so the CIS/TRANS phase fact could be
        # injected into the reasoning/second-triage sibling blocks before this
        # point — reused here by Layers 3/4/5 unchanged. Layer 6 uses a
        # separate X-linked-specific classifier since XLR/XLD read differently
        # off the same AB values.
        gene_chrom_cache = moi.build_gene_chrom_cache(variants, kept_indices, _group_by_gene)

        # ── Layer 3: de novo (AD / AD_AR / XLD genes, AND any X-linked mode —
        #     de novo occurrence is a valid PS2/PM6 signal for XLR/XL genes
        #     too, e.g. a de novo X-linked variant confirmed absent in both
        #     parents; the moi_xlinked layer only ever adds PP1, never PS2,
        #     so without this a confirmed-de-novo X-linked variant would get
        #     zero credit for that despite trio data confirming it.
        #     Also requires SOME parental AB data (trio or one parent): the
        #     prompt's own rule 3 says PS2/PM6 can NEVER apply when parental
        #     data is "none" — running the layer anyway on a singleton just
        #     produces a boilerplate "de novo status unassessed" section for
        #     every AD-relevant variant in every singleton case, which is
        #     exactly the redundant-MOI-section clutter this gate exists to
        #     avoid.) ──────────────────────────────────────────────────────
        denovo_genes = {
            g for g, m in gene_mode_cache.items()
            if m in ("AD", "AD_AR", "XLD", "XLR", "XLD_XLR", "XL")
        }

        def _parental_ab_presence(i: int) -> tuple[bool, bool]:
            ab_entry = parental_ab[i] if parental_ab else {}
            has_trio = segregation.has_parent_data(ab_entry)
            has_one_parent = (not has_trio) and segregation.has_any_parent_data(ab_entry)
            return has_trio, has_one_parent

        denovo_targets = [
            i for i in include_indices
            if variants[i].get("Gene", "NA") in denovo_genes
            and any(_parental_ab_presence(i))
            and i not in conclusion_failed
        ]
        denovo_outputs: dict[int, str] = {}

        def _denovo_one(i: int) -> tuple[int, str]:
            has_trio, has_one_parent = _parental_ab_presence(i)
            # Append the mosaicism caveat (if any) to the TEXT handed to the
            # prompt only — segregation_cache[i] itself must stay the bare
            # canonical string ("de_novo"/"maternal"/...) since it's also
            # used elsewhere for exact-match control flow (classify_phase(),
            # the maternal/paternal membership checks below).
            ab_entry = parental_ab[i] if parental_ab else {}
            seg_text = segregation_cache[i] + segregation.mosaicism_note(ab_entry)
            return i, moi_denovo.run_one(
                variant_context=context_slices[i],
                base_conclusion=conclusions[i],
                segregation=seg_text,
                has_trio=has_trio,
                has_one_parent=has_one_parent,
                llm=self._llm,
            )

        if denovo_targets:
            workers = min(MAX_WORKERS_LLM, len(denovo_targets))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(_denovo_one, i): i for i in denovo_targets}
                for future in as_completed(futures):
                    i, out = future.result()
                    denovo_outputs[i] = out
                    logger.info(
                        "[Pipeline] De novo layer done for variant %d (%s).",
                        i + 1, variants[i].get("Gene", "?"),
                    )

        # ── Layer 4: dominant-inherited (AD / AD_AR / XLD only — narrower than
        #     Layer 3's de novo gene set. "Cosegregation with an affected
        #     parent" framing doesn't fit XLR/XL genes, which Layer 3 now
        #     covers for de novo purposes but Layer 4 deliberately excludes.
        #     Also requires segregation to actually be "maternal" or
        #     "paternal": the prompt's own rule 1 says PP1 can NEVER apply
        #     otherwise (de_novo, insufficient_data, uncertain, both_carriers,
        #     homozygous_parent all fall through to the same "does not fit,
        #     no PP1" boilerplate) — gating here on the same condition the
        #     prompt already gates on avoids paying for an LLM call whose
        #     output is 100% predictable, and avoids a redundant
        #     DOMINANT-INHERITED section on every AD-relevant variant in
        #     every singleton (no-parent-data) case, which was the actual
        #     bulk of the "still see dominant" reports — not just the
        #     confirmed-de-novo case this gate originally only covered.) ──
        dominant_genes = {g for g, m in gene_mode_cache.items() if m in ("AD", "AD_AR", "XLD")}
        dominant_targets = [
            i for i in include_indices
            if variants[i].get("Gene", "NA") in dominant_genes
            and segregation_cache[i] in ("maternal", "paternal")
            and i not in conclusion_failed
        ]
        dominant_outputs: dict[int, str] = {}

        def _dominant_one(i: int) -> tuple[int, str]:
            ab_entry = parental_ab[i] if parental_ab else {}
            seg = segregation_cache[i]
            proband_ab_class = segregation.classify_ab_ratio(ab_entry.get("proband"))
            if seg == "maternal":
                transmitting_parent_ab_class = segregation.classify_ab_ratio(ab_entry.get("mother"))
            elif seg == "paternal":
                transmitting_parent_ab_class = segregation.classify_ab_ratio(ab_entry.get("father"))
            else:
                transmitting_parent_ab_class = "uncertain"
            return i, moi_dominant.run_one(
                variant_context=context_slices[i],
                base_conclusion=conclusions[i],
                segregation=seg,
                proband_ab_class=proband_ab_class,
                transmitting_parent_ab_class=transmitting_parent_ab_class,
                llm=self._llm,
            )

        if dominant_targets:
            workers = min(MAX_WORKERS_LLM, len(dominant_targets))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(_dominant_one, i): i for i in dominant_targets}
                for future in as_completed(futures):
                    i, out = future.result()
                    dominant_outputs[i] = out
                    logger.info(
                        "[Pipeline] Dominant layer done for variant %d (%s).",
                        i + 1, variants[i].get("Gene", "?"),
                    )

        # ── Layer 5: recessive / compound-het (AR / XLR / AD_AR / XLD_XLR
        #     genes, >=2 included variants, TRANS-gated in Python BEFORE any
        #     LLM call — a CIS pair never reaches moi_recessive.run_pair) ────
        recessive_gene_groups_included: dict[str, list[int]] = {}
        for gene, idxs in recessive_gene_groups.items():
            kept_and_included = [i for i in idxs if i in include_set and i not in conclusion_failed]
            if len(kept_and_included) >= 2:
                recessive_gene_groups_included[gene] = kept_and_included

        recessive_outputs: dict[str, list[str]] = defaultdict(list)
        recessive_pair_targets: list[tuple[str, int, int, str]] = []  # (gene, i, j, phase)
        for gene, idxs in recessive_gene_groups_included.items():
            for i, j in itertools.combinations(idxs, 2):
                phase = segregation.classify_phase(segregation_cache[i], segregation_cache[j])
                if phase == "cis":
                    logger.info(
                        "[Pipeline] Recessive layer: gene %s variants %d/%d excluded — "
                        "CIS phase (same parent), not compound heterozygous.",
                        gene, i + 1, j + 1,
                    )
                    continue
                recessive_pair_targets.append((gene, i, j, phase))

        def _recessive_one(item: tuple[str, int, int, str]) -> tuple[str, str]:
            gene, i, j, phase = item
            gene_ca = cross_analyses.get(gene)
            var_a, var_b = variants[i], variants[j]
            label_a = f"Variant {i + 1} — {gene} ({var_a.get('HGVS', var_a.get('Variant', '?'))})"
            label_b = f"Variant {j + 1} — {gene} ({var_b.get('HGVS', var_b.get('Variant', '?'))})"
            out = moi_recessive.run_pair(
                gene=gene,
                variant_a_label=label_a,
                variant_a_context=context_slices[i],
                variant_a_base_conclusion=conclusions[i],
                variant_b_label=label_b,
                variant_b_context=context_slices[j],
                variant_b_base_conclusion=conclusions[j],
                phase=phase,
                cross_analysis=gene_ca,
                llm=self._llm,
            )
            return gene, out

        if recessive_pair_targets:
            workers = min(MAX_WORKERS_LLM, len(recessive_pair_targets))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(_recessive_one, item): item for item in recessive_pair_targets}
                for future in as_completed(futures):
                    gene, out = future.result()
                    recessive_outputs[gene].append(out)
                    logger.info("[Pipeline] Recessive layer done for gene %s.", gene)

        # ── Layer 5 (solo path): confirmed-homozygous variants in a
        #     recessive-relevant gene with no compound-het partner (gene never
        #     reached the >=2-variant group above). Homozygosity alone
        #     satisfies the biallelic requirement, so these belong in the
        #     Recessive layer, not the Unclassified appendix. ─────────────────
        homozygous_solo_indices: set[int] = set()
        homozygous_solo_targets = [
            i for i in include_indices
            if gene_mode_cache.get(variants[i].get("Gene", "NA")) in ("AR", "XLR", "AD_AR", "XLD_XLR")
            and moi.zygosity_is_confirmed_hom(variants[i].get("Zygosity", ""))
            and variants[i].get("Gene", "NA") not in recessive_gene_groups_included
            and i not in conclusion_failed
        ]

        def _recessive_solo_one(i: int) -> tuple[int, str, str]:
            gene = variants[i].get("Gene", "NA")
            gene_ca = cross_analyses.get(gene)
            var = variants[i]
            label = f"Variant {i + 1} — {gene} ({var.get('HGVS', var.get('Variant', '?'))})"
            ab_entry = parental_ab[i] if parental_ab else {}
            seg = segregation.classify_segregation(
                ab_entry.get("proband"), ab_entry.get("mother"), ab_entry.get("father")
            )
            out = moi_recessive.run_solo(
                gene=gene,
                variant_label=label,
                variant_context=context_slices[i],
                variant_base_conclusion=conclusions[i],
                segregation=seg,
                cross_analysis=gene_ca,
                llm=self._llm,
            )
            return i, gene, out

        if homozygous_solo_targets:
            workers = min(MAX_WORKERS_LLM, len(homozygous_solo_targets))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(_recessive_solo_one, i): i for i in homozygous_solo_targets}
                for future in as_completed(futures):
                    i, gene, out = future.result()
                    recessive_outputs[gene].append(out)
                    homozygous_solo_indices.add(i)
                    logger.info(
                        "[Pipeline] Recessive layer (homozygous-solo) done for variant %d (%s).",
                        i + 1, gene,
                    )

        # ── Layer 6: X-linked (chrX-only genes — a different filter shape
        #     from Layers 3-5, chromosome-based rather than mode-based) ──────
        xlinked_targets = [
            i for i in include_indices
            if gene_chrom_cache.get(variants[i].get("Gene", "NA")) == "X"
            and i not in conclusion_failed
        ]
        xlinked_outputs: dict[int, str] = {}

        def _xlinked_one(i: int) -> tuple[int, str]:
            ab_entry = parental_ab[i] if parental_ab else {}
            pattern = segregation.classify_xlinked_ab(ab_entry.get("proband"), ab_entry.get("mother"))
            seg = segregation.classify_segregation(
                ab_entry.get("proband"), ab_entry.get("mother"), ab_entry.get("father")
            )
            return i, moi_xlinked.run_one(
                variant_context=context_slices[i],
                base_conclusion=conclusions[i],
                xlinked_pattern=pattern,
                segregation=seg,
                llm=self._llm,
            )

        if xlinked_targets:
            workers = min(MAX_WORKERS_LLM, len(xlinked_targets))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(_xlinked_one, i): i for i in xlinked_targets}
                for future in as_completed(futures):
                    i, out = future.result()
                    xlinked_outputs[i] = out
                    logger.info(
                        "[Pipeline] X-linked layer done for variant %d (%s).",
                        i + 1, variants[i].get("Gene", "?"),
                    )

        # ── Stage 5: Cross-MOI clinical conclusion (Layer 8) ──────────────────
        logger.info("[Pipeline] Generating cross-MOI clinical conclusion...")
        layer_outputs_for_summary = {
            "De Novo":            list(denovo_outputs.values()),
            "Dominant-Inherited": list(dominant_outputs.values()),
            "Recessive":          [block for blocks in recessive_outputs.values() for block in blocks],
            "X-Linked":           list(xlinked_outputs.values()),
        }
        layer_outputs_for_summary = {k: v for k, v in layer_outputs_for_summary.items() if v}

        moi_covered_indices = (
            set(denovo_outputs) | set(dominant_outputs)
            | {i for idxs in recessive_gene_groups_included.values() for i in idxs}
            | homozygous_solo_indices
            | set(xlinked_outputs)
        )
        unclassified_indices = [i for i in include_indices if i not in moi_covered_indices]
        unclassified_conclusions = [
            append_clinvar_reference(conclusions[i], context_slices[i])
            for i in unclassified_indices
        ]

        # ── ACMG SF actionable variants (built here, before final_conclusion, so
        #     the Clinical Conclusion prose can name them in its own dedicated
        #     "Actionable findings" section, not just the separate ACTIONABLE
        #     VARIANTS appendix below) — independent of include/exclude since
        #     flagged variants were already force-included above ─────────────────
        actionable_flagged = [
            {
                "index":          i,
                "gene":           variants[i].get("Gene", "?"),
                "hgvs":           variants[i].get("HGVS", variants[i].get("Variant", "?")),
                "condition":      acmg_sf.ACMG_SF_CONDITIONS.get(variants[i].get("Gene", ""), "n/a"),
                "reason":         actionable_reasons.get(i, ""),
                "zygosity":       variants[i].get("Zygosity", "NA"),
                # Phenotype-independent by design (see acmg_sf.build_actionable_set's
                # docstring) — must NOT be re-derived from this variant's own base
                # conclusion, which scores PP4/PVS1 against THIS patient's phenotype
                # and is the wrong source for a secondary finding's classification
                # precisely when phenotype and gene are unrelated (the common case
                # for an incidental finding).
                "classification": actionable_classifications.get(i, "Pathogenic/Likely pathogenic"),
            }
            for i in sorted(actionable_indices)
            if i in conclusions
        ]

        final_summary = final_conclusion.run(
            layer_outputs=layer_outputs_for_summary,
            unclassified_conclusions=unclassified_conclusions,
            patient_phenotype=patient_phenotype,
            actionable_variants=actionable_flagged,
            llm=self._llm,
        )
        # CLINVAR status next to every finding in sections 2-4 (causative,
        # actionable, notable VUS), from each variant's own ClinVar evidence.
        final_summary = add_clinvar_status_to_findings(final_summary, [
            (variants[i].get("Gene", "NA"),
             variants[i].get("HGVS", variants[i].get("Variant", "")),
             clinvar_status_from_context(context_slices[i]))
            for i in conclusions
        ])

        actionable_section = actionable.run(
            flagged=actionable_flagged,
            patient_phenotype=patient_phenotype,
            llm=self._llm,
        )

        # ── Assemble final output ─────────────────────────────────────────────

        # PROCESS DETAILS — all variants + triage summary + second triage summary
        triage_summary = _build_triage_summary(
            n=n,
            kept_indices=kept_indices,
            discarded_indices=discarded_indices,
            include_indices=include_indices,
            exclude_indices=exclude_indices,
            triage_results=triage_results,
            variants=variants,
            triage_ran=triage_ran,
        )
        second_triage_summary = _build_second_triage_summary(
            kept_indices=kept_indices,
            inclusion_decisions=inclusion_decisions,
            second_triage_justifications=second_triage_justifications,
            variants=variants,
        )
        process_parts = [process_details]
        if triage_summary:
            process_parts.append(triage_summary)
        process_parts.append(second_triage_summary)
        process_section = "\n\n".join(process_parts)

        # SEGREGATION ANALYSIS — parental AB data summary (if available)
        segregation_section = _build_segregation_analysis(
            variants=variants,
            parental_ab=parental_ab,
        )

        # AUGMENTED CONTEXT — all variants (unchanged)
        full_context_display = _reconstruct_full_context(context_slices)

        # REASONING — all kept variants, in original variant order
        reasoning_parts = []
        for i in kept_indices:
            variant  = variants[i]
            gene     = variant.get("Gene", "?")
            hgvs     = variant.get("HGVS", variant.get("Variant", "?"))
            decision = inclusion_decisions.get(i, "INCLUDE")
            tier_tag = f"[{decision}]"
            reasoning_parts.append(
                f"{SEP_VARIANT}\nVARIANT {i + 1} — {gene}  {hgvs}  {tier_tag}\n"
                f"{SEP_VARIANT}\n{reasonings[i]}"
            )
        reasoning_display = "\n\n".join(reasoning_parts)

        # FINAL REPORT — one section per MOI layer (each independently ranked
        # Pathogenic → Likely Pathogenic → VUS → Likely Benign → Benign via the
        # same _classification_rank regex, which matches "ACMG points" anywhere
        # in the text so it works unchanged on each layer's own "Total ACMG
        # points: N → Classification" line), a variant appearing in multiple
        # qualifying layers shown in full in each (not deduplicated), an
        # Unclassified appendix for variants whose gene MOI never resolved to
        # any layer, then the cross-MOI Clinical Conclusion + the pre-existing
        # exclude/discard appendix (unrelated, untouched).
        moi_sections = "".join([
            _build_moi_section("DE NOVO ANALYSIS", list(denovo_outputs.values())),
            _build_moi_section("DOMINANT-INHERITED ANALYSIS", list(dominant_outputs.values())),
            _build_moi_section(
                "RECESSIVE ANALYSIS (COMPOUND-HETEROZYGOUS / CONFIRMED HOMOZYGOUS)",
                [block for blocks in recessive_outputs.values() for block in blocks],
            ),
            _build_moi_section("X-LINKED ANALYSIS", list(xlinked_outputs.values())),
            _build_moi_section("UNCLASSIFIED — NO MOI-SPECIFIC ANALYSIS", unclassified_conclusions),
        ])

        gene_evidence_table = _build_gene_evidence_table(
            variants, genome_build, gene_mode_cache, gene_mode_reasoning_cache,
        )
        gene_evidence_section = (
            f"{SEP_VARIANT}\nGENE-LEVEL EVIDENCE SUMMARY (ClinVar missense:LoF ratio, gnomAD pLI/LOEUF, inheritance mode)\n"
            f"{SEP_VARIANT}\n\n{gene_evidence_table}\n\n"
            if gene_evidence_table else ""
        )

        final_report = (
            gene_evidence_section
            + moi_sections
            + f"\n{SEP_VARIANT}\nCLINICAL CONCLUSION\n{SEP_VARIANT}\n\n{final_summary}\n"
        )

        # PS3 is literature-only; downstream SLM rewrites (MOI layer blocks,
        # Clinical Conclusion) routinely drop the grounding PMID — re-attach
        # it from the validated Stage-4 PS3 line so the reader always sees it.
        final_report = surface_ps3_citations(final_report, list(conclusions.values()))

        if actionable_section:
            final_report += (
                f"\n{SEP_VARIANT}\nACTIONABLE VARIANTS (ACMG SF)\n{SEP_VARIANT}\n\n{actionable_section}\n"
            )

        # Not gated on triage_ran: the proband-AB-artifact gate can populate
        # discarded_indices even when n <= TRIAGE_ENABLED_THRESHOLD (SLM
        # triage skipped) — those discards must still be visible in the
        # appendix, not silently dropped from the report.
        if exclude_indices or discarded_indices:
            final_report += _build_report_appendix(
                exclude_indices=exclude_indices,
                discarded_indices=discarded_indices,
                triage_results=triage_results,
                variants=variants,
                conclusions=conclusions,
            )

        output = (
            (f"{SEP}\nCOLUMN HEADER INTERPRETATION\n{SEP}\n{header_mapping_summary}\n\n"
             if header_mapping_summary else "")
            + f"{SEP}\nPROCESS DETAILS\n{SEP}\n"
            + process_section + "\n\n"
            + (f"{SEP}\nSEGREGATION ANALYSIS\n{SEP}\n{segregation_section}\n\n" if segregation_section else "")
            + f"{SEP}\nAUGMENTED CONTEXT\n{SEP}\n"
            + full_context_display + "\n\n"
            + f"{SEP}\nREASONING\n{SEP}\n"
            + reasoning_display + "\n\n"
            + f"{SEP}\nFINAL REPORT\n{SEP}\n"
            + final_report + "\n"
        )
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT ASSEMBLY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_triage_summary(
    n: int,
    kept_indices: list[int],
    discarded_indices: list[int],
    include_indices: list[int],
    exclude_indices: list[int],
    triage_results: dict[int, tuple[str, str]],
    variants: list[dict],
    triage_ran: bool,
) -> str:
    if not triage_ran:
        return ""

    SEP = "─" * 56
    lines = [
        f"\n{SEP}",
        "TRIAGE SUMMARY",
        SEP,
        f"Total variants  : {n}",
        f"Kept (Tier 1+2) : {len(kept_indices)}",
        f"Discarded       : {len(discarded_indices)}",
    ]

    if discarded_indices:
        lines.append("\nDiscarded variants:")
        for i in discarded_indices:
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            hgvs    = variant.get("HGVS", variant.get("Variant", "?"))
            _, just = triage_results[i]
            lines.append(f"  Variant {i + 1} — {gene} ({hgvs}): {just}")

    # Compound-het exemptions: kept variants whose justification records the override
    exemptions = [
        i for i in kept_indices
        if triage_results[i][1].startswith("[compound-het exempt")
    ]
    if exemptions:
        lines.append("\nCompound-het exemptions (forced KEEP):")
        for i in exemptions:
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            hgvs    = variant.get("HGVS", variant.get("Variant", "?"))
            lines.append(f"  Variant {i + 1} — {gene} ({hgvs})")

    include_labels = ", ".join(f"Variant {i + 1}" for i in include_indices) or "(none)"
    exclude_labels = ", ".join(f"Variant {i + 1}" for i in exclude_indices) or "(none)"
    lines.append(f"\nIncluded (→ conclusion): {include_labels}")
    lines.append(f"Excluded (→ reasoning only): {exclude_labels}")

    return "\n".join(lines)


def _build_second_triage_summary(
    kept_indices: list[int],
    inclusion_decisions: dict[int, str],
    second_triage_justifications: dict[int, str],
    variants: list[dict],
) -> str:
    SEP = "─" * 56
    n_included = sum(1 for i in kept_indices if inclusion_decisions.get(i) == "INCLUDE")
    n_excluded  = sum(1 for i in kept_indices if inclusion_decisions.get(i) == "EXCLUDE")
    lines = [
        f"\n{SEP}",
        "SECOND TRIAGE (INCLUSION DECISIONS)",
        SEP,
        f"Evaluated : {len(kept_indices)}",
        f"Included  : {n_included}",
        f"Excluded  : {n_excluded}",
        "",
    ]
    for i in kept_indices:
        variant  = variants[i]
        gene     = variant.get("Gene", "?")
        hgvs     = variant.get("HGVS", variant.get("Variant", "?"))
        decision = inclusion_decisions.get(i, "INCLUDE")
        just     = second_triage_justifications.get(i, "")
        lines.append(f"  Variant {i + 1} — {gene}  {hgvs}  [{decision}]")
        if just:
            lines.append(f"    {just}")
    return "\n".join(lines)


_ACMG_BLOCK_RE = re.compile(r"\*\*ACMG criteria:\*\*.*?(?=\*\*Comment:\*\*|\Z)", re.DOTALL)


def _extract_acmg_block(conclusion_text: str) -> str:
    """Pull just the ACMG criteria list + points line out of a full conclusion block."""
    m = _ACMG_BLOCK_RE.search(conclusion_text)
    return m.group(0).strip() if m else ""


def _build_moi_section(title: str, blocks: list[str]) -> str:
    """One MOI-layer report section (Phase 7 of the MOI-layer restructuring).
    Blocks are ranked P → LP → VUS → LB → B by the same _classification_rank
    used for the base layer — it searches for "ACMG points" anywhere in the
    text, so it matches each layer's own "Total ACMG points: N →
    Classification" line unchanged, no separate rank function needed.
    Returns "" (omitted entirely) when there are no blocks for this layer —
    a case with no de novo findings simply has no DE NOVO ANALYSIS section,
    rather than an empty one."""
    if not blocks:
        return ""
    SEP = "─" * 56
    ranked = sorted(blocks, key=_classification_rank)
    body = "\n\n---\n\n".join(ranked)
    return f"\n{SEP}\n{title}\n{SEP}\n{body}\n"


def _build_report_appendix(
    exclude_indices: list[int],
    discarded_indices: list[int],
    triage_results: dict[int, tuple[str, str]],
    variants: list[dict],
    conclusions: dict[int, str],
) -> str:
    lines = ["\n\n---\n\n## Variants not included in conclusion"]

    if exclude_indices:
        lines.append("\n### Excluded — Reasoning available (see REASONING section)")
        for i in exclude_indices:
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            hgvs    = variant.get("HGVS", variant.get("Variant", "?"))
            lines.append(f"- Variant {i + 1} — {gene} ({hgvs})")
            acmg_block = _extract_acmg_block(conclusions.get(i, ""))
            if acmg_block:
                lines.append(f"  {acmg_block}")

    if discarded_indices:
        lines.append("\n### Discarded at first triage")
        for i in discarded_indices:
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            hgvs    = variant.get("HGVS", variant.get("Variant", "?"))
            _, just = triage_results[i]
            lines.append(f"- Variant {i + 1} — {gene} ({hgvs}): {just}")

    return "\n".join(lines)


def _build_segregation_analysis(
    variants: list[dict],
    parental_ab: list[dict] | None,
) -> str:
    """
    Build the SEGREGATION ANALYSIS section showing parental allelic balance data.
    Returns empty string if no parental data is available.
    """
    if not parental_ab:
        return ""

    SEP = "─" * 56
    lines = [
        SEP,
        "PARENTAL ALLELIC BALANCE DATA",
        SEP,
    ]

    # Detect column labels from first entry
    first_entry = parental_ab[0] if parental_ab else {}
    labels = list(first_entry.keys())

    if labels:
        header = "Variant".ljust(8) + "Gene".ljust(12)
        for label in labels:
            header += label.capitalize().rjust(12)
        lines.append(header)
        lines.append("─" * max(len(header), 40))

    for i, (variant, ab_entry) in enumerate(zip(variants, parental_ab)):
        gene = variant.get("Gene", "?")[:10]
        line = f"{i + 1:<8}{gene:<12}"
        for label in labels:
            val = ab_entry.get(label, "NA")
            line += val.rjust(12)
        lines.append(line)

    # Add interpretation summary
    n_with_parental = sum(
        1 for ab in parental_ab
        if any(k in ab for k in ("mother", "father", "extra1"))
    )
    if n_with_parental > 0:
        lines.append("")
        lines.append(f"Variants with parental data: {n_with_parental}/{len(parental_ab)}")
        lines.append("De novo candidates (AB < 0.1 in both parents):")
        for i, ab_entry in enumerate(parental_ab):
            try:
                mother_ab = float(ab_entry.get("mother", "1") or "1")
            except ValueError:
                mother_ab = 1.0
            try:
                father_ab = float(ab_entry.get("father", "1") or "1")
            except ValueError:
                father_ab = 1.0
            if mother_ab < 0.1 and father_ab < 0.1:
                variant = variants[i]
                gene = variant.get("Gene", "?")
                hgvs = variant.get("HGVS", variant.get("Variant", "?"))
                lines.append(f"  Variant {i + 1} — {gene} ({hgvs})")

    return "\n".join(lines)

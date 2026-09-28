"""
pipeline/tools/gnomad_constraint.py — gnomAD gene constraint (pLI, LOEUF) fetch.

Fetches gene-level LoF-intolerance metrics from the public gnomAD GraphQL API.
Gene-scoped, not variant-scoped: reused across every variant that shares a gene
via a class-level cache (same pattern as LitVar2SummaryTool._cgd_table).
"""

import logging
import threading

from pipeline.tools.base import NetworkTool
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError

logger = logging.getLogger(__name__)

GNOMAD_API_URL = "https://gnomad.broadinstitute.org/api"

_QUERY = """
query GnomadConstraint($geneSymbol: String!, $referenceGenome: ReferenceGenomeId!) {
  gene(gene_symbol: $geneSymbol, reference_genome: $referenceGenome) {
    gnomad_constraint {
      pLI
      oe_lof_upper
    }
  }
}
"""

_BUILD_TO_REFERENCE_GENOME = {"hg19": "GRCh37", "hg38": "GRCh38"}

PLI_INTOLERANT_THRESHOLD = 0.9
LOEUF_CONSTRAINED_THRESHOLD = 0.35


def classify_pli(pli: float | None) -> str:
    if pli is None:
        return "not available"
    return "LoF-intolerant" if pli >= PLI_INTOLERANT_THRESHOLD else "LoF-tolerant"


def classify_loeuf(loeuf: float | None) -> str:
    if loeuf is None:
        return "not available"
    return "constrained" if loeuf < LOEUF_CONSTRAINED_THRESHOLD else "unconstrained"


class GnomadConstraintTool(NetworkTool):
    """
    Fetches pLI and LOEUF (oe_lof_upper) for a gene from gnomAD.

    gate():  runs when Gene is present.
    run():   one GraphQL call per gene, cached at class level (gene-invariant,
             shared across all variants in that gene for the whole run).
    """

    name        = "gnomad_constraint"
    description = (
        "Fetches gnomAD gene constraint metrics (pLI, LOEUF) — used to judge "
        "whether loss-of-function is a plausible disease mechanism for this gene."
    )

    timeout: int = 15

    # Class-level cache: {"GENE:GRCh37": {"pli": float|None, "loeuf": float|None} | None}
    _constraint_cache: dict[str, dict | None] = {}
    _cache_lock = threading.Lock()

    def gate(self, variant: dict, context: ToolContext) -> bool:
        gene = context.field("Gene")
        return gene != "NA" and gene.strip() != ""

    def _fetch_constraint(self, gene: str, reference_genome: str) -> dict | None:
        try:
            resp = self._post(
                GNOMAD_API_URL,
                json={
                    "query": _QUERY,
                    "variables": {"geneSymbol": gene, "referenceGenome": reference_genome},
                },
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            raise ToolFetchError(f"gnomAD constraint request failed for {gene}: {e}") from e

        try:
            data = resp.json()
            gene_data = data.get("data", {}).get("gene")
            if not gene_data:
                return None
            constraint = gene_data.get("gnomad_constraint")
            if not constraint:
                return None
            return {
                "pli":   constraint.get("pLI"),
                "loeuf": constraint.get("oe_lof_upper"),
            }
        except Exception as e:
            raise ToolParseError(f"Failed to parse gnomAD response for {gene}: {e}") from e

    def _get_constraint(self, gene: str, reference_genome: str) -> dict | None:
        cache_key = f"{gene.upper()}:{reference_genome}"
        if cache_key not in self._constraint_cache:
            with self._cache_lock:
                if cache_key not in self._constraint_cache:   # double-checked locking
                    self._constraint_cache[cache_key] = self._fetch_constraint(gene, reference_genome)
        return self._constraint_cache[cache_key]

    def run(self, variant: dict, context: ToolContext) -> str | None:
        gene = context.field("Gene")
        reference_genome = _BUILD_TO_REFERENCE_GENOME.get(context.genome_build, "GRCh37")

        constraint = self._get_constraint(gene, reference_genome)
        if not constraint or (constraint["pli"] is None and constraint["loeuf"] is None):
            return (
                f"GNOMAD GENE CONSTRAINT ({gene}, {reference_genome}):\n"
                "Not available — gnomAD has no constraint entry for this gene."
            )

        pli   = constraint["pli"]
        loeuf = constraint["loeuf"]
        lines = [f"GNOMAD GENE CONSTRAINT ({gene}, {reference_genome}):"]

        if pli is not None:
            lines.append(f"pLI = {pli:.3f} ({classify_pli(pli)}, threshold pLI≥{PLI_INTOLERANT_THRESHOLD})")
        else:
            lines.append("pLI = not available")

        if loeuf is not None:
            lines.append(f"LOEUF = {loeuf:.3f} ({classify_loeuf(loeuf)}, threshold LOEUF<{LOEUF_CONSTRAINED_THRESHOLD})")
        else:
            lines.append("LOEUF = not available")

        return "\n".join(lines)

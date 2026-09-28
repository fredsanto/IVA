"""
pipeline/tools/gnomad_frequency.py — gnomAD variant-level allele frequency
(including homozygote/heterozygote carrier counts).

Two consumers of the shared `fetch_variant_frequency()`:
  1. GnomadFrequencyTool — retrieval-stage fallback, gated to run only when
     the input CSV's own Frequency column is blank. Exists because PM2 (and
     BS1) were being applied from "Frequency is blank, therefore presumed
     rare" reasoning, when a live gnomAD lookup is one API call away and can
     directly contradict that assumption (a real case: CSV Frequency=NA, but
     gnomAD exome AF=0.0013 — not rare enough for a presumed-recessive disease).
  2. pipeline.pipeline._build_gene_evidence_table — always calls it (ungated)
     for the final report's per-gene evidence summary, since homozygote/
     heterozygote carrier counts are never present in the input CSV at all,
     regardless of whether a plain AF was supplied.
"""

import logging

import requests

from pipeline.tools.base import NetworkTool
from pipeline.tools.autopvs1 import parse_variant_coords
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError

logger = logging.getLogger(__name__)

GNOMAD_API_URL = "https://gnomad.broadinstitute.org/api"

_QUERY = """
query GnomadVariantFrequency($variantId: String!, $datasetId: DatasetId!) {
  variant(variantId: $variantId, dataset: $datasetId) {
    genome { af ac an homozygote_count }
    exome { af ac an homozygote_count }
  }
}
"""

# GRCh38/gnomAD v4 is the current combined exomes+genomes release; GRCh37
# queries use v2.1.1, the last release with GRCh37 coordinates.
_BUILD_TO_DATASET = {"hg19": "gnomad_r2_1", "hg38": "gnomad_r4"}

_NA_VALUES = {"", "na", "n/a", "none", "."}

_DEFAULT_TIMEOUT = 15


def _het_count(ac: int | None, hom: int | None) -> int | None:
    """Heterozygote carriers = AC - 2*homozygotes (standard gnomAD-browser formula)."""
    if ac is None or hom is None:
        return None
    return ac - 2 * hom


def fetch_variant_frequency(
    chrom: str, pos: str, ref: str, alt: str, genome_build: str,
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict:
    """
    Single gnomAD GraphQL call for one variant's population frequency and
    homozygote/heterozygote carrier counts.

    Returns a dict:
        {"variant_id": str, "dataset_id": str,
         "genome": {"af", "ac", "an", "homozygote_count", "heterozygote_count"} | None,
         "exome":  {...} | None,
         "error":  str | None}   # set when gnomAD returned no match / an error
    Raises ToolFetchError/ToolParseError on network or parse failure (not on
    a legitimate "no match" result, which is returned via "error" instead).
    """
    chrom_key  = chrom.strip().upper().removeprefix("CHR")
    dataset_id = _BUILD_TO_DATASET.get(genome_build, "gnomad_r4")
    variant_id = f"{chrom_key}-{pos}-{ref.upper()}-{alt.upper()}"

    try:
        resp = requests.post(
            GNOMAD_API_URL,
            json={"query": _QUERY, "variables": {"variantId": variant_id, "datasetId": dataset_id}},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    except Exception as e:
        raise ToolFetchError(f"gnomAD variant frequency request failed for {variant_id}: {e}") from e

    try:
        data = resp.json()
    except Exception as e:
        raise ToolParseError(f"Failed to parse gnomAD response for {variant_id}: {e}") from e

    result = {"variant_id": variant_id, "dataset_id": dataset_id, "genome": None, "exome": None, "error": None}

    if data.get("errors"):
        result["error"] = data["errors"][0].get("message", "no match")
        return result

    v = (data.get("data") or {}).get("variant")
    if not v:
        result["error"] = "no genome/exome coverage at this position"
        return result

    for key in ("genome", "exome"):
        sub = v.get(key)
        if sub and sub.get("af") is not None:
            result[key] = {
                "af": sub["af"], "ac": sub.get("ac"), "an": sub.get("an"),
                "homozygote_count": sub.get("homozygote_count"),
                "heterozygote_count": _het_count(sub.get("ac"), sub.get("homozygote_count")),
            }

    if result["genome"] is None and result["exome"] is None:
        result["error"] = "no genome/exome coverage at this position"

    return result


def _format_frequency_block(header: str, result: dict, extra_note: str) -> str:
    if result["error"]:
        return f"{header}\nNot found in gnomAD — {result['error']}. {extra_note}"

    lines = [header]
    for key, label in (("genome", "Genome"), ("exome", "Exome")):
        sub = result.get(key)
        if sub:
            hom = sub["homozygote_count"]
            het = sub["heterozygote_count"]
            hom_str = str(hom) if hom is not None else "n/a"
            het_str = str(het) if het is not None else "n/a"
            lines.append(
                f"{label} AF = {sub['af']:.6g} (AC={sub['ac']}, AN={sub['an']}, "
                f"homozygotes={hom_str}, heterozygotes={het_str})"
            )
    lines.append(extra_note)
    return "\n".join(lines)


class GnomadFrequencyTool(NetworkTool):
    """
    Fetches this variant's own gnomAD population allele frequency and
    homozygote/heterozygote carrier counts — fallback only, used when the
    input CSV's Frequency field is blank.

    gate():  runs when Frequency is NA/blank AND genomic coordinates resolve.
    run():   single GraphQL call via fetch_variant_frequency(), no SLM, not
             cached (variant-specific, unlike the gene-scoped GnomadConstraintTool).
    """

    name        = "gnomad_frequency"
    description = (
        "Fetches this variant's own gnomAD population allele frequency and "
        "homozygote/heterozygote carrier counts — fallback for when the "
        "input CSV's Frequency field is blank, so PM2/BS1 are never "
        "grounded on absence of data alone."
    )

    timeout: int = 15

    def gate(self, variant: dict, context: ToolContext) -> bool:
        freq = str(variant.get("Frequency") or "").strip().lower()
        if freq not in _NA_VALUES:
            return False   # CSV already supplied a frequency — nothing to fetch
        try:
            parse_variant_coords(
                variant_str=variant.get("Variant", ""),
                chrom_field=variant.get("Chromosome", ""),
                pos_field=variant.get("Position", ""),
                ref_field=variant.get("Ref_seq", ""),
                alt_field=variant.get("Var_seq", ""),
            )
        except ValueError:
            return False
        return True

    def run(self, variant: dict, context: ToolContext) -> str | None:
        try:
            coords = parse_variant_coords(
                variant_str=variant.get("Variant", ""),
                chrom_field=variant.get("Chromosome", ""),
                pos_field=variant.get("Position", ""),
                ref_field=variant.get("Ref_seq", ""),
                alt_field=variant.get("Var_seq", ""),
            )
        except ValueError:
            return None
        chrom, pos, ref, alt = coords
        result = fetch_variant_frequency(chrom, pos, ref, alt, context.genome_build, self.timeout)
        header = f"GNOMAD VARIANT FREQUENCY ({result['variant_id']}, {result['dataset_id']}):"
        extra_note = (
            "The input CSV's own Frequency field was blank for this variant — this is a live "
            "gnomAD lookup; use it (not absence-of-data) to judge PM2/BS1."
        )
        return _format_frequency_block(header, result, extra_note)

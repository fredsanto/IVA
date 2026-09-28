"""
pipeline/tools/clingen_allele.py — ClinGen Allele Registry variant lookup.

Resolves a variant to ClinGen's canonical allele ID (CAid) via the public
Allele Registry API (reg.clinicalgenome.org) and surfaces its cross-database
records — ClinVar variation/RCV, dbSNP rsID, gnomAD/ExAC variant IDs. No API
key required; any syntactically valid HGVS is auto-registered on first lookup
even if it has never been reported anywhere else, so an empty cross-reference
set is itself a meaningful signal (variant is novel/unreported in every
database ClinGen indexes) rather than a failed lookup.
"""

import re
import logging

import requests

from pipeline.tools.base import NetworkTool
from pipeline.tools.autopvs1 import parse_variant_coords
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError

logger = logging.getLogger(__name__)

CLINGEN_ALLELE_REGISTRY_URL = "https://reg.clinicalgenome.org/allele"

# Extracts the cDNA-change token out of a combined/compound HGVS string, e.g.
# "GENE_X:NM_000000:exon4:c.100A>C:p.K34T" -> "c.100A>C". Same shape of problem
# as ncbi.py's ClinVar resolver — the Allele Registry wants a clean
# "TRANSCRIPT:c.change" pair, not a colon-glued compound annotation. "("
# excluded too — otherwise a "c.200T>A(p.Val67Asp)"-style string swallows
# the trailing protein annotation into the token.
_CDNA_CHANGE_RE = re.compile(r"c\.[^\s:;()]+")
_CLEAN_TRANSCRIPT_HGVS_RE = re.compile(r"^[A-Za-z0-9_]+\.\d+:c\.")

# RefSeq genomic accessions, used only for the genomic-coordinate SNV
# fallback when no usable transcript HGVS can be built.
_HG19_CHROM_TO_NC = {
    "1": "NC_000001.10", "2": "NC_000002.11", "3": "NC_000003.11", "4": "NC_000004.11",
    "5": "NC_000005.9",  "6": "NC_000006.11", "7": "NC_000007.13", "8": "NC_000008.10",
    "9": "NC_000009.11", "10": "NC_000010.10", "11": "NC_000011.9", "12": "NC_000012.11",
    "13": "NC_000013.10", "14": "NC_000014.8", "15": "NC_000015.9", "16": "NC_000016.9",
    "17": "NC_000017.10", "18": "NC_000018.9", "19": "NC_000019.9", "20": "NC_000020.10",
    "21": "NC_000021.8", "22": "NC_000022.10", "X": "NC_000023.10", "Y": "NC_000024.9",
}
_HG38_CHROM_TO_NC = {
    "1": "NC_000001.11", "2": "NC_000002.12", "3": "NC_000003.12", "4": "NC_000004.12",
    "5": "NC_000005.10", "6": "NC_000006.12", "7": "NC_000007.14", "8": "NC_000008.11",
    "9": "NC_000009.12", "10": "NC_000010.11", "11": "NC_000011.10", "12": "NC_000012.12",
    "13": "NC_000013.11", "14": "NC_000014.9", "15": "NC_000015.10", "16": "NC_000016.10",
    "17": "NC_000017.11", "18": "NC_000018.10", "19": "NC_000019.10", "20": "NC_000020.11",
    "21": "NC_000021.9", "22": "NC_000022.11", "X": "NC_000023.11", "Y": "NC_000024.10",
}


def _build_query_candidates(variant: dict, genome_build: str) -> list[str]:
    """
    Build an ordered list of HGVS strings to try against the Allele Registry:
      1. Transcript field + clean cDNA token extracted from HGVS -> "NM_x.x:c.xxx"
         (skipped if Transcript has no version suffix — ClinGen rejects a bare
         "NM_000000" with "Unknown reference", and the CSV's own Transcript
         field is frequently NA/versionless on compound-HGVS input rows)
      2. HGVS field already looks like a clean, versioned transcript:c. string
      3. Genomic-coordinate SNV fallback, declared build first, then the OTHER
         build's accession as a second attempt — the pipeline's declared
         genome_build is frequently wrong for a given upload (e.g. hg19
         declared, hg38-coordinated CSV), and there's no reliable way to know
         which one a CSV actually uses without trying both; a wrong build
         reads back as a ClinGen "IncorrectReferenceAllele" response, so
         trying the alternate build costs one extra request only when the
         first guess is wrong.
    Indel genomic HGVS construction is not attempted (single-base SNVs only).
    """
    candidates: list[str] = []

    transcript = str(variant.get("Transcript") or "").strip()
    hgvs = str(variant.get("HGVS") or "").strip()

    if transcript and transcript != "NA" and "." in transcript and hgvs and hgvs != "NA":
        m = _CDNA_CHANGE_RE.search(hgvs)
        if m:
            candidates.append(f"{transcript}:{m.group(0)}")

    if hgvs and hgvs != "NA" and _CLEAN_TRANSCRIPT_HGVS_RE.match(hgvs):
        candidates.append(hgvs)

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
        chrom, pos, ref, alt = coords
        if len(ref) == 1 and len(alt) == 1 and ref not in ("-", "") and alt not in ("-", ""):
            chrom_key = chrom.strip().upper().removeprefix("CHR")
            build_order = (
                [_HG38_CHROM_TO_NC, _HG19_CHROM_TO_NC] if genome_build == "hg38"
                else [_HG19_CHROM_TO_NC, _HG38_CHROM_TO_NC]
            )
            for nc_map in build_order:
                nc = nc_map.get(chrom_key)
                if nc:
                    candidates.append(f"{nc}:g.{pos}{ref.upper()}>{alt.upper()}")

    return candidates


class ClinGenAlleleTool(NetworkTool):
    """
    Resolves a variant against the ClinGen Allele Registry.

    gate():  runs when a usable query (transcript+cDNA, clean HGVS, or
             genomic SNV coordinates) can be built.
    run():   single GET to reg.clinicalgenome.org/allele?hgvs=..., no SLM.
             A 400/500 response from ClinGen (unparseable HGVS, coordinates
             outside the reference) is a legitimate "not resolvable for this
             variant" outcome, not an infrastructure failure — reported as a
             normal text result, not raised as ToolFetchError.
    """

    name        = "clingen_allele"
    description = (
        "Resolves the variant against the ClinGen Allele Registry — cross-"
        "references ClinVar/dbSNP/gnomAD/ExAC by canonical allele ID; an "
        "empty cross-reference set signals the variant is novel/unreported."
    )

    timeout: int = 15

    def gate(self, variant: dict, context: ToolContext) -> bool:
        return bool(_build_query_candidates(variant, context.genome_build))

    def _fetch(self, query: str) -> tuple[dict | None, str | None]:
        """Returns (parsed_json, None) on success, or (None, error_detail) on
        a ClinGen-side rejection (400/500 with a JSON error body)."""
        try:
            resp = requests.get(
                CLINGEN_ALLELE_REGISTRY_URL,
                params={"hgvs": query},
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as e:
            raise ToolFetchError(f"ClinGen Allele Registry request failed for {query}: {e}") from e

        if not resp.ok:
            try:
                err = resp.json()
                detail = f"{err.get('errorType', 'error')}: {err.get('message', resp.text[:200])}"
            except Exception:
                detail = resp.text[:200] if resp.text else f"HTTP {resp.status_code}"
            return None, detail

        try:
            return resp.json(), None
        except Exception as e:
            raise ToolParseError(f"Could not parse ClinGen response for {query}: {e}") from e

    def run(self, variant: dict, context: ToolContext) -> str | None:
        candidates = _build_query_candidates(variant, context.genome_build)
        if not candidates:
            return None

        query = None
        data = None
        last_detail = None
        for candidate in candidates:
            data, detail = self._fetch(candidate)
            if data is not None:
                query = candidate
                break
            last_detail = detail  # keep trying remaining candidates on failure

        if data is None:
            # All candidates failed (e.g. both build guesses gave a reference
            # mismatch) — report the last attempt's detail, not a silent None.
            return f"CLINGEN ALLELE REGISTRY ({candidates[-1]}):\nNot resolvable — {last_detail}"

        ca_id   = (data.get("@id") or "").rsplit("/", 1)[-1] or "unknown"
        title   = (data.get("communityStandardTitle") or [None])[0]
        records = data.get("externalRecords", {}) or {}

        lines = [f"CLINGEN ALLELE REGISTRY ({query}):", f"CAid: {ca_id}"]
        if title:
            lines.append(f"Community title: {title}")

        clinvar_variations = records.get("ClinVarVariations", [])
        if clinvar_variations:
            v = clinvar_variations[0]
            rcvs = ", ".join(v.get("RCV", [])) or "none listed"
            lines.append(f"ClinVar variation ID: {v.get('variationId')} (RCV: {rcvs})")

        dbsnp = records.get("dbSNP", [])
        if dbsnp:
            lines.append(f"dbSNP: rs{dbsnp[0].get('rs')}")

        gnomad = records.get("gnomAD_4") or records.get("gnomAD_3") or records.get("gnomAD_2")
        if gnomad:
            lines.append(f"gnomAD variant: {gnomad[0].get('variant', gnomad[0].get('id'))}")

        if not clinvar_variations and not dbsnp and not gnomad:
            lines.append(
                "No cross-references in ClinVar, dbSNP, or gnomAD — this allele appears "
                "novel/unreported in every database ClinGen indexes."
            )

        lines.append(f"URL: {data.get('@id', '')}")
        return "\n".join(lines)

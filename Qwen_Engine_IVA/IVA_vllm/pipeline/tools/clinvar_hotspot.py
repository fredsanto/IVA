"""
pipeline/tools/clinvar_hotspot.py — deterministic PM1 mutational-hotspot check
from ClinVar missense classifications in a window around this variant's residue.

PM1 was previously left entirely to the SLM ("located in a mutational hot spot
and/or critical and well-established functional domain") with no tool ever
supplying hotspot or domain evidence — so it was either never applied, or
applied from a gene-level functional description that says nothing about this
residue's neighbourhood. This tool answers the hotspot half of PM1 directly
from ClinVar, using VarSome's hotspot rule: within +/-25 bp of the variant
(+/-8 codons), at least 4 Pathogenic/Likely pathogenic missense variants and
no Benign/Likely benign missense variants.
"""

import logging
import re
import threading

from pipeline.tools.base import NetworkTool
from pipeline.tools.websearch import _ncbi_get, DEFAULT_TIMEOUT
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError
from pipeline.core.protein_change import PROTEIN_CHANGE_RE, _to_aa3
from pipeline.core.acmg_sf import is_pathogenic_clinvar

logger = logging.getLogger(__name__)

# VarSome hotspot thresholds: +/-25 bp around the variant = +/-8 codons.
WINDOW_AA = 8
MIN_PATHOGENIC = 4
MAX_BENIGN = 0

# Bounds one gene's esearch; genes with more classified missense records than
# this are truncated and the hotspot check is reported as not evaluable.
_MAX_RECORDS = 10000
_ESUMMARY_BATCH = 400

_PLP_TERM = '("clinsig pathogenic"[Properties] OR "clinsig likely pathogenic"[Properties])'
_BLB_TERM = '("clinsig benign"[Properties] OR "clinsig likely benign"[Properties])'
_MISSENSE_TERM = '"missense variant"[molecular consequence]'

# ClinVar's own title, e.g. "NM_000000.1(GENE):c.100A>G (p.Lys34Glu)".
_TITLE_RE = re.compile(r":(c\.[^\s(]+)\s*\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)")

_STOP_TOKENS = {"ter", "x", "*"}


def _is_missense_change(ref: str, alt: str) -> bool:
    return alt.lower() not in _STOP_TOKENS and _to_aa3(ref).lower() != _to_aa3(alt).lower()


class ClinVarHotspotTool(NetworkTool):
    """
    For each distinct residue in this (missense) variant's HGVS, counts ClinVar
    P/LP and B/LB missense variants within +/-WINDOW_AA residues and states
    whether the VarSome hotspot rule for PM1 is met.

    gate():  runs when the HGVS yields at least one missense protein change.
    run():   two esearch calls (P/LP, B/LB missense) plus batched esummary
             calls per gene, cached at class level.
    """

    name        = "clinvar_hotspot"
    description = (
        "Deterministic ClinVar P/LP vs B/LB missense count in a +/-8 residue "
        "window around this variant — grounds the hotspot half of PM1."
    )

    timeout: int = DEFAULT_TIMEOUT

    # {"GENE": [record, ...] | None (truncated)}
    _records_cache: dict[str, list[dict] | None] = {}
    _cache_lock = threading.Lock()

    @staticmethod
    def _missense_seeds(hgvs: str) -> list[tuple[str, int, str]]:
        seeds, seen = [], set()
        for ref, pos, alt in PROTEIN_CHANGE_RE.findall(hgvs):
            if pos in seen or not _is_missense_change(ref, alt):
                continue
            seen.add(pos)
            seeds.append((_to_aa3(ref), int(pos), _to_aa3(alt)))
        return seeds

    def gate(self, variant: dict, context: ToolContext) -> bool:
        gene = context.field("Gene")
        hgvs = context.field("HGVS")
        if gene == "NA" or hgvs == "NA":
            return False
        return bool(self._missense_seeds(hgvs))

    def _esearch_ids(self, term: str) -> tuple[list[str], int]:
        try:
            data = _ncbi_get(
                "esearch.fcgi",
                {"db": "clinvar", "term": term, "retmode": "json", "retmax": _MAX_RECORDS},
                self.timeout,
            ).json()["esearchresult"]
            return data.get("idlist", []), int(data.get("count", 0))
        except Exception as e:
            raise ToolFetchError(f"ClinVar esearch failed for term={term!r}: {e}") from e

    def _esummary(self, ids: list[str]) -> list[dict]:
        out = []
        for i in range(0, len(ids), _ESUMMARY_BATCH):
            batch = ids[i:i + _ESUMMARY_BATCH]
            try:
                result = _ncbi_get(
                    "esummary.fcgi",
                    {"db": "clinvar", "id": ",".join(batch), "retmode": "json"},
                    self.timeout,
                ).json().get("result", {})
            except Exception as e:
                raise ToolParseError(f"ClinVar esummary failed for {len(batch)} ids: {e}") from e
            out.extend(result.get(uid, {}) | {"uid": uid} for uid in result.get("uids", []))
        return out

    def _fetch_records(self, gene: str) -> list[dict] | None:
        records = []
        for term in (_PLP_TERM, _BLB_TERM):
            ids, count = self._esearch_ids(f"{gene}[gene] AND {term} AND {_MISSENSE_TERM}")
            if count > len(ids):
                logger.warning("[ClinVarHotspot] %s: %d records exceed cap %d — not evaluable.",
                               gene, count, _MAX_RECORDS)
                return None
            for obj in self._esummary(ids):
                m = _TITLE_RE.search(obj.get("title", ""))
                if not m:
                    continue
                nt, ref, pos, alt = m.group(1), m.group(2), int(m.group(3)), m.group(4)
                if not _is_missense_change(ref, alt):
                    continue
                classification = (obj.get("germline_classification", {}) or {}).get("description", "") or ""
                records.append({
                    "nt": nt, "ref": ref, "pos": pos, "alt": alt, "uid": obj["uid"],
                    "classification": classification,
                    "pathogenic": is_pathogenic_clinvar(classification),
                })
        return records

    def _get_records(self, gene: str) -> list[dict] | None:
        key = gene.upper()
        if key not in self._records_cache:
            with self._cache_lock:
                if key not in self._records_cache:
                    self._records_cache[key] = self._fetch_records(gene)
        return self._records_cache[key]

    def run(self, variant: dict, context: ToolContext) -> str | None:
        gene = context.field("Gene")
        hgvs = context.field("HGVS")
        seeds = self._missense_seeds(hgvs)
        if not seeds:
            return None

        header = f"CLINVAR HOTSPOT WINDOW ({gene}):"
        records = self._get_records(gene)
        if records is None:
            return (
                f"{header}\nPM1 HOTSPOT: NOT EVALUABLE — too many ClinVar missense "
                "records for this gene to count within the window."
            )

        own_cdnas = set(re.findall(r"c\.[^\s:;|]+", hgvs))
        lines, met = [], False
        for ref, pos, alt in seeds:
            # A ClinVar record at this exact position with a different
            # reference residue means ClinVar's transcript numbers this gene
            # differently from this HGVS segment — its window is meaningless.
            if any(r["pos"] == pos and r["ref"].lower() != ref.lower() for r in records):
                lines.append(
                    f"Residue p.{ref}{pos}{alt}: ClinVar reference residue at position {pos} "
                    "differs (transcript numbering mismatch) — window not evaluated."
                )
                continue
            window = [
                r for r in records
                if abs(r["pos"] - pos) <= WINDOW_AA and r["nt"] not in own_cdnas
                and not (r["pos"] == pos and r["alt"].lower() == alt.lower())
            ]
            plp = [r for r in window if r["pathogenic"]]
            blb = [r for r in window if not r["pathogenic"]]
            seed_met = len(plp) >= MIN_PATHOGENIC and len(blb) <= MAX_BENIGN
            met = met or seed_met
            lines.append(
                f"Residue p.{ref}{pos}{alt}, window {pos - WINDOW_AA}-{pos + WINDOW_AA}: "
                f"{len(plp)} P/LP missense, {len(blb)} B/LB missense (this variant's own "
                f"record excluded) — {'MET' if seed_met else 'NOT MET'}"
            )
            for r in sorted(window, key=lambda r: r["pos"]):
                lines.append(
                    f"  - {r['nt']} (p.{r['ref']}{r['pos']}{r['alt']}): {r['classification']} "
                    f"— ClinVar Variation ID {r['uid']}"
                )

        verdict = "MET" if met else "NOT MET"
        rule = (
            f"Rule: >= {MIN_PATHOGENIC} P/LP and <= {MAX_BENIGN} B/LB ClinVar missense "
            f"variants within +/-{WINDOW_AA} residues (VarSome hotspot rule)."
        )
        return "\n".join([header, *lines, rule, f"PM1 HOTSPOT: {verdict}"])

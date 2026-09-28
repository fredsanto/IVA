"""
pipeline/tools/clinvar_residue_search.py — deterministic ClinVar query for
every classified variant at this variant's own protein residue(s).

Replaces reliance on the general-purpose websearch agent for PS1/PM5
grounding, which is a real search engine with variable ranking/recall — a
live comparison on a genuine case found it missed a directly-relevant ClinVar
record (the exact PS1 precedent) that a structured NCBI esearch found
immediately and deterministically. PS1 ("same amino acid change, different
nucleotide, established pathogenic") and PM5 ("different amino acid change,
same residue, established pathogenic") are precisely answerable from ClinVar
data alone — this tool answers them directly instead of hoping a web search
surfaces the right page.
"""

import logging
import re

from pipeline.tools.base import NetworkTool
from pipeline.tools.websearch import _ncbi_get, DEFAULT_TIMEOUT
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError
from pipeline.core.protein_change import extract_residue_queries
from pipeline.core.acmg_sf import is_pathogenic_clinvar

logger = logging.getLogger(__name__)

# Max distinct ClinVar records fetched per residue query — bounds a single
# esummary call; residues with more hits than this are truncated with a note.
_MAX_HITS_PER_RESIDUE = 15

# Parses ClinVar's own title string, e.g.
# "NM_000000.1(GENE_X):c.100A>C (p.Lys34Thr)" -> nucleotide + protein change.
_TITLE_RE = re.compile(r":(c\.[^\s(]+)\s*\(p\.([A-Za-z*]{1,3}\d+[A-Za-z*]{1,3})\)")
_HIT_POSITION_RE = re.compile(r"\d+")


class ClinVarResidueSearchTool(NetworkTool):
    """
    For each distinct protein residue in this variant's HGVS, queries ClinVar
    (NCBI esearch + esummary) for every classified variant reported at that
    exact residue, then classifies each hit against this variant's own change:
      - identical destination amino acid, different nucleotide  -> PS1 candidate
      - different destination amino acid                        -> PM5 candidate
      - identical nucleotide                                    -> this variant's
                                                                     own record (skip)

    gate():  runs when the HGVS yields at least one parseable protein change.
    run():   esearch (`{gene}[gene] AND {aa3}{pos}[variant name]`) then a
             single batched esummary call for all returned IDs — 2 NCBI
             requests per distinct residue, not per hit.
    """

    name        = "clinvar_residue_search"
    description = (
        "Deterministic ClinVar query for every classified variant at this "
        "variant's own protein residue(s) — grounds PS1 (same AA change, "
        "different nucleotide) and PM5 (different AA change) directly from "
        "ClinVar instead of relying on general web search recall."
    )

    timeout: int = DEFAULT_TIMEOUT

    def gate(self, variant: dict, context: ToolContext) -> bool:
        gene = context.field("Gene")
        hgvs = context.field("HGVS")
        if gene == "NA" or hgvs == "NA":
            return False
        return bool(extract_residue_queries(hgvs))

    def _esearch_ids(self, gene: str, aa3_position: str) -> list[str]:
        try:
            data = _ncbi_get(
                "esearch.fcgi",
                {
                    "db": "clinvar",
                    "term": f"{gene}[gene] AND {aa3_position}[variant name]",
                    "retmode": "json",
                    "retmax": _MAX_HITS_PER_RESIDUE,
                },
                self.timeout,
            ).json()
            return data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            raise ToolFetchError(
                f"ClinVar esearch failed for {gene} {aa3_position}: {e}"
            ) from e

    def _esummary(self, ids: list[str]) -> dict:
        if not ids:
            return {}
        try:
            data = _ncbi_get(
                "esummary.fcgi",
                {"db": "clinvar", "id": ",".join(ids), "retmode": "json"},
                self.timeout,
            ).json()
            return data.get("result", {})
        except Exception as e:
            raise ToolParseError(f"ClinVar esummary failed for ids={ids}: {e}") from e

    def run(self, variant: dict, context: ToolContext) -> str | None:
        gene = context.field("Gene")
        hgvs = context.field("HGVS")
        seeds = extract_residue_queries(hgvs)
        if not seeds:
            return None

        blocks = []
        for seed in seeds:
            aa3_position = seed["query"].removeprefix("p.")  # e.g. "Lys34"
            own_change   = seed["own_change"]                 # e.g. "p.Lys34Thr"
            own_aa3      = seed["own_aa3"]                     # e.g. "Thr"
            own_position = seed["position"]                    # e.g. "34"

            ids = self._esearch_ids(gene, aa3_position)
            summaries = self._esummary(ids)

            hits = []
            for uid in summaries.get("uids", []):
                obj = summaries.get(uid, {})
                title = obj.get("title", "")
                m = _TITLE_RE.search(title)
                if not m:
                    continue
                hit_nt, hit_change = m.group(1), m.group(2)

                # NCBI's [variant name] field does fuzzy/tokenized matching,
                # not exact substring — a query for a residue with zero real
                # hits (e.g. a non-canonical transcript's numbering) can still
                # return unrelated hits at a DIFFERENT position. Discard any
                # result whose own position doesn't match this seed's exactly.
                hit_pos_m = _HIT_POSITION_RE.search(hit_change)
                if not hit_pos_m or hit_pos_m.group(0) != own_position:
                    continue

                hit_aa3 = re.match(r"[A-Za-z*]{1,3}\d+([A-Za-z*]{1,3})", hit_change)
                hit_aa3 = hit_aa3.group(1) if hit_aa3 else None

                germline = obj.get("germline_classification", {}) or {}
                classification = germline.get("description", "") or ""
                review_status  = germline.get("review_status", "") or ""

                if hit_aa3 is None:
                    continue
                if hit_aa3.upper() == own_aa3.upper():
                    tag = "SAME amino acid change as this variant"
                else:
                    tag = "DIFFERENT amino acid change than this variant"

                hits.append({
                    "nt": hit_nt, "change": hit_change, "classification": classification,
                    "review_status": review_status, "tag": tag, "variation_id": uid,
                    "pathogenic": is_pathogenic_clinvar(classification),
                })

            if not hits:
                blocks.append(
                    f"Residue {own_change} ({aa3_position}): no other classified ClinVar "
                    "variants found at this residue."
                )
                continue

            lines = [f"Residue {own_change} ({aa3_position}) — {len(hits)} ClinVar record(s) at this position:"]
            for h in hits:
                path_flag = " [PATHOGENIC/LIKELY PATHOGENIC]" if h["pathogenic"] else ""
                lines.append(
                    f"  - {h['nt']} (p.{h['change']}): {h['classification']} "
                    f"({h['review_status']}) — {h['tag']}{path_flag} "
                    f"— ClinVar Variation ID {h['variation_id']} "
                    f"(https://www.ncbi.nlm.nih.gov/clinvar/variation/{h['variation_id']}/)"
                )
            blocks.append("\n".join(lines))

        if not blocks:
            return None

        header = f"CLINVAR RESIDUE-LEVEL SEARCH ({gene}):"
        guidance = (
            "PS1: use a hit tagged \"SAME amino acid change\" + pathogenic/likely "
            "pathogenic + a nucleotide change different from this variant's own. "
            "PM5: use a hit tagged \"DIFFERENT amino acid change\" + pathogenic/"
            "likely pathogenic. A hit with the SAME nucleotide as this variant is "
            "this variant's own record, not a separate precedent. When citing PS1 "
            "or PM5 from a hit here, include its ClinVar Variation ID/URL in the "
            "justification so the precedent is independently checkable."
        )
        return header + "\n" + "\n".join(blocks) + f"\n{guidance}"

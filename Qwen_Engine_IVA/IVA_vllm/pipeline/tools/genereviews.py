"""
pipeline/tools/genereviews.py — GeneReviews clinical-description fetch.

Retrieves the canonical GeneReviews chapter(s) for a gene via NCBI E-utilities
(gene -> elink -> books) and returns each chapter's structured summary — clinical
characteristics, diagnosis/testing and genetic counseling (mode of inheritance) —
from the chapter's own PubMed record. The Bookshelf chapter page itself is not
fetched: it answers scripted requests with a CAPTCHA page.

Why this exists: LitVar2's PubMed tracks (litvar2.py) rank a broad, uncurated
paper pool by relevance or publication date (deliberately pub_date-sorted in
_gene_search, to surface recently-characterized gene-disease links). For a
heavily published gene that also causes a rare syndrome — e.g. a common
cancer gene with thousands of oncology papers — that pool can end up
dominated by unrelated or narrow recent case reports (e.g. a single 2026
hepatic-complication case report), silently excluding the one canonical
phenotype description a clinician would actually consult. This surfaced in
practice: such a gene's rare syndrome was scored "no neurological association"
from retrieved literature, when GeneReviews' own chapter for that syndrome explicitly
lists "developmental delay / intellectual disability" and "epilepsy" as part
of the syndrome. GeneReviews is authored per-gene/per-condition specifically
to be the single curated summary, so this tool bypasses the PubMed
relevance/date lottery entirely and fetches it directly.

Gene-scoped, not variant-scoped: one lookup per gene per run, cached at class
level (same pattern as GnomadConstraintTool._constraint_cache).
"""

import logging
import threading
import xml.etree.ElementTree as ET

from pipeline.tools.base import NetworkTool
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError
from pipeline.tools.websearch import _ncbi_get, _clean_xml_text

logger = logging.getLogger(__name__)

# Labelled sections of a GeneReviews chapter's PubMed summary kept, in output
# order (MANAGEMENT is dropped — not used for variant interpretation).
_SUMMARY_LABELS = ("CLINICAL CHARACTERISTICS", "DIAGNOSIS/TESTING", "GENETIC COUNSELING")

_MAX_CHAPTERS   = 4       # cap chapters fetched per gene (a few genes link many)


class GeneReviewsTool(NetworkTool):
    """
    Fetches the GeneReviews chapter(s) linked to a gene and extracts the
    curated clinical-description section for each.

    gate():  runs when Gene is present.
    run():   resolved once per gene, cached at class level.
    """

    name        = "genereviews"
    description = (
        "Fetches the canonical GeneReviews clinical-description section(s) for "
        "this gene directly from NCBI Bookshelf — the curated per-gene phenotype "
        "summary, independent of PubMed search ranking."
    )

    timeout: int = 15

    # Class-level cache: {"GENE": formatted output block str, or None if no
    # chapter found} — shared across all variants in that gene for the run.
    _chapter_cache: dict[str, str | None] = {}
    _cache_lock = threading.Lock()

    def gate(self, variant: dict, context: ToolContext) -> bool:
        gene = context.field("Gene")
        return gene != "NA" and gene.strip() != ""

    # ── NCBI resolution: gene symbol -> GeneReviews chapter accession IDs ──

    def _resolve_gene_id(self, gene: str) -> str | None:
        try:
            data = _ncbi_get(
                "esearch.fcgi",
                {"db": "gene", "term": f"{gene}[sym] AND Homo sapiens[orgn]",
                 "retmode": "json", "retmax": 1},
                self.timeout,
            ).json()
        except Exception as e:
            raise ToolFetchError(f"NCBI gene esearch failed for {gene}: {e}") from e
        try:
            ids = data["esearchresult"]["idlist"]
        except (KeyError, TypeError) as e:
            raise ToolParseError(f"Failed to parse gene esearch response for {gene}: {e}") from e
        return ids[0] if ids else None

    def _resolve_book_uids(self, gene: str, gene_id: str) -> list[str]:
        try:
            data = _ncbi_get(
                "elink.fcgi",
                {"dbfrom": "gene", "db": "books", "id": gene_id, "retmode": "json"},
                self.timeout,
            ).json()
        except Exception as e:
            raise ToolFetchError(f"NCBI gene->books elink failed for {gene} (gene_id={gene_id}): {e}") from e

        linksets = data.get("linksets", [])
        if not linksets:
            return []
        for linksetdb in linksets[0].get("linksetdbs", []):
            if linksetdb.get("linkname") == "gene_books":
                return linksetdb.get("links", [])
        return []

    def _resolve_chapters(self, gene: str, book_uids: list[str]) -> list[dict]:
        """
        Return [{title, accession, pubdate}] for each distinct GeneReviews
        chapter linked to this gene. The raw elink result mixes in
        non-chapter sub-objects (tables, sections) and occasionally other
        Bookshelf sources entirely (e.g. StemBook) that happen to cite the
        same gene — filtered here to book=='gene' (GeneReviews specifically)
        and rtype=='chapter' (the top-level condition article, not a
        table/section fragment of it).
        """
        if not book_uids:
            return []
        try:
            data = _ncbi_get(
                "esummary.fcgi",
                {"db": "books", "id": ",".join(book_uids), "retmode": "json"},
                self.timeout,
            ).json()
        except Exception as e:
            raise ToolFetchError(f"NCBI books esummary failed for {gene}: {e}") from e

        result = data.get("result", {})
        chapters = []
        seen_accessions = set()
        for uid in result.get("uids", []):
            entry = result.get(uid, {})
            if entry.get("book") != "gene" or entry.get("rtype") != "chapter":
                continue
            accession = entry.get("chapteraccessionid") or entry.get("accessionid")
            if not accession or accession in seen_accessions:
                continue
            seen_accessions.add(accession)
            chapters.append({
                "title":   entry.get("title", "Untitled"),
                "accession": accession,
                "pubdate": entry.get("pubdate", "unknown"),
            })
        return chapters[:_MAX_CHAPTERS]

    # ── chapter summaries from PubMed ───────────────────────────────────

    def _fetch_summaries(self, gene: str, accessions: list[str]) -> dict[str, str]:
        """{accession: summary text} from each chapter's PubMed record
        (PubmedBookArticle, abstract sections labelled per _SUMMARY_LABELS)."""
        term = " OR ".join(f"{acc}[aid]" for acc in accessions)
        try:
            pmids = _ncbi_get(
                "esearch.fcgi",
                {"db": "pubmed", "term": term, "retmode": "json", "retmax": len(accessions)},
                self.timeout,
            ).json()["esearchresult"]["idlist"]
        except Exception as e:
            raise ToolFetchError(f"PubMed esearch for GeneReviews chapters failed for {gene}: {e}") from e
        if not pmids:
            return {}
        try:
            root = ET.fromstring(_ncbi_get(
                "efetch.fcgi",
                {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
                self.timeout,
            ).text)
        except ET.ParseError as e:
            raise ToolParseError(f"GeneReviews PubMed XML unparseable for {gene}: {e}") from e
        except Exception as e:
            raise ToolFetchError(f"PubMed efetch for GeneReviews chapters failed for {gene}: {e}") from e

        summaries = {}
        for article in root.iter("PubmedBookArticle"):
            accession = next((i.text for i in article.iter("ArticleId")
                              if i.get("IdType") == "bookaccession"), None)
            sections = {
                t.get("Label"): _clean_xml_text(ET.tostring(t, encoding="unicode"))
                for t in article.iter("AbstractText")
            }
            kept = [f"{label}: {sections[label]}" for label in _SUMMARY_LABELS if sections.get(label)]
            if accession and kept:
                summaries[accession] = "\n".join(kept)
        return summaries

    # ── gene-level resolution, cached ───────────────────────────────────

    def _get_gene_block(self, gene: str) -> str | None:
        cache_key = gene.upper()
        if cache_key not in self._chapter_cache:
            with self._cache_lock:
                if cache_key not in self._chapter_cache:   # double-checked locking
                    self._chapter_cache[cache_key] = self._build_gene_block(gene)
        return self._chapter_cache[cache_key]

    def _build_gene_block(self, gene: str) -> str | None:
        gene_id = self._resolve_gene_id(gene)
        if gene_id is None:
            return None
        book_uids = self._resolve_book_uids(gene, gene_id)
        if not book_uids:
            return None
        chapters = self._resolve_chapters(gene, book_uids)
        if not chapters:
            return None

        summaries = self._fetch_summaries(gene, [ch["accession"] for ch in chapters])
        blocks = [
            f"--- {ch['title']} (GeneReviews {ch['accession']}, updated {ch['pubdate']}) ---\n"
            f"{summaries[ch['accession']]}"
            for ch in chapters if ch["accession"] in summaries
        ]
        return "\n\n".join(blocks) if blocks else None

    def run(self, variant: dict, context: ToolContext) -> str | None:
        gene = context.field("Gene")
        block = self._get_gene_block(gene)
        if block is None:
            return (
                f"GENEREVIEWS ({gene}):\n"
                "No GeneReviews chapter found for this gene (or its clinical-"
                "description section could not be located) — this gene may not "
                "yet have a curated GeneReviews entry."
            )
        return f"GENEREVIEWS ({gene}) — canonical clinical description(s):\n\n{block}"

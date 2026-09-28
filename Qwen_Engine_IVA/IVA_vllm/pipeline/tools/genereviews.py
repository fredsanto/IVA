"""
pipeline/tools/genereviews.py — GeneReviews clinical-description fetch.

Retrieves the canonical GeneReviews chapter(s) for a gene via NCBI E-utilities
(gene -> elink -> books) and extracts the "Clinical Description" section — the
curated, disease-defining phenotype summary GeneReviews maintains per gene.

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

from pipeline.tools.base import NetworkTool
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError
from pipeline.tools.websearch import _ncbi_get, _extract_body_text

logger = logging.getLogger(__name__)

BOOKSHELF_URL_TMPL = "https://www.ncbi.nlm.nih.gov/books/{accession}/"

# Section headings GeneReviews chapters use for the phenotype-defining text,
# tried in priority order — the first one found is where extraction starts.
_CLINICAL_SECTION_HEADINGS = (
    "Clinical Description",
    "Clinical Characteristics",
    "Suggestive Findings",
)

_SECTION_CHARS  = 3500    # window of clinical text pulled per chapter
_MAX_CHAPTERS   = 4       # cap chapters fetched per gene (a few genes link many)
_PAGE_MAX_CHARS = 200_000  # effectively "whole page" for _extract_body_text


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

    # ── page fetch + section extraction ─────────────────────────────────

    def _fetch_clinical_section(self, accession: str) -> str | None:
        url = BOOKSHELF_URL_TMPL.format(accession=accession)
        try:
            resp = self._get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"},
            )
        except Exception as e:
            raise ToolFetchError(f"Bookshelf fetch failed for {accession}: {e}") from e

        text = _extract_body_text(resp.text, max_chars=_PAGE_MAX_CHARS)
        for heading in _CLINICAL_SECTION_HEADINGS:
            idx = text.find(heading)
            if idx != -1:
                return text[idx: idx + _SECTION_CHARS]
        return None   # heading not found — page likely malformed or not a chapter

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

        blocks = []
        for ch in chapters:
            try:
                section = self._fetch_clinical_section(ch["accession"])
            except ToolFetchError as e:
                logger.warning("GeneReviews chapter fetch failed for %s (%s): %s",
                                gene, ch["accession"], e)
                continue
            if not section:
                continue
            blocks.append(
                f"--- {ch['title']} (GeneReviews {ch['accession']}, updated {ch['pubdate']}) ---\n"
                f"{section}"
            )
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

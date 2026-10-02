"""
pipeline/tools/websearch_agent.py — literature search for genes with no curated
disease entry.

A gene has three possible states:
  1. Known disease gene — covered by the curated sources (GeneReviews, OMIM
     phenotype, MedGen, CGD) and the LitVar2 tool.
  2. Newly published disease gene — not yet in any curated source, but already
     described in the literature (journal article or bioRxiv/medRxiv preprint).
  3. Gene with no known phenotype — curated sources and literature are empty.

This tool covers state 2. It runs the literature search only when every
curated source is empty for the gene, so the output never duplicates what the
curated sources already say:
  - PubMed (E-utilities) for journal articles and Crossref (openRxiv, the
    publisher of bioRxiv and medRxiv) for preprints — title/abstract only,
    gene + generic disease vocabulary, newest first. If Crossref fails, the
    PubMed part is kept and the header says so.
  - Titles scored by the SLM for a named disease caused by the gene, top
    abstracts summarised — same scoring/summary steps as LitVar2's
    condition-inventory track.

The patient phenotype is never used: this block also feeds Stage 1b's
gene-phenotype extraction, which must see gene-level evidence only.

Also runs, independently of the curated-source check, the forced ClinVar
submission-level fetch for P/LP variants (per-submitter tally, rationale and
cited PMIDs — grounds PS3).

Replaces the earlier ReAct web-search agent (Brave/DuckDuckGo via ddgs), which
was rate-limited or timed out on nearly every query in production.
"""

from __future__ import annotations

import logging
import re
import threading
import time

import requests

from pipeline.tools.base import SLMTool
from pipeline.core.context import ToolContext
from pipeline.core.errors import PipelineError, ToolFetchError, ToolParseError
from pipeline.core.acmg_sf import is_pathogenic_clinvar
from pipeline.tools.websearch import _ncbi_get, _clean_xml_text, DEFAULT_TIMEOUT
from pipeline.tools.ncbi import NCBIFetchTool
from pipeline.tools.genereviews import GeneReviewsTool
from pipeline.tools.litvar2 import LitVar2SummaryTool

logger = logging.getLogger(__name__)

CROSSREF_WORKS_URL = "https://api.crossref.org/works"
CROSSREF_OPENRXIV_MEMBER = "54368"   # openRxiv — publisher of bioRxiv and medRxiv

# Generic disease vocabulary — never patient-derived. Matched in title/abstract.
_DISEASE_VOCAB = (
    "disease", "disorder", "syndrome", "patient", "patients", "proband",
    "de novo", "biallelic", "dominant", "recessive", "x-linked",
)
_MAX_PER_SOURCE = 20      # newest records taken from each of PubMed / Crossref


def _quote(word: str) -> str:
    return f'"{word}"' if (" " in word or "-" in word) else word


def _crossref_get(params: dict, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Crossref works query with the same retry policy as _ncbi_get (429/5xx/network)."""
    headers = {"User-Agent": "IVA-pipeline/1.0 (clinical variant interpretation)"}
    for attempt in range(4):
        try:
            resp = requests.get(CROSSREF_WORKS_URL, params=params, headers=headers, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            wait = 2.0 ** attempt
            logger.warning("Crossref request failed (%s) — retrying in %.0fs (attempt %d/4)",
                           type(e).__name__, wait, attempt + 1)
            time.sleep(wait)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 2.0 ** attempt
            logger.warning("Crossref %d response — backing off %.0fs (attempt %d/4)",
                           resp.status_code, wait, attempt + 1)
            time.sleep(wait)
            continue
        try:
            resp.raise_for_status()
            return resp.json()
        except (requests.HTTPError, ValueError) as e:
            raise ToolFetchError(f"Crossref search failed: {e}") from e
    raise ToolFetchError("Crossref search failed after 4 attempts")


def _mentions(text: str, gene: str) -> bool:
    """True when title/abstract text names the gene as a word and uses at least
    one term of the generic disease vocabulary."""
    low = text.lower()
    words = set(re.findall(r"[a-z0-9][a-z0-9-]*", low))
    return gene.lower() in words and any(
        (w in low) if " " in w else (w in words) for w in _DISEASE_VOCAB
    )


class WebSearchAgentTool(SLMTool):
    """Literature search (PubMed + bioRxiv/medRxiv preprints) for genes absent
    from every curated source, plus the forced ClinVar submission-level check."""

    name        = "websearch_agent"
    description = (
        "Searches PubMed and bioRxiv/medRxiv (via Crossref) for a gene's "
        "disease literature when GeneReviews, OMIM, MedGen and CGD are all empty."
    )

    # Class-level caches, keyed by upper-case gene symbol, shared across all
    # variants in that gene for the process.
    _curated_cache: dict[str, list[str]] = {}
    _literature_cache: dict[str, str] = {}
    _cache_lock = threading.Lock()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tl = threading.local()   # thread-local storage for _last_trace
        self._genereviews = GeneReviewsTool()
        self._litvar2 = LitVar2SummaryTool()

    @property
    def _last_trace(self) -> list[str]:
        return getattr(self._tl, "trace", [])

    @_last_trace.setter
    def _last_trace(self, value: list[str]) -> None:
        self._tl.trace = value

    # ── curated-source check ─────────────────────────────────────────────────

    def _genereviews_chapter_count(self, gene: str, gene_id: str) -> int:
        # Chapter existence from E-utilities metadata, not the chapter page —
        # the Bookshelf HTML page answers scripted requests with a CAPTCHA.
        book_uids = self._genereviews._resolve_book_uids(gene, gene_id)
        return len(self._genereviews._resolve_chapters(gene, book_uids))

    def _medgen_disease_count(self, gene: str, gene_id: str) -> int:
        try:
            data = _ncbi_get(
                "elink.fcgi",
                {"dbfrom": "gene", "db": "medgen", "id": gene_id, "retmode": "json"},
                self.timeout,
            ).json()
        except Exception as e:
            raise ToolFetchError(f"NCBI gene->medgen elink failed for {gene}: {e}") from e
        for linkset in data.get("linksets", [])[:1]:
            for linksetdb in linkset.get("linksetdbs", []):
                if linksetdb.get("linkname") == "gene_medgen_diseases":
                    return len(linksetdb.get("links", []))
        return 0

    def _curated_sources(self, gene: str, omim_phenotype: str) -> list[str]:
        """Names of the curated sources that hold a disease entry for this gene."""
        found = []
        if omim_phenotype not in ("", "NA"):
            found.append("OMIM phenotype")
        key = gene.upper()
        if key not in self._curated_cache:
            gene_level = []
            gene_id = self._genereviews._resolve_gene_id(gene)
            if gene_id is not None:
                if self._genereviews_chapter_count(gene, gene_id) > 0:
                    gene_level.append("GeneReviews")
                if self._medgen_disease_count(gene, gene_id) > 0:
                    gene_level.append("MedGen")
            if self._litvar2._get_cgd_conditions(gene):
                gene_level.append("CGD")
            with self._cache_lock:
                self._curated_cache[key] = gene_level
        return found + self._curated_cache[key]

    # ── literature search ────────────────────────────────────────────────────

    def _pubmed_newest(self, gene: str) -> list[str]:
        vocab = " OR ".join(f"{_quote(w)}[tiab]" for w in _DISEASE_VOCAB)
        term = f"({gene}[tiab] OR {gene}[Gene]) AND ({vocab})"
        try:
            data = _ncbi_get(
                "esearch.fcgi",
                {"db": "pubmed", "term": term, "retmax": _MAX_PER_SOURCE,
                 "sort": "pub_date", "retmode": "json"},
                self.timeout,
            ).json()
        except Exception as e:
            raise ToolFetchError(f"PubMed esearch failed for {gene}: {e}") from e
        try:
            return data["esearchresult"]["idlist"]
        except (KeyError, TypeError) as e:
            raise ToolParseError(f"Failed to parse PubMed esearch for {gene}: {e}") from e

    def _preprints_newest(self, gene: str) -> list[dict]:
        """Newest bioRxiv/medRxiv preprints naming the gene with a disease term
        in title/abstract, as [{id, title, abstract, year, server, url}]."""
        data = _crossref_get({
            "query.bibliographic": gene,
            "filter": f"member:{CROSSREF_OPENRXIV_MEMBER},type:posted-content",
            "sort": "created", "order": "desc", "rows": _MAX_PER_SOURCE,
        }, self.timeout)
        try:
            items = data["message"]["items"]
        except (KeyError, TypeError) as e:
            raise ToolParseError(f"Failed to parse Crossref response for {gene}: {e}") from e
        preprints = []
        for it in items:
            title = _clean_xml_text(" ".join(it.get("title") or []))
            abstract = _clean_xml_text(it.get("abstract") or "")
            if not title or not _mentions(f"{title} {abstract}", gene):
                continue
            posted = ((it.get("posted") or {}).get("date-parts") or [[None]])[0][0]
            preprints.append({
                "id": it["DOI"],
                "title": title,
                "abstract": (abstract or "No abstract available.")[: self._litvar2.max_chars],
                "year": str(posted or "n.d."),
                "server": ((it.get("institution") or [{}])[0].get("name") or "preprint"),
                "url": f"https://doi.org/{it['DOI']}",
            })
        return preprints

    def _literature_block(self, gene: str, context: ToolContext) -> tuple[str, bool]:
        """(block, complete) — complete is False when the preprint source failed
        and only PubMed was searched; such a block is not cached."""
        pubmed_ids = self._pubmed_newest(gene)
        try:
            found_preprints = self._preprints_newest(gene)
            complete = True
        except PipelineError as e:
            logger.warning("Preprint search failed for %s (PubMed results kept): %s", gene, e)
            found_preprints, complete = [], False

        # Candidate pool: PubMed IDs + preprints. A preprint whose title matches
        # a pool paper is its published version's earlier copy — dropped.
        titles = self._litvar2._fetch_titles(pubmed_ids) if pubmed_ids else {}
        seen_titles = {t.lower().strip(" .") for t in titles.values()}
        preprints: dict[str, dict] = {}
        for rec in found_preprints:
            norm = rec["title"].lower().strip(" .")
            if norm in seen_titles:
                continue
            seen_titles.add(norm)
            titles[rec["id"]] = rec["title"]
            preprints[rec["id"]] = rec

        sources = ("PubMed + bioRxiv/medRxiv (Crossref)" if complete else
                   "PubMed only — bioRxiv/medRxiv search unavailable (Crossref error)")
        header = (
            f"LITERATURE SEARCH for {gene} — no curated entry (GeneReviews, OMIM, "
            f"MedGen, CGD all empty); {sources}, title/abstract, newest first, "
            f"phenotype-agnostic"
        )
        if not titles:
            return (f"{header}\n"
                    f"No publication or preprint names {gene} together with a disease, "
                    f"disorder, syndrome or patient in its title/abstract."), complete

        question = (
            f"Does this paper name or describe a specific disease, syndrome, or "
            f"clinical condition caused by {gene}? Score high for ANY named "
            f"condition, low for papers that are purely population-genetics, "
            f"mechanism-only, or that mention {gene} only as one gene in a list."
        )
        selected = self._litvar2._select_relevant_pmids(titles, question, context)
        papers = self._litvar2._fetch_abstracts([i for i in selected if i not in preprints])
        for i in selected:
            if i in preprints:
                papers[i] = preprints[i]
        if not papers:
            return (f"{header}\n[{len(titles)} screened, 0 selected]\n"
                    f"No screened paper names a disease caused by {gene}."), complete

        summary_question = (
            f"List every distinct disease, condition, or syndrome these abstracts "
            f"attribute to {gene}, with the evidence type (patients/families reported, "
            f"functional or animal-model data) and mode of inheritance where stated."
        )
        summary = self._litvar2._summarise(papers, summary_question, gene, context)
        # Preprints are keyed by DOI; the summariser labels every source "PMID:".
        summary = summary.replace("PMID:10.", "preprint doi:10.")

        source_lines = "\n".join(
            f"  - [{p['server']} preprint {i}, {p['year']}] {p['title']}  {p['url']}"
            if i in preprints else
            f"  - [PMID:{i}, {p.get('year', 'n.d.')}] {p['title']}  {p['url']}"
            for i, p in papers.items()
        )
        return (f"{header}\n[{len(titles)} screened, {len(papers)} selected]\n\n"
                f"{summary}\n\nSources:\n{source_lines}"), complete

    # ── pipeline entry point ─────────────────────────────────────────────────

    def run(self, variant: dict, context: ToolContext) -> str | None:
        self._last_trace = []
        gene = context.field("Gene")
        hgvs = context.field("HGVS")
        parts = []

        if gene != "NA":
            try:
                curated = self._curated_sources(gene, context.field("OMIM_phenotype"))
            except PipelineError as e:
                # Curated state unknown — search rather than risk missing a new gene.
                logger.warning("Curated-source check failed for %s: %s", gene, e)
                self._last_trace.append(f"Curated-source check FAILED ({type(e).__name__}) — searching anyway")
                curated = []
            if curated:
                self._last_trace.append(
                    f"Curated disease entry present ({', '.join(curated)}) — literature search skipped"
                )
            else:
                key = gene.upper()
                block = self._literature_cache.get(key)
                if block is None:
                    try:
                        block, complete = self._literature_block(gene, context)
                        if complete:
                            with self._cache_lock:
                                self._literature_cache[key] = block
                    except PipelineError as e:
                        logger.warning("Literature search failed for %s: %s", gene, e)
                        block = f"LITERATURE SEARCH for {gene} [FAILED — {type(e).__name__}]: {e}"
                self._last_trace.append("No curated disease entry — literature search run")
                parts.append(block)

        # Forced ClinVar submission-level check (P/LP variants) — the CSV's
        # ClinVar_class is only an aggregate label; this pulls the per-submitter
        # tally and, for each P/LP submission, its rationale and cited PMIDs.
        clinvar_class = context.field("ClinVar_class")
        if is_pathogenic_clinvar(clinvar_class) and gene != "NA":
            ncbi = NCBIFetchTool()
            variation_id = ncbi.resolve_clinvar_id(gene, hgvs)
            if variation_id:
                parts.append(
                    f"CLINVAR SUBMISSION-LEVEL CHECK (forced — ClinVar_class={clinvar_class}):\n"
                    f"{ncbi._fetch_clinvar(variation_id)}"
                )
            else:
                parts.append(
                    f"CLINVAR SUBMISSION-LEVEL CHECK (forced — ClinVar_class={clinvar_class}): "
                    "could not resolve a ClinVar variation ID for gene+HGVS — classification "
                    "breakdown and PS3 evidence cannot be pulled from ClinVar for this variant."
                )

        return "\n\n".join(parts) if parts else None

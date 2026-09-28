"""
pipeline/tools/litvar2.py — LitVar2-powered literature summarisation tool.

Search strategy (gene-first, three tracks):

  1. Gene + generic disease/inheritance vocabulary (primary, PHENOTYPE-AGNOSTIC —
     always runs when Gene is present)
       Uses PubMed esearch with "{gene}[Gene] AND (disease OR syndrome OR <inheritance
       terms>)" — deliberately NEVER the patient's phenotype or the LLM-resolved
       self._disease_query — to retrieve papers describing what this gene causes, full
       stop. One API call, no LitVar2 loop.
       - Zero hits  → explicit "not linked to disease" string (negative signal for triage)
       - Non-zero   → titles → SLM filter (gene-disease relevance, not patient match) →
         abstracts → SLM summary (objective inventory of the gene's conditions; does not
         compare to or judge fit against the patient's case)
       Comparing this evidence against the patient's own phenotype (CLUSTER_PHENOTYPE) is
       Stage 1 reasoning's job (prompts/reasoning.txt) — it sees this block AND the patient
       phenotype together. Anchoring retrieval to the patient's phenotype here would
       pre-judge that comparison before Stage 1 runs, and — since disease-query resolution
       and relevance-scoring are themselves LLM calls — make the evidence pool (and every
       downstream stage, including MOI classification) sensitive to run-to-run LLM
       sampling noise on top of the gene's actual literature. See _gene_search's docstring.

  2. Gene + known disease association (supplemental — runs when a disease term is available)
       Bridges the terminology gap between how the patient's condition is described and how
       OMIM/CGD name the canonical disease for that gene.
       Disease term source priority:
         a. OMIM_phenotype from the variant dict (already curated for this variant)
         b. NHGRI Clinical Genomic Database (CGD) conditions for the gene — downloaded once
            per process and cached at the class level; empty dict on download failure.
       - No disease term available → track skipped silently
       - Term found → titles → SLM filter (against the gene's own known condition, not
         the patient's phenotype) → abstracts → SLM summary (framed on patient phenotype
         for clinical readability)

  3. Variant-level search (supplemental — runs when the variant resolves in LitVar2 by
     its own c./p. names, else by RS_ID; the one track still
     scoped to the patient's own phenotype via self._disease_query — "does this paper
     discuss THIS variant in the patient's condition" is a narrower, legitimately
     patient-anchored question)
       Uses LitVar2 variant publications endpoint (variant ID from c./p. autocomplete).
       - No relevant papers → omitted from output
       - Papers found       → titles → SLM filter → abstracts → SLM summary

  Tracks 1 and 2 now answer closely related questions from different angles: track 1 asks
  "what does this gene cause, per generic disease/inheritance literature?", track 2 asks
  "what does OMIM/CGD's own curated condition entry for this gene say?" — both feed the
  same downstream phenotype comparison rather than pre-deciding it. All tracks that yield
  evidence are combined in the output, separated by dividers. Section headers embed the
  exact resolved query and source for downstream traceability, unconditionally (not just
  on a no-hits fallback path).
"""

import gzip
import io
import os
import re
import json
import time
import logging
import threading
import requests
import xml.etree.ElementTree as ET

from pipeline.tools.base import SLMTool
from pipeline.core.context import ToolContext
from pipeline.core.errors import PipelineError, ToolFetchError, ToolParseError
from pipeline.core.protein_change import PROTEIN_CHANGE_RE, _to_aa3
from pipeline.tools.clinvar_gene_stats import _FUNCTIONAL_MARKERS, _COMMENT_EXCLUDE_MARKERS

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

LITVAR2_BASE       = "https://www.ncbi.nlm.nih.gov/research/litvar2-api"
LITVAR2_URL        = f"{LITVAR2_BASE}/variant/get/{{litvar_id}}/publications"
LITVAR2_AUTOCOMPLETE_URL = f"{LITVAR2_BASE}/variant/autocomplete/"

# Extracts the protein-change token from a combined HGVS field, e.g.
# "NM_000000.1(GENE_X):c.100A>C (p.Lys34Thr)" → "p.Lys34Thr", or a bare
# "p.K34T" short form. LitVar2's autocomplete accepts either 3-letter or
# 1-letter amino acid codes.
_PROTEIN_CHANGE_RE = re.compile(r"p\.\(?([A-Za-z*]{1,3}\d+[A-Za-z*]{1,3}(?:fs\*?\d*)?)\)?")

# Extracts the cDNA-change token out of a combined/compound HGVS string, e.g.
# "GENE_X:NM_000000:exon4:c.100A>C:p.K34T" -> "c.100A>C". Same pattern as
# clinvar_gene_stats.py's own _CDNA_CHANGE_RE (independent copy — see that
# file's docstring on why this project duplicates ClinVar-lookup helpers
# across tools rather than sharing state between order-1-parallel tools).
_CDNA_CHANGE_RE = re.compile(r"c\.[^\s:;()]+")

NCBI_BASE    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Without API key: 3 req/s → 0.34 s delay.  With API key: 10 req/s → 0.11 s delay.
# Set NCBI_API_KEY env var (free key from https://www.ncbi.nlm.nih.gov/account/).
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
NCBI_DELAY   = 0.11 if NCBI_API_KEY else 0.34
CGD_URL      = "https://research.nhgri.nih.gov/CGD/download/txt/CGD.txt.gz"
CGD_MAX_CONDITIONS = 3        # cap OR terms from CGD to keep the query focused

MAX_PMIDS     = 40             # hard cap on PMIDs fetched from LitVar2
TOP_N_PAPERS  = 8              # how many titles the SLM keeps
MAX_CHARS     = 1200           # abstract truncation per paper (~full abstract)
DEFAULT_TIMEOUT = 15
SELECT_BATCH_SIZE = 10         # titles per relevance-scoring SLM call — a single call
                                # over the full pool (up to ~58 titles: 40 relevance +
                                # 20 pub_date recency, deduped) exhibits positional bias
                                # in the 9B non-thinking model: it reliably picks from the
                                # first ~10 titles and never meaningfully considers entries
                                # further down the list, regardless of actual relevance.
                                # Scoring in small batches forces the model to emit a
                                # number for every title instead of making one holistic
                                # pick over the whole list.

# Global rate limiter: serialises NCBI/LitVar2 request slots across all threads.
# With 32 concurrent workers, per-thread sleep(0.34) would burst 32 requests at
# once.  This lock + timestamp ensures at most 1 request every NCBI_DELAY seconds
# globally, staying within NCBI's 3 req/s unauthenticated limit.
_NCBI_RATE_LOCK      = threading.Lock()
_NCBI_LAST_CALL_TIME = 0.0


def _ncbi_rate_limit() -> None:
    global _NCBI_LAST_CALL_TIME
    with _NCBI_RATE_LOCK:
        elapsed = time.time() - _NCBI_LAST_CALL_TIME
        wait = NCBI_DELAY - elapsed
        if wait > 0:
            time.sleep(wait)
        _NCBI_LAST_CALL_TIME = time.time()


# ── helpers ───────────────────────────────────────────────────────────────────

def _ncbi_get(endpoint: str, params: dict, timeout: int = DEFAULT_TIMEOUT) -> requests.Response:
    _ncbi_rate_limit()
    if NCBI_API_KEY:
        params = {**params, "api_key": NCBI_API_KEY}
    for attempt in range(4):
        resp = requests.get(f"{NCBI_BASE}/{endpoint}", params=params, timeout=timeout)
        if resp.status_code == 429:
            wait = 2.0 ** attempt   # 1s, 2s, 4s, 8s
            logger.warning("NCBI rate limit (429) — backing off %.0fs (attempt %d/4)", wait, attempt + 1)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()  # re-raise on final attempt
    return resp


def _clean_xml_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", text).strip()


# ── main tool ─────────────────────────────────────────────────────────────────

class LitVar2SummaryTool(SLMTool):
    """
    Retrieves gene- and variant-level literature via PubMed and LitVar2.

    gate():  runs when RS_ID is valid OR Gene is non-empty
    run():   gene-level PubMed search always runs first (primary); LitVar2 rsID search
             runs second as supplemental variant-specific evidence when available.
             Both outputs are combined; section headers embed source and query so
             downstream reasoning stages can interpret provenance unambiguously.
    """

    name        = "litvar2_summary"
    description = (
        "Fetches variant-specific PubMed articles from LitVar2, selects the most "
        "relevant ones for the patient phenotype using an SLM, and returns a concise "
        "evidence summary derived from their abstracts."
    )

    # Class-level defaults — override at instantiation if needed
    max_pmids: int = MAX_PMIDS
    top_n:     int = TOP_N_PAPERS
    max_chars: int = MAX_CHARS
    timeout:   int = DEFAULT_TIMEOUT

    # CGD table cache — populated on first use, shared across all instances and variants
    _cgd_table: dict[str, list[str]] | None = None
    _cgd_inheritance: dict[str, str] | None = None
    _cgd_lock  = threading.Lock()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._disease_query: str | None = None
        self._disease_query_lock = threading.Lock()

    # ── gate ──────────────────────────────────────────────────────────────────

    def gate(self, variant: dict, context: ToolContext) -> bool:
        """Run when a valid rsID is present OR when a gene symbol is available."""
        rsid = context.field("RS_ID")
        gene = context.field("Gene")
        rsid_ok = rsid != "NA" and re.match(r"^rs\d+$", rsid, re.IGNORECASE) is not None
        gene_ok = gene != "NA" and gene.strip() != ""
        return rsid_ok or gene_ok

    # ── disease query resolution ──────────────────────────────────────────────

    def _resolve_disease_query(self, context: ToolContext) -> str:
        """
        Ask the SLM once to produce 2–4 PubMed-compatible disease terms from the
        patient phenotype, returned as a PubMed OR expression. Multiple synonymous
        or related terms improve recall for genes associated with phenotypically
        overlapping conditions (e.g. a gene causing microphthalmia would be missed
        by a query narrowed to a single syndromic term). Cached and reused across
        all variants in the run.
        """
        system = (
            "Given a patient phenotype, return 2 to 4 PubMed disease search terms as a "
            "comma-separated list. Include the main disease and its closest synonyms or "
            "phenotypically overlapping categories that share causal genes "
            "(e.g. for Marfan syndrome: "
            "'marfan syndrome, connective tissue disorder, aortic aneurysm, ectopia lentis'). "
            "Prefer terms indexed in MeSH. Lowercase, no punctuation except commas. "
            "Output ONLY the comma-separated terms — no explanation, no numbering, no preamble."
        )
        raw = context.llm.generate(
            system=system, user=context.patient_phenotype, max_tokens=60
        ).strip().lower()

        return self._parse_disease_terms(raw)

    @staticmethod
    def _parse_disease_terms(raw: str) -> str:
        """
        Parse SLM output into a PubMed OR expression.

        Handles: comma-separated, semicolon-separated, numbered lists ("1. term"),
        bullet points, extra whitespace. Falls back to the first four words of raw
        if nothing useful can be extracted.

        Each term is capped at 5 words. Total terms capped at 4.
        Returns a string like "term1 OR term2 OR term3".
        """
        # Strip common noise: "1." / "- " / bullet chars
        cleaned = re.sub(r"^\s*\d+\.\s*", "", raw, flags=re.MULTILINE)
        cleaned = re.sub(r"[-•*]\s*", " ", cleaned)

        # Split on comma, semicolon, or newline (model may use any separator)
        if "," in cleaned or ";" in cleaned:
            parts = re.split(r"[,;]", cleaned)
        elif "\n" in cleaned.strip():
            parts = cleaned.strip().splitlines()
        else:
            parts = [cleaned]

        terms = []
        for part in parts:
            term = part.strip().strip("\"'")
            term = " ".join(term.split()[:5])   # max 5 words per term
            if term:
                terms.append(term)

        terms = [t for t in terms[:4] if t]     # max 4 terms, drop empties

        if not terms:
            terms = [" ".join(raw.split()[:4])]

        result = " OR ".join(terms)
        logger.debug("Disease query resolved: %r → %r", raw, result)
        return result

    # ── CGD table (gene → known conditions) ──────────────────────────────────

    @classmethod
    def _load_cgd_table(cls) -> tuple[dict[str, list[str]], dict[str, str]]:
        """
        Download and parse the NHGRI Clinical Genomic Database (CGD).
        Returns ({GENE_UPPER: [condition, ...]}, {GENE_UPPER: inheritance_string}).
        Called at most once per process. Returns empty dicts on any failure so
        callers degrade gracefully.
        """
        try:
            resp = requests.get(CGD_URL, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("CGD download failed (%s) — known-disease track disabled", e)
            return {}, {}

        try:
            with gzip.open(io.BytesIO(resp.content)) as f:
                text = f.read().decode("utf-8")
        except Exception as e:
            logger.warning("CGD decompression failed (%s) — known-disease track disabled", e)
            return {}, {}

        try:
            lines = text.splitlines()

            # Locate header row (first non-comment non-empty line)
            header_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("#GENE") or (not line.startswith("#") and line.strip()):
                    header_idx = i
                    break

            headers  = [h.strip() for h in lines[header_idx].lstrip("#").split("\t")]
            gene_col = next((i for i, h in enumerate(headers) if h.upper() == "GENE"), None)
            cond_col = next(
                (i for i, h in enumerate(headers) if h.upper() in {"CONDITION", "PHENOTYPE"}),
                None,
            )
            inh_col = next(
                (i for i, h in enumerate(headers) if h.upper() == "INHERITANCE"),
                None,
            )
            if gene_col is None or cond_col is None:
                logger.warning("CGD missing GENE/CONDITION column — known-disease track disabled")
                return {}, {}

            table:       dict[str, list[str]] = {}
            inheritance: dict[str, str]       = {}
            for line in lines[header_idx + 1:]:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) <= max(gene_col, cond_col):
                    continue
                gene      = fields[gene_col].strip().upper()
                condition = fields[cond_col].strip()
                if gene and condition:
                    table.setdefault(gene, []).append(condition)
                if gene and inh_col is not None and len(fields) > inh_col:
                    inh = fields[inh_col].strip()
                    if inh and gene not in inheritance:
                        inheritance[gene] = inh

        except Exception as e:
            logger.warning("CGD parsing failed (%s) — known-disease track disabled", e)
            return {}, {}

        logger.info("CGD table loaded: %d genes indexed (%d with inheritance)", len(table), len(inheritance))
        return table, inheritance

    @classmethod
    def _get_cgd_conditions(cls, gene: str) -> list[str]:
        """Return known CGD conditions for *gene*, loading the table on first call."""
        if cls._cgd_table is None:
            with cls._cgd_lock:
                if cls._cgd_table is None:   # double-checked locking
                    cls._cgd_table, cls._cgd_inheritance = cls._load_cgd_table()
        return cls._cgd_table.get(gene.upper(), [])

    @classmethod
    def get_cgd_inheritance(cls, gene: str) -> str:
        """Return the CGD-reported inheritance mode for *gene* (e.g. "AR", "AD"), or "" if unknown."""
        if cls._cgd_inheritance is None:
            with cls._cgd_lock:
                if cls._cgd_inheritance is None:   # double-checked locking
                    cls._cgd_table, cls._cgd_inheritance = cls._load_cgd_table()
        return cls._cgd_inheritance.get(gene.upper(), "")

    # ── step 0: gene + c./p. change → LitVar2 variant ID (autocomplete path) ──

    @staticmethod
    def _variant_tokens(hgvs: str) -> tuple[list[str], list[str]]:
        """All distinct c. and p. change tokens across every transcript of a
        combined HGVS field, e.g. 'GENE_X:NM_000000:exon4:c.100A>C:p.K34T|...'
        -> (['c.100A>C', ...], ['p.K34T', ...]). Synonymous p. tokens are kept
        (LitVar2 indexes some variants under them) but c. is always tried first."""
        if not hgvs or hgvs == "NA":
            return [], []
        cdna = list(dict.fromkeys(_CDNA_CHANGE_RE.findall(hgvs)))
        prot = list(dict.fromkeys(f"p.{m}" for m in _PROTEIN_CHANGE_RE.findall(hgvs)))
        return cdna, prot

    @staticmethod
    def _norm_change(token: str) -> str:
        t = (token or "").lower().replace("(", "").replace(")", "")
        t = re.sub(r"^[cp]\.", "", t)
        return re.sub(r"(\*|ter)$", "x", t)

    def _resolve_litvar_id(self, gene: str, hgvs: str, rsid: str | None) -> tuple[str, str] | None:
        """
        Resolve this variant to its LitVar2 variant ID by its own c./p. names
        (autocomplete 'GENE c.change', then 'GENE p.change'), independent of the
        input RS_ID column. LitVar2 IDs are rsID-based when dbSNP has one, but
        variants without an rsID have their own non-rs IDs — both work with the
        publications endpoint.

        The autocomplete top hit is accepted only when it is for this gene AND
        its own name/HGVS equals one of this variant's c./p. tokens (or its rsID
        equals the input RS_ID) — free-text autocomplete can otherwise silently
        return a different variant, whose empty literature would look identical
        to a genuinely unstudied variant.

        Returns (litvar_id, matched_token) or None.
        """
        cdna, prot = self._variant_tokens(hgvs)
        own = {self._norm_change(t) for t in cdna + prot}
        for token in cdna + prot:
            query = f"{gene} {token}"
            try:
                _ncbi_rate_limit()
                resp = requests.get(
                    LITVAR2_AUTOCOMPLETE_URL, params={"query": query}, timeout=self.timeout
                )
                resp.raise_for_status()
                results = resp.json()
            except Exception as e:
                logger.debug("LitVar2 autocomplete failed for %r: %s", query, e)
                continue
            for r in results or []:
                if gene.upper() not in {g.upper() for g in (r.get("gene") or [])}:
                    continue
                names = {self._norm_change(r.get("name", "")), self._norm_change(r.get("hgvs", ""))}
                same_rs = bool(rsid) and (r.get("rsid") or "").lower() == rsid.lower()
                if (names & own or same_rs) and r.get("_id"):
                    return r["_id"], token
        return None

    # ── step 1: LitVar2 → PMIDs (variant-ID path) ────────────────────────────

    def _fetch_pmids(self, litvar_id: str) -> list[str]:
        url = LITVAR2_URL.format(litvar_id=requests.utils.quote(litvar_id, safe=""))
        logger.debug("LitVar2 fetch: %s", url)
        try:
            _ncbi_rate_limit()
            api_params = {"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}
            for attempt in range(4):
                resp = requests.get(url, params=api_params, timeout=self.timeout)
                if resp.status_code == 429:
                    wait = 2.0 ** attempt
                    logger.warning("LitVar2 rate limit (429) — backing off %.0fs (attempt %d/4)", wait, attempt + 1)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            data = resp.json()
        except Exception as e:
            raise ToolFetchError(f"LitVar2 API request failed for {litvar_id}: {e}") from e

        # API returns {"pmids": [...], "pmcids": [...], "pmids_count": N}
        raw_pmids = data.get("pmids", [])
        pmids = [str(p) for p in raw_pmids if str(p).isdigit()][:self.max_pmids]

        logger.debug("LitVar2 returned %d PMIDs for %s", len(pmids), litvar_id)
        return pmids

    # ── step 2: PMIDs → titles ────────────────────────────────────────────────

    def _fetch_titles(self, pmids: list[str]) -> dict[str, str]:
        """Returns {pmid: title}."""
        if not pmids:
            return {}
        try:
            data = _ncbi_get(
                "esummary.fcgi",
                {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
                self.timeout,
            ).json()
        except Exception as e:
            raise ToolFetchError(f"PubMed esummary request failed: {e}") from e

        result = data.get("result", {})
        try:
            return {
                pmid: result[pmid].get("title", "No title")
                for pmid in pmids
                if pmid in result
            }
        except Exception as e:
            raise ToolParseError(f"Failed to parse PubMed titles: {e}") from e

    # ── step 3: SLM title filter ──────────────────────────────────────────────

    def _score_titles_batch(
        self,
        batch: list[tuple[str, str]],
        question: str,
        context: ToolContext,
    ) -> dict[str, int]:
        """
        Ask the SLM to score every title in *batch* (0-10) against *question*.
        Small batch size keeps the model attending to each title individually
        instead of skimming a long list. Returns {pmid: score}; a pmid missing
        from the parsed response (bad JSON, model dropped it) is simply absent —
        callers must treat that as "unscored", not "irrelevant".
        """
        numbered = "\n".join(
            f"{i+1}. [PMID:{pmid}] {title}" for i, (pmid, title) in enumerate(batch)
        )
        valid_pmids = {pmid for pmid, _ in batch}

        system = (
            "You are a biomedical literature relevance scorer. "
            "For EACH paper title below, output a relevance score from 0 (irrelevant) "
            "to 10 (directly relevant) to the research question. Score every title — "
            "do not skip any, do not select a subset. "
            "Output ONLY a JSON object mapping each PMID string to its integer score, "
            "e.g. {\"12345678\": 8, \"87654321\": 2}. No explanation, no markdown, no extra text."
        )
        user = f"Research question: {question}\n\nTitles:\n{numbered}"

        raw = context.llm.generate(system=system, user=user, max_tokens=200).strip()

        try:
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return {
                    str(pmid): int(score)
                    for pmid, score in parsed.items()
                    if str(pmid) in valid_pmids and str(int(score)) == str(score).strip()
                }
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("Could not parse batch relevance-scoring JSON (%s) — batch unscored", e)

        return {}

    def _select_relevant_pmids(
        self,
        titles: dict[str, str],
        question: str,
        context: ToolContext,
    ) -> list[str]:
        """
        Score every title 0-10 against the question in small batches
        (SELECT_BATCH_SIZE at a time), then keep the top_n highest-scoring PMIDs.
        """
        items = list(titles.items())

        scores: dict[str, int] = {}
        for i in range(0, len(items), SELECT_BATCH_SIZE):
            batch = items[i : i + SELECT_BATCH_SIZE]
            scores.update(self._score_titles_batch(batch, question, context))

        if not scores:
            logger.warning("No titles could be scored — falling back to first %d", self.top_n)
            return list(titles.keys())[: self.top_n]

        # Unscored PMIDs (dropped by the model in a malformed batch response) rank
        # last rather than being silently excluded from consideration.
        ranked = sorted(
            items, key=lambda kv: scores.get(kv[0], -1), reverse=True
        )
        selected = [pmid for pmid, _ in ranked[: self.top_n] if scores.get(pmid, -1) > 0]

        if selected:
            logger.debug("SLM scored selection: %d/%d PMIDs above 0", len(selected), len(items))
            return selected

        # Every title scored 0 (or unscored) — fall back rather than return nothing.
        return list(titles.keys())[: self.top_n]

    # ── step 4: PMIDs → abstracts ─────────────────────────────────────────────

    def _fetch_abstracts(self, pmids: list[str]) -> dict[str, dict]:
        """Returns {pmid: {"title": ..., "abstract": ..., "url": ..., "year": ...}}."""
        if not pmids:
            return {}

        try:
            xml_text = _ncbi_get(
                "efetch.fcgi",
                {"db": "pubmed", "id": ",".join(pmids), "rettype": "xml", "retmode": "xml"},
                self.timeout,
            ).text
            root = ET.fromstring(xml_text)
        except Exception as e:
            raise ToolFetchError(f"PubMed efetch request failed: {e}") from e

        try:
            results = {}
            for article in root.iter("PubmedArticle"):
                pmid_el = article.find(".//PMID")
                if pmid_el is None:
                    continue
                pmid = pmid_el.text.strip()

                title_el = article.find(".//ArticleTitle")
                title    = _clean_xml_text(ET.tostring(title_el, encoding="unicode")) if title_el is not None else "No title"

                abstract_parts = [
                    _clean_xml_text(ET.tostring(a, encoding="unicode"))
                    for a in article.iter("AbstractText")
                ]
                abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available."

                year_el = article.find(".//PubDate/Year")
                if year_el is not None and year_el.text:
                    year = year_el.text.strip()
                else:
                    medline_el = article.find(".//PubDate/MedlineDate")
                    match = re.search(r"\d{4}", medline_el.text) if medline_el is not None and medline_el.text else None
                    year = match.group(0) if match else "n.d."

                results[pmid] = {
                    "title":    title,
                    "abstract": abstract[: self.max_chars],
                    "url":      f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "year":     year,
                }

            return results
        except Exception as e:
            raise ToolParseError(f"Failed to parse PubMed abstracts: {e}") from e

    # ── step 5: SLM summary ───────────────────────────────────────────────────

    def _summarise(
        self,
        papers: dict[str, dict],
        question: str,
        identifier: str,
        context: ToolContext,
        require_functional: bool = False,
    ) -> str:
        """
        Ask the SLM to produce a prose summary with inline PMID citations.

        require_functional=True is used for tracks where every abstract is already
        known to directly mention the specific variant (e.g. LitVar2 rsID hits) — in
        that case any functional/structural/enzymatic evidence for the variant itself
        must be reported even if it doesn't bear on the patient's phenotype match,
        since it's the kind of mechanistic evidence ACMG functional-evidence criteria
        (e.g. PS3/BS3) are built on and must not be dropped for narrative brevity.
        Same flag also forces extraction of case-count/proband-enumeration evidence
        (PS4 grounding) — the summarizer otherwise tends to surface qualitative
        mechanism statements (e.g. "a known mutational hotspot") over the actual
        headcount of affected individuals reported with this variant, even when both
        are present in the same abstract.
        """

        corpus = "\n\n".join(
            f"[PMID:{pmid}, {p.get('year', 'n.d.')}] {p['title']}\n{p['abstract']}"
            for pmid, p in papers.items()
        )

        functional_clause = (
            " Every abstract here was retrieved via a direct search for this specific "
            f"variant ({identifier}), not just its gene — if ANY abstract reports "
            "functional, structural, or enzymatic evidence for this exact variant "
            "(e.g. in vitro/in vivo assays, enzyme activity, protein structure or "
            "stability effects, splicing effects), you MUST report that finding "
            "explicitly with its citation, even if it does not relate to the patient's "
            "phenotype. Do not omit direct variant-level functional evidence for the "
            "sake of brevity — extend the paragraph if needed. "
            "Separately, and just as mandatorily: scan every abstract for case-count/"
            "proband evidence for this exact variant — any explicit number of affected "
            "individuals, patients, probands, or families reported carrying it (e.g. "
            "'identified in 12 unrelated probands', 'observed in 3 of 40 patients with "
            "this phenotype', 'reported in 2 families'). This is the evidence that "
            "grounds PS4 (prevalence in affected individuals vs. controls) and is easy "
            "to miss when an abstract also contains a qualitative mechanism statement "
            "(e.g. a hotspot or domain claim) — report the actual headcount explicitly "
            "with its citation whenever a number is stated, even in a single clause, "
            "rather than only reporting the qualitative claim. If no abstract states an "
            "explicit affected-individual count for this variant, say so explicitly "
            "rather than leaving it unaddressed."
            if require_functional else ""
        )

        system = (
            "You are a biomedical evidence synthesiser. "
            "Write a short paragraph (3-5 sentences, longer if needed per instructions below) "
            "summarising findings from the abstracts that are relevant to the research question. "
            "When abstracts conflict, prefer the more recent publication year and note the "
            "disagreement. "
            "Cite sources inline using (PMID:XXXXXXXX) immediately after the relevant claim. "
            "If no abstract contains relevant evidence, write one sentence saying so. "
            "Do NOT use bullet points, headers, or raw URLs. "
            "Do NOT mention abstracts or papers explicitly, just report the findings naturally. "
            "Begin directly with the summary. "
            "CRITICAL — do not manufacture relevance: report only what the abstracts EXPLICITLY "
            "state. If the research question asks about a specific phenotype and the abstracts "
            "describe a DIFFERENT phenotype (e.g. abstracts describe intellectual "
            "disability/epilepsy but the research question asks about visual impairment), say "
            "plainly that no evidence links this gene/variant to that specific phenotype — do "
            "NOT bridge the gap yourself with inferential language ('may indicate', 'suggests "
            "potential for', 'broader phenotype', 'could plausibly include') to imply a "
            "connection that no abstract actually makes. A hedge you invent reads as evidence "
            "to whoever uses this summary next and can produce a wrong diagnosis — when the "
            "phenotype match isn't explicitly stated, say there is none, full stop."
            f"{functional_clause}"
        )
        user = (
            f"Research question: {question}\n"
            f"Variant/gene: {identifier}\n\n"
            f"Abstracts:\n{corpus}"
        )

        return context.llm.generate(system=system, user=user, max_tokens=500).strip()

    # ── primary: gene-level PubMed search ────────────────────────────────────

    def _esearch_gene(
        self, gene: str, disease_query: str | None = None, sort: str = "relevance"
    ) -> tuple[list[str], int, str]:
        """
        Query PubMed esearch for the gene, optionally filtered by disease term.

        Gene match uses ({gene}[tiab] OR {gene}[Gene]) so papers that mention the gene
        in title/abstract are included even when the gene is not formally MeSH-indexed.

        When disease_query is None the search is gene-only — used as a fallback to
        distinguish "gene has no literature at all" from "gene-disease co-occurrence
        not found."

        Primary call uses `sort` (default relevance). If total results exceed max_pmids
        AND the primary sort wasn't already pub_date, a second call sorted by
        publication date adds recent papers not in the primary window.

        Returns:
            pmids             — deduplicated list, primary-sort-ranked first then recency
            total_count       — total PubMed hits (may exceed len(pmids))
            query_translation — PubMed's MeSH expansion of the submitted term
        """
        gene_clause = f"({gene}[tiab] OR {gene}[Gene])"
        term = f"{gene_clause} AND ({disease_query})" if disease_query else gene_clause

        # ── primary call ────────────────────────────────────────────────────────
        try:
            data = _ncbi_get(
                "esearch.fcgi",
                {"db": "pubmed", "term": term, "retmax": self.max_pmids,
                 "retmode": "json", "sort": sort},
                self.timeout,
            ).json()
        except Exception as e:
            raise ToolFetchError(f"PubMed esearch failed for gene={gene}: {e}") from e

        try:
            result       = data["esearchresult"]
            pmids        = [str(p) for p in result["idlist"] if str(p).isdigit()]
            total_count  = int(result.get("count", len(pmids)))
            query_translation = result.get("querytranslation", term)
        except (KeyError, TypeError, ValueError) as e:
            raise ToolParseError(
                f"Failed to parse PubMed esearch response for {gene}: {e}"
            ) from e

        logger.debug(
            "PubMed esearch (%s): %d/%d PMIDs for gene=%s",
            sort, len(pmids), total_count, gene,
        )

        # ── conditional second call: recency sort ─────────────────────────────
        # Only fires when total results exceed what the first call retrieved
        # AND the primary call wasn't already sorted by pub_date (no point
        # repeating the same sort twice).
        if total_count > self.max_pmids and sort != "pub_date":
            try:
                data2 = _ncbi_get(
                    "esearch.fcgi",
                    {"db": "pubmed", "term": term, "retmax": self.max_pmids // 2,
                     "retmode": "json", "sort": "pub_date"},
                    self.timeout,
                ).json()
                recent = [
                    str(p) for p in data2["esearchresult"]["idlist"]
                    if str(p).isdigit() and str(p) not in set(pmids)
                ]
                pmids = pmids + recent
                logger.debug(
                    "PubMed esearch (pub_date): added %d recent PMIDs for gene=%s",
                    len(recent), gene,
                )
            except Exception as e:
                logger.warning(
                    "PubMed esearch recency call failed for gene=%s (%s) — using relevance only",
                    gene, e,
                )

        return pmids, total_count, query_translation

    def _gene_search(self, gene: str, context: ToolContext) -> str:
        """
        Primary gene-level search via PubMed esearch.

        Deliberately PHENOTYPE-AGNOSTIC, same principle as run_condition_inventory()
        below: the esearch query and the relevance-selection question are built from
        generic disease/syndrome/inheritance vocabulary only — never from
        context.patient_phenotype or self._disease_query (both patient-derived).
        This track's job is to retrieve and describe the FULL set of conditions this
        gene is known to cause; comparing that against the patient's own phenotype
        (CLUSTER_PHENOTYPE) is Stage 1 reasoning's job (prompts/reasoning.txt), which
        sees both this evidence block and the patient phenotype together. Anchoring
        retrieval/selection/summarization to the patient's phenotype here pre-judges
        that comparison before Stage 1 ever runs, and — since the disease-query
        resolution and relevance-scoring are themselves LLM calls — makes the
        evidence pool (and therefore every downstream stage, including MOI
        classification) sensitive to run-to-run LLM sampling noise on top of the
        gene's actual literature. self._disease_query remains used by the rsID-level
        Track 3 search (_variant_search) only, where "does this paper discuss THIS
        variant in the context of the patient's condition" is a narrower, legitimately
        patient-scoped question.

        Returns a positive evidence block when disease-relevant papers are found,
        or an explicit "not linked" message when none are — providing a clear
        negative signal for downstream reasoning and triage stages.

        Output header always logs the exact resolved query used (not just on the
        empty-hits fallback path) so downstream stages — and a human reading the
        report — can audit what was searched without re-deriving it.
        """
        # Generic disease/syndrome/inheritance vocabulary only — see docstring above
        # for why context.patient_phenotype/self._disease_query are never used here.
        # Inheritance-mode vocabulary is included so papers establishing how the
        # gene's disease is inherited (needed downstream for the AR/AD/XLR gate and
        # phase-check logic) are preferentially retrieved within the max_pmids-capped
        # pool, not just papers matching generic disease wording.
        #
        # Sort by relevance (the _esearch_gene default), not pub_date: with this
        # OR-heavy query almost any paper mentioning the gene near a disease/
        # inheritance word matches, so a pub_date-primary sort returns the N most
        # recent papers on the gene regardless of topic — for a gene whose
        # founding gene-disease paper is old (e.g. a gene-disease link
        # established two decades ago) that paper is pushed out of the max_pmids window entirely
        # and every candidate the SLM sees is an unrelated recent publication.
        # _esearch_gene already adds a supplemental pub_date-sorted call when
        # total_count exceeds max_pmids, so recent literature is still covered.
        inheritance_terms = 'recessive OR dominant OR "x-linked" OR "de novo" OR biallelic'
        disease_query = f'disease OR syndrome OR {inheritance_terms}'
        pmids, total_count, _ = self._esearch_gene(gene, disease_query)

        # Header line shared by all output branches — always states the exact query
        # used, unconditionally (see docstring: auditability, not just on fallback).
        def _header(extra: str = "") -> str:
            pool_note = (
                f"{len(pmids)} retrieved (pub_date-sorted)"
                if total_count > self.max_pmids
                else f"{total_count} total"
            )
            base = (
                f"PubMed gene-disease search for {gene}\n"
                f"[query: {disease_query} | {pool_note}{extra}]"
            )
            return base

        if not pmids:
            # Retry with gene alone to distinguish "no literature at all" from
            # "literature exists but none of it uses generic disease/inheritance
            # vocabulary" (rare — e.g. a gene whose only papers are pure
            # population-genetics/mechanism studies with no clinical framing yet).
            pmids_gene_only, total_count_gene_only, _ = self._esearch_gene(gene, None)
            if not pmids_gene_only:
                return (
                    f"{_header()}\n"
                    f"NO DISEASE LINK — PubMed contains no publications mentioning {gene} "
                    f"at all."
                )
            # Gene has literature but none matched the generic disease/inheritance
            # vocabulary — proceed through normal pipeline using the gene-only pool,
            # with a header that signals the vocabulary filter was dropped.
            pmids = pmids_gene_only
            total_count = total_count_gene_only
            disease_query_used = disease_query

            def _header(extra: str = "") -> str:  # noqa: F811 — shadow outer _header
                pool_note = (
                    f"{len(pmids)} retrieved (relevance + recency)"
                    if total_count > self.max_pmids
                    else f"{total_count} total"
                )
                return (
                    f"PubMed gene-only evidence for {gene} "
                    f"(generic disease/inheritance vocabulary '{disease_query_used}' "
                    f"yielded no hits — gene-only search)\n"
                    f"[{pool_note}{extra}]"
                )

        titles = self._fetch_titles(pmids)
        if not titles:
            return (
                f"{_header()}\n"
                f"Gene-disease association could not be assessed (title fetch failed)."
            )

        selection_question = (
            f"Does this paper name or describe a specific disease, syndrome, or "
            f"clinical condition caused by or associated with {gene} — any such "
            f"condition, not limited to any particular patient's presentation? "
            f"Score high for a paper establishing or discussing a real gene-disease "
            f"relationship (mechanism, cohort, case report, inheritance pattern); "
            f"score low for population-genetics-only, unrelated-pathway, or "
            f"GWAS-risk-allele papers with no named clinical entity attached."
        )
        selected_pmids = self._select_relevant_pmids(titles, selection_question, context)
        logger.info("Gene search %s: selected %d/%d papers", gene, len(selected_pmids), len(titles))

        if not selected_pmids:
            return (
                f"{_header(f', {len(titles)} screened, 0 relevant')}\n"
                f"NO DISEASE LINK — No publications naming a specific disease/syndrome "
                f"for {gene} were found. This gene does not appear to have an "
                f"established clinical condition in the literature retrieved."
            )

        papers = self._fetch_abstracts(selected_pmids)
        if not papers:
            return (
                f"{_header(f', abstract fetch failed after selecting {len(selected_pmids)}')}\n"
                f"Gene-disease association could not be assessed."
            )

        summary_question = (
            f"Summarize what disease(s), syndrome(s), or clinical condition(s) these "
            f"abstracts attribute to {gene} — every distinct one reported, not just "
            f"the most prominent — including phenotypic features, inheritance "
            f"pattern, and molecular mechanism where stated. Describe the "
            f"gene-disease relationship(s) as reported in the literature; do not "
            f"compare them to, or judge their fit against, any specific patient case."
        )
        summary = self._summarise(papers, summary_question, gene, context)

        source_lines = "\n".join(
            f"  - [PMID:{pmid}, {p.get('year', 'n.d.')}] {p['title']}  {p['url']}"
            for pmid, p in papers.items()
        )
        return (
            f"PubMed gene-disease evidence for {gene}\n"
            f"[query: {disease_query} | {len(titles)} screened, {len(papers)} selected]\n\n"
            f"{summary}\n\n"
            f"Sources:\n{source_lines}"
        )

    # ── supplemental: known-disease PubMed search (track 2) ──────────────────

    def _omim_gene_search(self, gene: str, context: ToolContext) -> str | None:
        """
        Supplemental gene-level search anchored on the gene's known disease association
        rather than the patient's described phenotype. Bridges the terminology gap between
        how the patient's condition is described and how OMIM/CGD name the disease.

        Disease term source priority:
          1. OMIM_phenotype from the variant dict (already curated for this variant)
          2. NHGRI CGD conditions for the gene (class-level cache, loaded once per process)

        Relevance filtering uses the known condition as the question so the SLM selects
        papers about that specific disease. Summarisation is framed on the patient phenotype
        so the output remains clinically grounded. Returns None when no disease term is
        available or no relevant papers are found.
        """
        omim_phenotype = context.field("OMIM_phenotype")
        if omim_phenotype != "NA":
            conditions = [omim_phenotype]
            source     = "OMIM"
        else:
            conditions = self._get_cgd_conditions(gene)
            source     = "CGD"

        if not conditions:
            return None

        # Strip trailing OMIM subtype numbers ("Microphthalmia, isolated 8" → "Microphthalmia, isolated")
        # before quoting — PubMed cannot phrase-match these numbered subtypes and breaks them apart noisily.
        def _strip_subtype(s: str) -> str:
            return re.sub(r",?\s*\d+\s*$", "", s).strip()

        # Build quoted-phrase OR query; cap to keep PubMed query focused
        capped        = conditions[:CGD_MAX_CONDITIONS]
        disease_query = " OR ".join(f'"{_strip_subtype(c)}"' for c in capped)

        try:
            pmids, _, _ = self._esearch_gene(gene, disease_query)
        except PipelineError as e:
            logger.warning("Known-disease search failed for gene=%s (%s)", gene, e)
            return None

        if not pmids:
            return None

        titles = self._fetch_titles(pmids)
        if not titles:
            return None

        # Relevance filter: anchor on the known condition, not the patient phenotype
        question       = f"{gene} and {', '.join(capped)}"
        selected_pmids = self._select_relevant_pmids(titles, question, context)
        if not selected_pmids:
            return None

        papers = self._fetch_abstracts(selected_pmids)
        if not papers:
            return None

        # Summarise in the context of the patient phenotype so output is clinically framed
        summary = self._summarise(papers, context.patient_phenotype, gene, context)

        source_lines = "\n".join(
            f"  - [PMID:{pmid}, {p.get('year', 'n.d.')}] {p['title']}  {p['url']}"
            for pmid, p in papers.items()
        )
        return (
            f"PubMed gene-disease evidence for {gene} (known condition, source: {source})\n"
            f"[{len(titles)} screened, {len(papers)} selected | "
            f"conditions: {'; '.join(capped)}]\n\n"
            f"{summary}\n\n"
            f"Sources:\n{source_lines}"
        )

    # ── supplemental: rsID-level LitVar2 search ──────────────────────────────

    # PS3 from LitVar2: only the variant-level (rsID) track, only a summary
    # sentence that (a) reports functional/experimental work, (b) is not
    # negated/in-silico, (c) names THIS variant (rsID, its own c./p. notation,
    # or "this/the specific/exact variant"), and (d) cites a PMID among the
    # papers this track itself retrieved. Gene-level tracks never count —
    # a paper about the gene is not functional evidence for the variant.
    _NEGATION_MARKERS = (
        " no ", "none ", "neither", " nor ", "without", "lack", "absent",
        " not ", "n't ",
    )
    _THIS_VARIANT_PHRASES = (
        "this variant", "this specific variant", "this exact variant",
        "the specific variant", "the exact variant", "the variant itself",
    )

    @classmethod
    def _variant_functional_pmids(cls, summary: str, paper_pmids: set[str],
                                  rsid: str, hgvs: str) -> list[str]:
        own = {t.lower() for t in re.findall(r"rs\d+", rsid or "")} | \
              {t.lower() for t in re.findall(r"c\.[^\s:;|()]+", hgvs or "")}
        for ref, pos, alt in PROTEIN_CHANGE_RE.findall(hgvs or ""):
            own |= {f"p.{ref}{pos}{alt}".lower(), f"p.{_to_aa3(ref)}{pos}{_to_aa3(alt)}".lower()}
        out: list[str] = []
        for sent in re.split(r"(?<=[.!?])\s+", summary):
            low = f" {sent.lower()} "
            if not any(m in low for m in _FUNCTIONAL_MARKERS):
                continue
            if any(x in low for x in _COMMENT_EXCLUDE_MARKERS + cls._NEGATION_MARKERS):
                continue
            if not (any(p in low for p in cls._THIS_VARIANT_PHRASES) or any(t in low for t in own)):
                continue
            for pmid in re.findall(r"PMID:?\s*(\d{6,9})", sent):
                if pmid in paper_pmids and pmid not in out:
                    out.append(pmid)
        return out


    def _variant_search(self, litvar_id: str, label: str, context: ToolContext) -> str | None:
        """
        Supplemental variant-specific search via the LitVar2 variant publications
        endpoint. `label` is the variant's own c./p. name (plus rsID when known) —
        it is what the summariser matches abstracts against, since papers name
        variants by HGVS far more often than by rsID.
        Returns an evidence block, or None if no relevant variant-level literature found.
        Output header identifies LitVar2 as source so it is distinguished from the
        gene-level PubMed block.
        """
        pmids = self._fetch_pmids(litvar_id)
        if not pmids:
            return None

        titles = self._fetch_titles(pmids)
        if not titles:
            return None

        selected_pmids = self._select_relevant_pmids(titles, self._disease_query, context)
        logger.info("LitVar2 %s: selected %d/%d papers", label, len(selected_pmids), len(titles))

        if not selected_pmids:
            return None

        papers = self._fetch_abstracts(selected_pmids)
        if not papers:
            return None

        summary = self._summarise(
            papers, context.patient_phenotype, label, context, require_functional=True
        )

        source_lines = "\n".join(
            f"  - [PMID:{pmid}, {p.get('year', 'n.d.')}] {p['title']}  {p['url']}"
            for pmid, p in papers.items()
        )
        func_pmids = self._variant_functional_pmids(
            summary, {str(k) for k in papers}, label, context.field("HGVS")
        )
        func_tag = (
            "\n[variant-level functional evidence reported by LitVar2, citing "
            + ", ".join(f"PMID:{p}" for p in func_pmids[:5]) + "]"
            if func_pmids else ""
        )
        return (
            f"LitVar2 variant-specific evidence for {label}\n"
            f"[Source: LitVar2 variant publications endpoint — "
            f"{len(titles)} records screened, {len(papers)} selected]{func_tag}\n\n"
            f"{summary}\n\n"
            f"Sources:\n{source_lines}"
        )

    # ── supplemental: phenotype-agnostic condition inventory (for PHENOTYPE tag) ──

    def run_condition_inventory(self, gene: str, context: ToolContext) -> str | None:
        """
        Gene-level PubMed search for the PHENOTYPE tag (see prompts/
        gene_phenotype_extraction.txt) — deliberately anchored on GENERIC
        disease/inheritance vocabulary only ("disease", "syndrome",
        "biallelic", "dominant", "recessive", "X-linked"), never on
        context.patient_phenotype or self._disease_query (both patient-
        derived). Both the esearch query AND the relevance-selection question
        below are built from this fixed generic vocabulary alone.

        Real observed failure this fixes: _gene_search (track 1) and
        _omim_gene_search (track 2) both ultimately select/frame relevance
        against the patient's own phenotype (self._disease_query, or the
        gene's OMIM/CGD-known condition when a patient-independent one
        happens to be catalogued there) — for a gene with a real, well-
        documented condition that is neither the patient's phenotype nor
        catalogued in OMIM/CGD as a distinct "known condition" entry (e.g. a
        common-variant/GWAS-style complex-trait association rather than a
        curated Mendelian entry), that condition is invisible to both
        existing tracks and never reaches ANY downstream stage — including
        the isolated gene-phenotype-extraction call, however well-worded its
        prompt, since a call can only report conditions present in what it
        was given. This track exists to retrieve those conditions in the
        first place, independent of what the patient has or what OMIM/CGD
        already curated for this gene.

        Returns a positive evidence block when condition-bearing papers are
        found, or None if the gene has no such literature at all (never an
        error string — a clean negative is a legitimate outcome here, unlike
        _gene_search's NO DISEASE LINK message, since this track is
        supplemental input to a single tag, not a triage signal).
        """
        generic_query = (
            'disease OR syndrome OR biallelic OR dominant OR recessive OR "x-linked"'
        )
        try:
            pmids, total_count, _ = self._esearch_gene(gene, generic_query)
        except PipelineError as e:
            logger.warning("Condition-inventory search failed for gene=%s (%s)", gene, e)
            return None

        if not pmids:
            return None

        titles = self._fetch_titles(pmids)
        if not titles:
            return None

        question = (
            f"Does this paper name or describe a specific disease, syndrome, or "
            f"clinical condition caused by {gene}? Score high for ANY named "
            f"condition (regardless of what it is or how common/rare), low for "
            f"papers that are purely population-genetics, mechanism-only, or "
            f"risk-allele statistics with no named clinical entity attached."
        )
        selected_pmids = self._select_relevant_pmids(titles, question, context)
        logger.info(
            "Condition-inventory search %s: selected %d/%d papers",
            gene, len(selected_pmids), len(titles),
        )
        if not selected_pmids:
            return None

        papers = self._fetch_abstracts(selected_pmids)
        if not papers:
            return None

        inventory_question = (
            f"List every distinct disease, condition, or syndrome name these "
            f"abstracts attribute to {gene} — a complete inventory, not "
            f"filtered to any particular phenotype. Name each one explicitly; "
            f"do not summarize them into a single narrative."
        )
        summary = self._summarise(papers, inventory_question, gene, context)

        source_lines = "\n".join(
            f"  - [PMID:{pmid}, {p.get('year', 'n.d.')}] {p['title']}  {p['url']}"
            for pmid, p in papers.items()
        )
        return (
            f"PubMed condition inventory for {gene} (phenotype-agnostic — "
            f"generic disease/inheritance query, not filtered by patient phenotype)\n"
            f"[{len(titles)} screened, {len(papers)} selected]\n\n"
            f"{summary}\n\n"
            f"Sources:\n{source_lines}"
        )

    # ── public entry point ────────────────────────────────────────────────────

    def run(self, variant: dict, context: ToolContext) -> str | None:
        """
        Fetch literature evidence using a gene-first three-track strategy.

        1. Gene + generic disease/inheritance vocabulary (primary, phenotype-agnostic):
           always runs when Gene is present. Retrieves and describes the full set of
           conditions the gene causes per the literature — comparing that against the
           patient's own phenotype (CLUSTER_PHENOTYPE) is Stage 1 reasoning's job, not
           this tool's; see _gene_search's docstring.
        2. Gene + known disease term (supplemental): runs when OMIM_phenotype is not NA
           or CGD has an entry for the gene. Bridges phenotype terminology gaps.
        3. Variant-level (supplemental): LitVar2 rsID endpoint when RS_ID is valid —
           the one track still scoped to the patient's own phenotype (self._disease_query),
           since "does this paper discuss THIS variant in the patient's condition" is a
           narrower, legitimately patient-anchored question.

        All tracks that yield evidence are combined, separated by dividers. Each block
        header embeds source and query for downstream traceability.
        """
        rsid = context.field("RS_ID")
        gene = context.field("Gene")
        hgvs = context.field("HGVS")

        if self._disease_query is None:
            with self._disease_query_lock:
                if self._disease_query is None:   # double-checked locking
                    self._disease_query = self._resolve_disease_query(context)

        rsid_ok = rsid != "NA" and re.match(r"^rs\d+$", rsid, re.IGNORECASE) is not None
        gene_ok = gene != "NA" and gene.strip() != ""

        # Variant-level literature is looked up by the variant's own c./p. names
        # (RS_ID is often NA in clinical exports, and variants without an rsID
        # have their own LitVar2 IDs); the input rsID is only a last resort.
        litvar = self._resolve_litvar_id(gene, hgvs, rsid if rsid_ok else None) if gene_ok else None
        if litvar:
            litvar_id, matched = litvar
            label = f"{gene} {matched}" + (f" ({rsid})" if rsid_ok else "")
        elif rsid_ok:
            litvar_id, label = f"litvar@{rsid}##", rsid
        else:
            litvar_id = label = None

        parts = []

        # 1. Gene + patient phenotype (primary)
        if gene_ok:
            logger.info("LitVar2SummaryTool: gene+phenotype search for gene=%s", gene)
            parts.append(self._gene_search(gene, context))

        # 2. Gene + known disease association (supplemental)
        if gene_ok:
            logger.info("LitVar2SummaryTool: known-disease search for gene=%s", gene)
            try:
                omim_block = self._omim_gene_search(gene, context)
                if omim_block:
                    parts.append(omim_block)
            except PipelineError as e:
                logger.warning(
                    "Known-disease search failed for gene=%s (%s) — primary results preserved",
                    gene, e,
                )

        # 3. Variant-level search (supplemental variant-specific evidence)
        if litvar_id:
            logger.info("LitVar2SummaryTool: variant-level LitVar2 search for %s (%s)", label, litvar_id)
            try:
                variant_result = self._variant_search(litvar_id, label, context)
                if variant_result:
                    parts.append(variant_result)
            except PipelineError as e:
                logger.warning(
                    "LitVar2 variant search failed for %s (%s) — gene-level results preserved",
                    label, e,
                )

        return "\n\n---\n\n".join(parts) if parts else None


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    import sys
    if len(sys.argv) != 2:
        sys.exit("usage: python -m pipeline.tools.litvar2 <rsid>")
    rsid = sys.argv[1]

    # Step 1: raw API response
    import requests
    url = f"https://www.ncbi.nlm.nih.gov/research/litvar2-api/variant/get/litvar@{rsid}%23%23/publications"
    resp = requests.get(url, timeout=15)
    data = resp.json()
    print("=== RAW RESPONSE (first 2 items) ===")
    items = data if isinstance(data, list) else data.get("publications", [])
    for item in items[:2]:
        print(item)

    # Step 2: fetch steps (no SLM — skips steps 3 and 5)
    tool = LitVar2SummaryTool()
    pmids = tool._fetch_pmids(f"litvar@{rsid}##")
    print(f"\n=== PMIDs ({len(pmids)}) ===")
    print(pmids)

    titles = tool._fetch_titles(pmids)
    print(f"\n=== TITLES ===")
    for pmid, title in titles.items():
        print(f"  {pmid}: {title}")

    papers = tool._fetch_abstracts(list(titles.keys())[:3])
    print(f"\n=== ABSTRACTS (first 3) ===")
    for pmid, p in papers.items():
        print(f"\n  [{pmid}] {p['title']}\n  {p['abstract'][:200]}")

"""
pipeline/tools/ncbi.py — NCBI fetch tool for the ReAct agent.

Contains:
  - NCBIFetchTool — routes PubMed / PMC / ClinVar URLs to the right E-utilities call

Shared helpers (_ncbi_get, _clean_xml_text) are imported from websearch.py.

Note: NCBIFetchTool is a sub-tool consumed by WebSearchAgentTool's internal ReActAgent.
It is NOT a pipeline-level tool. It overrides run(query: str) for ReActAgent compatibility.
"""

import re
import logging
import xml.etree.ElementTree as ET

from pipeline.tools.base import NetworkTool
from pipeline.tools.websearch import (
    _ncbi_get,
    _clean_xml_text,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_CHARS,
)
from pipeline.core.acmg_sf import is_pathogenic_clinvar
from pipeline.tools.clinvar_gene_stats import comment_functional_pmids

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TOOL 2 — NCBI FETCH
# ─────────────────────────────────────────────

class NCBIFetchTool(NetworkTool):
    """
    Fetches structured data from any NCBI resource using the E-utilities API.
    Covers PubMed abstracts, PMC full-text articles, and ClinVar variant records.
    Use this for ANY URL containing ncbi.nlm.nih.gov.
    Do NOT use web_fetch for NCBI URLs.
    """
    name        = "ncbi_fetch"
    description = (
        "Fetches clean, structured data from NCBI resources via the E-utilities API. "
        "Handles: PubMed abstracts (pubmed.ncbi.nlm.nih.gov), "
        "PMC full-text articles (pmc.ncbi.nlm.nih.gov), "
        "and ClinVar variant records (ncbi.nlm.nih.gov/clinvar). "
        "Always prefer this tool over web_fetch for any ncbi.nlm.nih.gov URL."
    )
    input_system = (
        "You are an NCBI URL extractor. "
        "Extract the single most relevant NCBI URL from the context. "
        "Only accept URLs from ncbi.nlm.nih.gov subdomains. "
        "NEVER include tool names, function calls, parentheses, or code syntax. "
        "Output ONLY the bare URL string. No explanation, no punctuation."
    )
    input_template = "Extract the NCBI URL from: {user_input}"

    # URL pattern matchers
    _PUBMED_RE  = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")
    _PMC_RE     = re.compile(r"pmc\.ncbi\.nlm\.nih\.gov/articles/PMC(\d+)")
    _CLINVAR_RE = re.compile(r"ncbi\.nlm\.nih\.gov/clinvar/variation/(\d+)")
    _NCBI_RE    = re.compile(r"ncbi\.nlm\.nih\.gov")

    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS, timeout: int = DEFAULT_TIMEOUT):
        self.max_chars = max_chars
        self.timeout   = timeout

    # ── PubMed abstract ────────────────────────────────────────────────────────

    def _fetch_pubmed(self, pmid: str) -> str:
        logger.debug("Fetching PubMed PMID=%s", pmid)

        # Metadata
        try:
            meta = _ncbi_get(
                "esummary.fcgi",
                {"db": "pubmed", "id": pmid, "retmode": "json"},
                self.timeout,
            ).json()["result"].get(pmid, {})
        except (Exception, KeyError, ValueError) as e:
            return f"PubMed metadata error for PMID {pmid}: {e}"

        title   = meta.get("title", "No title")
        journal = meta.get("fulljournalname", meta.get("source", "Unknown journal"))
        pubdate = meta.get("pubdate", "Unknown date")
        raw_authors = meta.get("authors", [])
        authors = ", ".join(a["name"] for a in raw_authors[:5] if "name" in a)
        if len(raw_authors) > 5:
            authors += " et al."

        # Full abstract
        try:
            xml = _ncbi_get(
                "efetch.fcgi",
                {"db": "pubmed", "id": pmid, "rettype": "xml", "retmode": "xml"},
                self.timeout,
            ).text
            root    = ET.fromstring(xml)
            parts   = [_clean_xml_text(ET.tostring(a, encoding="unicode"))
                       for a in root.iter("AbstractText")]
            abstract = " ".join(parts) if parts else "No abstract available."
        except Exception as e:
            abstract = f"Abstract fetch error: {e}"

        return (
            f"Title:   {title}\n"
            #f"Authors: {authors}\n"
            #f"Journal: {journal}\n"
            #f"Date:    {pubdate}\n"
            f"URL:     https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n\n"
            f"{abstract[:self.max_chars]}"
        )

    # ── PMC full text ──────────────────────────────────────────────────────────

    def _fetch_pmc(self, pmc_id: str) -> str:
        logger.debug("Fetching PMC ID=%s", pmc_id)
        try:
            xml = _ncbi_get(
                "efetch.fcgi",
                {"db": "pmc", "id": pmc_id, "rettype": "xml", "retmode": "xml"},
                self.timeout,
            ).text
            root = ET.fromstring(xml)
        except Exception as e:
            return f"PMC fetch error for PMC{pmc_id}: {e}"

        # Abstract first, then body paragraphs
        parts = [_clean_xml_text(ET.tostring(a, encoding="unicode"))
                 for a in root.iter("abstract")]
        if not parts:
            parts = [_clean_xml_text(ET.tostring(p, encoding="unicode"))
                     for p in root.iter("p")]

        text = " ".join(parts).strip()
        if not text:
            return f"PMC{pmc_id}: content found but could not be extracted."

        return f"PMC{pmc_id} content:\n\n{text[:self.max_chars]}"

    # ── ClinVar variation ──────────────────────────────────────────────────────

    # Normalizes whatever GermlineClassification text ClinVar submitters used
    # into one of the 5 standard tally buckets.
    _CLASS_BUCKETS = {
        "pathogenic": "Pathogenic",
        "likely pathogenic": "Likely pathogenic",
        "uncertain significance": "VUS",
        "likely benign": "Likely benign",
        "benign": "Benign",
    }

    def _fetch_clinvar_submissions(self, variation_id: str) -> str:
        """
        Fetch the per-submission record (efetch rettype=vcv, the current format —
        the older 'clinvarset' rettype used previously is deprecated and returns
        an empty <set/> silently, which is why this used to report "no comments
        found" on every variant regardless of what ClinVar actually holds).

        Returns:
          1. Classification tally across all individual submitters (P/LP/VUS/LB/B
             counts) — shows whether a P/LP call is a strong multi-submitter
             consensus or a single outlier assertion.
          2. For each Pathogenic/Likely pathogenic submission: its evidentiary
             rationale (Classification/Comment) and its source (submitter name +
             SCV accession + cited PMIDs) — so a P/LP call can be traced back to
             why and by whom, not just accepted as an aggregate label.
        """
        try:
            xml = _ncbi_get(
                "efetch.fcgi",
                {"db": "clinvar", "id": variation_id, "rettype": "vcv",
                 "is_variationid": "true", "retmode": "xml"},
                self.timeout,
            ).text
            root = ET.fromstring(xml)
        except Exception as e:
            return f"ClinVar submission fetch error for variation {variation_id}: {e}"

        va = root.find(".//VariationArchive")
        if va is None:
            return "No ClinVar variation archive found for this ID."
        cr = va.find("ClassifiedRecord")
        if cr is None:
            return "No classified record in ClinVar archive for this ID."

        cal = cr.find("ClinicalAssertionList")
        assertions = cal.findall("ClinicalAssertion") if cal is not None else []
        if not assertions:
            return "No individual submissions found in ClinVar archive for this ID."

        counts: dict[str, int] = {}
        pl_evidence: list[str] = []
        for ca in assertions:
            cl = ca.find("Classification")
            if cl is None:
                continue
            desc_el = cl.find("GermlineClassification")
            desc = (desc_el.text or "").strip() if desc_el is not None and desc_el.text else "Not provided"
            bucket = self._CLASS_BUCKETS.get(desc.lower(), desc)
            counts[bucket] = counts.get(bucket, 0) + 1

            if desc.lower() in ("pathogenic", "likely pathogenic"):
                acc = ca.find("ClinVarAccession")
                submitter = acc.get("SubmitterName", "Unknown submitter") if acc is not None else "Unknown submitter"
                scv = acc.get("Accession", "") if acc is not None else ""
                comment_el = cl.find("Comment")
                comment = _clean_xml_text(comment_el.text) if comment_el is not None and comment_el.text else ""
                pmids = [c.find("ID").text for c in cl.findall("Citation")
                         if c.find("ID") is not None and c.find("ID").get("Source") == "PubMed"]

                line = f"- [{desc}] source: {submitter} ({scv})"
                if pmids:
                    line += f" — cites PMID: {', '.join(pmids[:8])}"

                # PS3 trigger: the Comment reports functional work and the
                # submission carries a reference for it (see
                # clinvar_gene_stats.comment_functional_pmids). No reference, no tag.
                tag = ""
                ref_pmids = comment_functional_pmids(comment, pmids) if comment else []
                if ref_pmids:
                    tag = (" [functional evidence stated in submitter comment, citing "
                           + ", ".join(f"PMID:{p}" for p in ref_pmids[:5]) + "]")

                if tag:
                    line += tag
                if comment:
                    line += f"\n  Rationale: {comment[:600]}"
                pl_evidence.append(line)

        total = sum(counts.values())
        tally = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        parts = [f"ClinVar submission breakdown ({total} submissions): {tally}"]

        if pl_evidence:
            parts.append(
                "Pathogenic/Likely pathogenic submissions — evidence & source:\n"
                + "\n".join(pl_evidence[:10])
            )
        else:
            parts.append(
                "No individual submission was itself classified Pathogenic/Likely "
                "pathogenic (the aggregate classification, if P/LP, comes from expert "
                "panel review or consensus across the counts above rather than a "
                "single traceable submission)."
            )

        return "\n\n".join(parts)

    # Matches the cDNA-change token out of a combined/compound HGVS string,
    # e.g. "NM_000000:exon4:c.100A>C:p.K34T" -> "c.100A>C". ClinVar's own
    # esearch [variant name] index only matches this clean token — the full
    # colon-glued compound string comes back as one unsplittable phrase with
    # zero hits, even when the variant is in ClinVar (verified live:
    # such a variant was findable only via the bare cDNA token). "(" also
    # excluded — a "c.200T>A(p.Val67Asp)"-style string otherwise swallows
    # the trailing protein annotation into the token, producing an
    # unmatchable esearch term (verified live: a real
    # ClinVar variant silently failed to resolve until this exclusion was added).
    _CDNA_CHANGE_RE = re.compile(r"c\.[^\s:;()]+")

    def resolve_clinvar_id(self, gene: str, hgvs: str) -> str | None:
        """Look up a ClinVar variation ID for gene+HGVS via esearch (no URL known yet)."""
        cdna_match = self._CDNA_CHANGE_RE.search(hgvs) if hgvs and hgvs != "NA" else None
        variant_term = cdna_match.group(0) if cdna_match else hgvs
        term = f"{gene}[gene] AND {variant_term}[variant name]" if variant_term and variant_term != "NA" else f"{gene}[gene]"
        try:
            data = _ncbi_get(
                "esearch.fcgi",
                {"db": "clinvar", "term": term, "retmode": "json", "retmax": 1},
                self.timeout,
            ).json()
            ids = data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            logger.debug("ClinVar esearch failed for %s: %s", term, e)
            return None
        return ids[0] if ids else None

    def _fetch_clinvar(self, variation_id: str) -> str:
        logger.debug("Fetching ClinVar variation_id=%s", variation_id)
        try:
            data   = _ncbi_get(
                "esummary.fcgi",
                {"db": "clinvar", "id": variation_id, "retmode": "json"},
                self.timeout,
            ).json()
            result = data.get("result", {}).get(variation_id, {})
        except Exception as e:
            return f"ClinVar fetch error for variation {variation_id}: {e}"

        if not result:
            return f"No ClinVar data found for variation ID {variation_id}."

        name       = result.get("title", "No title")
        germline   = result.get("germline_classification", {}).get("description", "Unknown")
        review     = result.get("germline_classification", {}).get("review_status", "")
        last_eval  = result.get("germline_classification", {}).get("last_evaluated", "")
        gene       = ", ".join(g["symbol"] for g in result.get("genes", []) if "symbol" in g)
        conditions = ", ".join(t["name"]   for t in result.get("trait_set", []) if "name" in t)
        # An empty ClinVar trait_set is a load-bearing fact, not a formatting
        # gap — a Pathogenic/Likely pathogenic call with no associated
        # condition at all cannot be checked against the patient's phenotype
        # and must not read as an unremarkable blank line the model can skim
        # past (this exact case let an unrelated-disease P call reach INCLUDE).
        conditions = conditions if conditions else "(none listed — no disease/condition associated with this ClinVar record)"

        comments_block = ""
        if is_pathogenic_clinvar(germline):
            comments_block = "\n\n" + self._fetch_clinvar_submissions(variation_id)

        return (
            f"Variant:    {name}\n"
            f"Gene:       {gene}\n"
            f"Conditions: {conditions}\n"
            f"Class:      {germline}\n"
            f"Review:     {review}\n"
            f"Evaluated:  {last_eval}\n"
            f"URL:        https://www.ncbi.nlm.nih.gov/clinvar/variation/{variation_id}/"
            f"{comments_block}"
        )

    # ── Router ─────────────────────────────────────────────────────────────────

    # Also registered as legacy alias so agents calling "pubmed_fetch" still work
    aliases = ["pubmed_fetch"]

    # Bare PMID (digits only) — agent may pass just the ID without a full URL
    _BARE_PMID_RE = re.compile(r"^\d+$")

    def run(self, url: str) -> str:
        url = url.strip()

        # Handle bare PMID e.g. "39500001"
        if self._BARE_PMID_RE.match(url):
            logger.debug("NCBIFetchTool received bare PMID=%s", url)
            return self._fetch_pubmed(url)

        if not self._NCBI_RE.search(url):
            return "Error: ncbi_fetch only handles ncbi.nlm.nih.gov URLs. Use web_fetch for other sites."

        m = self._PUBMED_RE.search(url)
        if m:
            return self._fetch_pubmed(m.group(1))

        m = self._PMC_RE.search(url)
        if m:
            return self._fetch_pmc(m.group(1))

        m = self._CLINVAR_RE.search(url)
        if m:
            return self._fetch_clinvar(m.group(1))

        return (
            f"Unrecognised NCBI URL pattern: {url}\n"
            "Supported patterns:\n"
            "  pubmed.ncbi.nlm.nih.gov/<PMID>\n"
            "  pmc.ncbi.nlm.nih.gov/articles/PMC<ID>\n"
            "  ncbi.nlm.nih.gov/clinvar/variation/<ID>"
        )

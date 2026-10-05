"""
pipeline/tools/lof_mechanism.py — disease mechanism of a gene's dominant
disease (loss of function, gain of function, dominant negative), read from the
literature together with gnomAD constraint and the ClinVar consequence split.

PVS1 presumes loss of function is the disease mechanism. For a gene whose
dominant disease is caused by gain of function or a dominant-negative effect, a
heterozygous null allele is not pathogenic for that disease (e.g. a carrier of
a recessive allele in a gene whose dominant disease is caused by toxic missense
variants). ClinVar's gene-level counts cannot tell this apart when the gene
also causes a recessive disease, so the mechanism is read from abstracts.

Not a manifest tool: pipeline.py calls block_for() after second triage, only
for variants that passed it (phenotype check included), and appends the block
to the variant context before cross-analysis and conclusion. Genes that fail
the phenotype check are never looked up. One lookup per gene, cached at class
level:
  1. PubMed: gene + mechanism terms in title/abstract, top 12 by relevance.
  2. MedGen: the gene's autosomal dominant conditions (gene_disease_inheritance).
  3. One SLM call (prompts/lof_mechanism.txt), also given the gnomAD pLI/LOEUF
     (gnomad_constraint) and ClinVar P/LP missense vs nonsense/frameshift counts
     (clinvar_gene_stats): mechanism + verbatim quote, whether an experimental
     functional study demonstrates it + verbatim quote, and a 2-4 sentence
     mechanism reasoning (PMIDs not among the abstracts are removed).
  4. Both quotes are checked against the cited abstract (word overlap >= 90%).

"LOF MECHANISM VERDICT: NO_PVS1" only for GAIN_OF_FUNCTION/DOMINANT_NEGATIVE
with a verified mechanism quote AND a verified functional-study quote; read by
pipeline/core/acmg_pvs1.py, which applies it only to a heterozygous
loss-of-function variant. Anything else is "NOT_ESTABLISHED".
"""

import logging
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

from pipeline.core.errors import ToolFetchError
from pipeline.tools.websearch import _ncbi_get
from pipeline.tools.medgen_features import gene_disease_inheritance

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "lof_mechanism.txt"
_SYSTEM = "You are a clinical geneticist. Answer only from the provided abstracts."

MAX_ABSTRACTS = 12
QUOTE_MIN_OVERLAP = 0.9
_MECH_TERMS = ('("dominant negative" OR "dominant-negative" OR "gain of function" OR "gain-of-function" OR '
               'haploinsufficiency OR haploinsufficient OR "loss of function" OR "loss-of-function")')
_NO_PVS1_MECHANISMS = {"GAIN_OF_FUNCTION", "DOMINANT_NEGATIVE"}

_PLI_RE = re.compile(r"pLI = ([\d.]+)")
_LOEUF_RE = re.compile(r"LOEUF = ([\d.]+)")
_CLINVAR_COUNTS_RE = re.compile(
    r"P/LP missense variants\s*:\s*(\d+)\s*\n\s*P/LP nonsense/frameshift\s*:\s*(\d+)")
_PMID_CITE_RE = re.compile(r"\s*\(?PMID:?\s*(\d+)\)?")

_WORD_RE = re.compile(r"[a-z0-9]+")


def _quote_verified(quote: str, pmid: str | None, abstracts: dict[str, str]) -> bool:
    """True when `quote` is (near-)verbatim in the abstract of `pmid`: at least
    QUOTE_MIN_OVERLAP of its words appear, in order, as one contiguous run
    allowing small omissions (a dropped parenthesis)."""
    if not pmid or pmid not in abstracts:
        return False
    q = _WORD_RE.findall(quote.lower())
    if len(q) < 6:
        return False
    body = _WORD_RE.findall(abstracts[pmid].lower())
    best = 0
    for start in (i for i, w in enumerate(body) if w == q[0]):
        j, hits = start, 0
        for w in q:
            # look a few words ahead so an omitted fragment does not break the run
            k = next((k for k in range(j, min(j + 12, len(body))) if body[k] == w), None)
            if k is not None:
                hits, j = hits + 1, k + 1
        best = max(best, hits)
    return best / len(q) >= QUOTE_MIN_OVERLAP


def _field(text: str, name: str) -> str:
    m = re.search(rf"^{name}:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip().strip('"') if m else "NONE"


class LofMechanismTool:
    name        = "lof_mechanism"
    description = "Dominant-disease mechanism (LoF / GoF / dominant-negative) from PubMed abstracts, gnomAD constraint and ClinVar."

    _cache: dict[str, str] = {}
    _lock = threading.Lock()

    def block_for(self, gene: str, variant_context: str, llm) -> str | None:
        """The LOF MECHANISM block for `gene` (cached), or None when there is
        no gene. variant_context supplies the gnomAD constraint and ClinVar
        gene-level blocks already retrieved for this variant. A failed PubMed
        search raises ToolFetchError."""
        if not gene or gene == "NA":
            return None
        with self._lock:
            if gene in self._cache:
                return self._cache[gene]
        block = self._evaluate(gene, variant_context, llm)
        with self._lock:
            self._cache[gene] = block
        return block

    def _evaluate(self, gene: str, variant_context: str, llm) -> str:
        abstracts = self._abstracts(gene)
        conditions = [t for t, mois in (gene_disease_inheritance(gene) or [])
                      if any("dominant" in m.lower() for m in mois)]
        pli, loeuf = _PLI_RE.search(variant_context), _LOEUF_RE.search(variant_context)
        constraint = (f"pLI = {pli.group(1) if pli else 'unavailable'}, "
                      f"LOEUF = {loeuf.group(1) if loeuf else 'unavailable'}")
        cm = _CLINVAR_COUNTS_RE.search(variant_context)
        clinvar_counts = (f"{cm.group(1)} missense, {cm.group(2)} nonsense/frameshift" if cm else "unavailable")
        text = "\n\n".join(f"[PMID:{p}] {a}" for p, a in abstracts.items()) or "(none found)"
        prompt = _PROMPT_PATH.read_text().format(
            gene=gene, conditions=", ".join(conditions) or "not listed", abstracts=text,
            constraint=constraint, clinvar_counts=clinvar_counts)
        out = llm.generate(system=_SYSTEM, user=prompt, max_tokens=400)

        mech = _field(out, "MECHANISM").split()[0].upper()
        quote, pmid = _field(out, "QUOTE"), _field(out, "PMID")
        functional = _field(out, "FUNCTIONAL").upper().startswith("YES")
        f_quote, f_pmid = _field(out, "FUNCTIONAL_QUOTE"), _field(out, "FUNCTIONAL_PMID")
        pmid = re.sub(r"\D", "", pmid) or None
        f_pmid = re.sub(r"\D", "", f_pmid) or None
        q_ok = _quote_verified(quote, pmid, abstracts)
        f_ok = functional and _quote_verified(f_quote, f_pmid, abstracts)

        # Citations outside the supplied abstracts are dropped from the reasoning.
        reasoning = _PMID_CITE_RE.sub(lambda m: m.group(0) if m.group(1) in abstracts else "",
                                      _field(out, "REASONING"))

        verdict = "NO_PVS1" if mech in _NO_PVS1_MECHANISMS and q_ok and f_ok else "NOT_ESTABLISHED"
        lines = [f"LOF MECHANISM ({gene}): {mech} (dominant condition(s): {', '.join(conditions) or 'not listed'})"]
        if q_ok:
            lines.append(f'Mechanism evidence: "{quote}" (PMID:{pmid})')
        elif mech != "UNKNOWN":
            lines.append("Mechanism quote could not be verified against the cited abstract — not used.")
        if f_ok:
            lines.append(f'Functional study: "{f_quote}" (PMID:{f_pmid})')
        else:
            lines.append("Functional study demonstrating the mechanism: none verified.")
        lines.append(f"Mechanism reasoning: {reasoning}")
        lines.append(f"LOF MECHANISM VERDICT: {verdict}")
        return "\n".join(lines)

    def _abstracts(self, gene: str) -> dict[str, str]:
        """Top MAX_ABSTRACTS mechanism abstracts for `gene`. A preprint that
        PubMed links to its published version ("Update in") is replaced by
        that version: dropped when the version is already in the pool, else
        the version is fetched in its place. Real observed failure: a preprint
        claimed gain of function that its journal version revised to loss of
        function for most variants, and the model quoted the preprint."""
        term = f"{gene}[tiab] AND {_MECH_TERMS}[tiab]"
        try:
            ids = _ncbi_get("esearch.fcgi", {"db": "pubmed", "term": term, "retmax": MAX_ABSTRACTS,
                                             "sort": "relevance", "retmode": "json"}).json()["esearchresult"]["idlist"]
            if not ids:
                return {}
            out, update_in = self._fetch(ids)
            missing = sorted({u for p, u in update_in.items() if u not in out})
            if missing:
                out.update(self._fetch(missing)[0])
        except Exception as e:
            raise ToolFetchError(f"PubMed mechanism search failed for {gene}: {e}") from e
        for preprint, version in update_in.items():
            if version in out:
                logger.info("[LofMechanism] %s: preprint PMID:%s replaced by its published version PMID:%s",
                            gene, preprint, version)
                out.pop(preprint, None)
        return out

    @staticmethod
    def _fetch(ids: list[str]) -> tuple[dict[str, str], dict[str, str]]:
        """({pmid: title + abstract}, {preprint pmid: pmid of its published
        version}) for the PubMed records `ids`."""
        root = ET.fromstring(_ncbi_get("efetch.fcgi", {"db": "pubmed", "id": ",".join(ids),
                                                       "retmode": "xml"}).text)
        out, update_in = {}, {}
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title_el = art.find(".//ArticleTitle")
            title = "".join(title_el.itertext()) if title_el is not None else ""
            abstract = " ".join("".join(a.itertext()) for a in art.findall(".//AbstractText"))
            if pmid and abstract:
                out[pmid] = f"{title}\n{abstract}"
                version = art.findtext(".//CommentsCorrections[@RefType='UpdateIn']/PMID")
                if version:
                    update_in[pmid] = version
        return out, update_in

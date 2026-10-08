"""
pipeline/tools/gene_function.py — what a gene does, for the clinical report.

Not a manifest tool: clinical_report.py calls function_of() for the causative
genes only. Source order:
  1. UniProt — the reviewed human entry's curated "Function" comment, cited by
     UniProt accession and the PubMed IDs UniProt attaches to it;
  2. NCBI Gene — the RefSeq summary, cited by NCBI Gene ID, when UniProt has
     no Function comment.
One lookup per gene, cached at class level. A failed lookup returns None and
is not cached.
"""

import logging
import re
import threading

import requests

from pipeline.tools.websearch import _ncbi_get

logger = logging.getLogger(__name__)

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
TIMEOUT = 20
MAX_PMIDS = 3
_INLINE_PUBMED_RE = re.compile(r"\s*\((?:PubMed:\d+(?:,\s*)?)+\)")

_cache: dict[str, dict | None] = {}
_lock = threading.Lock()


def function_of(gene: str) -> dict | None:
    """{"text", "source", "pmids"} for `gene`, or None when neither source
    has a function text (or both lookups failed)."""
    key = gene.strip().upper()
    with _lock:
        if key in _cache:
            return _cache[key]
    failed = False
    try:
        value = _uniprot(gene)
    except Exception as e:
        logger.warning("[GeneFunction] UniProt lookup failed for %s: %s", gene, e)
        value, failed = None, True
    if value is None:
        try:
            value = _ncbi_summary(gene)
        except Exception as e:
            logger.warning("[GeneFunction] NCBI Gene lookup failed for %s: %s", gene, e)
            return None
    if value is not None or not failed:
        with _lock:
            _cache[key] = value
    return value


def _uniprot(gene: str) -> dict | None:
    r = requests.get(UNIPROT_URL, timeout=TIMEOUT, params={
        "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
        "fields": "accession,cc_function", "format": "json", "size": 1,
    })
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        return None
    entry = results[0]
    texts = [t for c in entry.get("comments", []) if c.get("commentType") == "FUNCTION"
             for t in c.get("texts", [])]
    if not texts:
        return None
    text = _INLINE_PUBMED_RE.sub("", " ".join(t.get("value", "") for t in texts)).strip()
    pmids = list(dict.fromkeys(e["id"] for t in texts for e in t.get("evidences", [])
                               if e.get("source") == "PubMed" and e.get("id")))
    return {"text": text, "source": f"UniProt {entry['primaryAccession']}",
            "pmids": pmids[:MAX_PMIDS]}


def _ncbi_summary(gene: str) -> dict | None:
    ids = _ncbi_get("esearch.fcgi", {
        "db": "gene", "term": f"{gene}[sym] AND Homo sapiens[orgn]", "retmode": "json", "retmax": 1,
    }).json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return None
    rec = _ncbi_get("esummary.fcgi", {"db": "gene", "id": ids[0], "retmode": "json"}
                    ).json().get("result", {}).get(ids[0], {})
    summary = (rec.get("summary") or "").strip()
    if not summary:
        return None
    return {"text": re.sub(r"\s*\[provided by [^\]]*\]\.?$", "", summary),
            "source": f"NCBI Gene {ids[0]}", "pmids": []}

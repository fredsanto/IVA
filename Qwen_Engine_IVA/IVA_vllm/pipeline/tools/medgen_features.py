"""
pipeline/tools/medgen_features.py — clinical features (HPO terms) of a named
condition, from NCBI MedGen.

Used by the isolated phenotype-overlap call (reasoning.run_phenotype_overlap)
so a condition name that does not itself say what it involves (an eponymous
syndrome, a numbered disease subtype) is compared against the patient's
phenotype together with its actual clinical features. Real observed failure:
a gene's condition list named an eponymous syndrome whose features include
retinal degeneration, but nothing in the retrieved evidence said so, and the
gene was excluded as "no link to visual impairment".

Also gene_disease_inheritance(): the mode of inheritance of each MedGen
disease concept the NCBI Gene record links to the gene (curated gene→disease
links, not a name search) — one of the sources of the MOI decision in
core/moi.py.

Best effort: returns None on no match or any fetch failure — a missing
feature list only means the overlap call judges that condition by its name.
"""

import html
import logging
import re
import threading

from pipeline.tools.websearch import _ncbi_get, DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)

_MAX_HITS = 5
_MAX_FEATURES = 30

# Fragments produced by splitting a comma-separated condition list that are
# not condition names on their own (e.g. "Thrombophilia, hereditary, due to
# ..." splits into "hereditary").
_NON_CONDITION_FRAGMENTS = {
    "hereditary", "familial", "autosomal dominant", "autosomal recessive",
    "x-linked", "congenital", "juvenile", "adult-onset", "early-onset",
    "type", "susceptibility to", "digenic",
}

_FEATURE_RE = re.compile(r"<ClinicalFeature[^>]*>\s*<Name>([^<]+)</Name>")
_INHERITANCE_RE = re.compile(r"<ModeOfInheritance[^>]*>.*?<Name>([^<]+)</Name>", re.S)

_cache: dict[str, tuple[str, list[str]] | None] = {}
_lock = threading.Lock()


# Stage 1b's answer when the evidence names no condition for the gene
# (prompts/gene_phenotype_extraction.txt) — not a condition name.
NO_CONDITION_ANSWER = "none established"


def split_condition_list(phenotype_list: str) -> list[str]:
    """Condition names from Stage 1b's PHENOTYPE tag ("A, B, C" or "A; B")."""
    if not phenotype_list:
        return []
    answer = phenotype_list.strip().strip("\"'*.").strip().lower()
    if answer == "na" or answer.startswith(NO_CONDITION_ANSWER):
        return []
    sep = ";" if ";" in phenotype_list else ","
    names = []
    for part in phenotype_list.split(sep):
        name = part.strip().strip(".")
        if len(name) < 4 or name.lower() in _NON_CONDITION_FRAGMENTS:
            continue
        if name.lower() not in (n.lower() for n in names):
            names.append(name)
    return names


def _fetch(name: str, gene: str) -> tuple[str, list[str]] | None:
    # Always restricted to MedGen concepts linked to the gene: a name-only search
    # returned other genes' diseases (observed: "Rett syndrome" -> FOXG1 disorder,
    # a generic DEE name -> a DEE subtype of another gene), so the patient was
    # compared against the wrong disease's features.
    if not gene:
        return None
    ids: list[str] = []
    for term in (f'"{name}"[title] AND {gene}[gene]', f'({name}) AND {gene}[gene]'):
        ids = _ncbi_get(
            "esearch.fcgi",
            {"db": "medgen", "term": term, "retmode": "json", "retmax": _MAX_HITS},
            DEFAULT_TIMEOUT,
        ).json().get("esearchresult", {}).get("idlist", [])
        if ids:
            break
    if not ids:
        return None
    result = _ncbi_get(
        "esummary.fcgi", {"db": "medgen", "id": ",".join(ids), "retmode": "json"},
        DEFAULT_TIMEOUT,
    ).json().get("result", {})
    hits = []
    for uid in result.get("uids", []):
        rec = result.get(uid, {})
        feats = _FEATURE_RE.findall(html.unescape(rec.get("conceptmeta", "")))
        hits.append((rec.get("title", ""), list(dict.fromkeys(feats))[:_MAX_FEATURES]))
    # Exact-title hit with features, else the only hit (if it has features);
    # several non-exact hits are ambiguous -> None (judged by name only).
    for title, feats in hits:
        if feats and title.strip().lower() == name.lower():
            return title, feats
    if len(hits) == 1 and hits[0][1]:
        return hits[0]
    return None


def condition_features(name: str, gene: str) -> tuple[str, list[str]] | None:
    """(MedGen title, clinical feature names) for a condition name among the
    MedGen concepts linked to the gene, or None."""
    key = f"{gene.strip().upper()}|{name.strip().lower()}"
    with _lock:
        if key in _cache:
            return _cache[key]
    try:
        value = _fetch(name, gene)
    except Exception as e:
        logger.warning("[MedGen] feature lookup failed for %r (%s): %s", name, gene, e)
        return None  # not cached: a transient failure may succeed next time
    with _lock:
        _cache[key] = value
    return value


_gene_cache: dict[str, list[tuple[str, list[str]]]] = {}


def _fetch_gene_diseases(gene: str) -> list[tuple[str, list[str]]]:
    ids = _ncbi_get(
        "esearch.fcgi",
        {"db": "gene", "term": f"{gene}[sym] AND Homo sapiens[orgn]", "retmode": "json", "retmax": 1},
        DEFAULT_TIMEOUT,
    ).json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    linksets = _ncbi_get(
        "elink.fcgi",
        {"dbfrom": "gene", "db": "medgen", "id": ids[0], "linkname": "gene_medgen_diseases",
         "retmode": "json"},
        DEFAULT_TIMEOUT,
    ).json().get("linksets", [])
    uids = [u for ls in linksets[:1] for db in ls.get("linksetdbs", []) for u in db.get("links", [])]
    if not uids:
        return []
    result = _ncbi_get(
        "esummary.fcgi", {"db": "medgen", "id": ",".join(uids), "retmode": "json"},
        DEFAULT_TIMEOUT,
    ).json().get("result", {})
    return [
        (result[u].get("title", ""),
         list(dict.fromkeys(_INHERITANCE_RE.findall(html.unescape(result[u].get("conceptmeta", ""))))))
        for u in result.get("uids", [])
    ]


def gene_disease_inheritance(gene: str) -> list[tuple[str, list[str]]] | None:
    """[(MedGen disease title, mode-of-inheritance names)] for every MedGen disease
    concept linked to the gene's NCBI Gene record; [] if none; None on a fetch
    failure (not cached, so a later call can retry)."""
    key = gene.strip().upper()
    with _lock:
        if key in _gene_cache:
            return _gene_cache[key]
    try:
        value = _fetch_gene_diseases(gene)
    except Exception as e:
        logger.warning("[MedGen] gene disease lookup failed for %s: %s", gene, e)
        return None
    with _lock:
        _gene_cache[key] = value
    return value

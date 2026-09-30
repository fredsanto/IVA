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


def _fetch(name: str) -> tuple[str, list[str]] | None:
    ids: list[str] = []
    for term in (f'"{name}"[title]', name):
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
    # Prefer an exact-title hit with features, then any hit with features.
    for title, feats in hits:
        if feats and title.strip().lower() == name.lower():
            return title, feats
    for title, feats in hits:
        if feats:
            return title, feats
    return None


def condition_features(name: str) -> tuple[str, list[str]] | None:
    """(MedGen title, clinical feature names) for a condition name, or None."""
    key = name.strip().lower()
    with _lock:
        if key in _cache:
            return _cache[key]
    try:
        value = _fetch(name)
    except Exception as e:
        logger.warning("[MedGen] feature lookup failed for %r: %s", name, e)
        return None  # not cached: a transient failure may succeed next time
    with _lock:
        _cache[key] = value
    return value

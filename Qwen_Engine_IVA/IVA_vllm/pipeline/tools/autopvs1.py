"""
pipeline/tools/autopvs1.py — AutoPVS1 lookup with HGVS-first query strategy.

Query strategy (in order):
  1. HGVS search path  — GET /search?q={hgvs}&genome_version={build}
                         AutoPVS1 resolves the transcript to its canonical
                         coordinates and redirects; immune to stale genomic
                         coords in the source CSV.
  2. VCF coords path   — GET /variant/{build}/{chrom}-{pos}-{ref}-{alt}
                         Fallback when no valid HGVS is available.

Both paths reuse the same HTML parser. Results where AutoPVS1 returns
"Intergenic" or a mismatched gene are discarded before returning.

Public functions (module-level, used internally by AutoPVS1Tool):
  vcf_to_autopvs1_id(chrom, pos, ref, alt)         -> "6-100896130-T-C"
  parse_variant_coords(variant_str, ...)            -> (chrom, pos, ref, alt); raises ValueError
  fetch_variant_data(variant_id, hg)                -> dict | None
  fetch_variant_data_search(hgvs_str, hg)           -> (dict, resolved_id) | (None, None)
  fetch_and_format_autopvs1(chrom, pos, ref, alt,
                             hgvs="", hg="hg19",
                             expected_gene="")       -> formatted string block | None

Pipeline class:
  AutoPVS1Tool — NetworkTool; gate uses the hybrid rule+LLM _pvs1_decision logic
                 ported from server_main._pvs1_decision().
"""

import re
import logging
import threading
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

from pipeline.tools.base import NetworkTool
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError


# ═══════════════════════════════════════════════════════════════════════════════
# COORD EXTRACTION  (no network call — pure string parsing)
# ═══════════════════════════════════════════════════════════════════════════════

# SNV: "chr1:1000000 T⇒C"  or  "chr1:1000000 T>C"  or  "chr1:1000000 T-C"
_SNV_RE = re.compile(
    r"chr(\w+):(\d+)\s+([ACGTacgt]+)\s*[=>\u21d2\-]+\s*([ACGTacgt]+)"
)
# Deletion: "chr1:1000000 delA"
_DEL_RE = re.compile(
    r"chr(\w+):(\d+)\s+del([ACGTacgt]+)",
    re.IGNORECASE,
)
# Insertion: "chrX:12345 insACG"
_INS_RE = re.compile(
    r"chr(\w+):(\d+)\s+ins([ACGTacgt]+)",
    re.IGNORECASE,
)
# DelIns: "chr1:12345 ACGdelinsT"
_DELINS_RE = re.compile(
    r"chr(\w+):(\d+)\s+([ACGTacgt]+)\s*delins\s*([ACGTacgt]+)",
    re.IGNORECASE,
)


def vcf_to_autopvs1_id(chrom: str, pos: str, ref: str, alt: str) -> str:
    """
    Build the AutoPVS1 variant ID string from raw VCF-style fields.
    Strips 'chr' prefix, normalises chrom name.
    e.g. ("chr6", "100896130", "T", "C") → "6-100896130-T-C"
    """
    chrom = chrom.strip().lower().removeprefix("chr")
    if chrom.isdigit():
        chrom = str(int(chrom))
    return f"{chrom}-{pos.strip()}-{ref.strip().upper()}-{alt.strip().upper()}"


def parse_variant_coords(
    variant_str: str,
    chrom_field: str = "",
    pos_field:   str = "",
    ref_field:   str = "",
    alt_field:   str = "",
) -> tuple[str, str, str, str]:
    """
    Extract (chrom, pos, ref, alt) from available CSV fields.

    Strategy (in order):
      1. If Ref_seq and Var_seq are non-NA → use them directly
      2. Parse the Variant column free-text string
      3. Fall back to chrom+pos from columns + loose base extraction from Variant string

    Raises ValueError if extraction fails entirely — e.g. the Variant string
    uses a separator character none of the regexes recognize (observed case:
    a mangled '?' where an ref>alt arrow should be, upstream CSV encoding
    damage). Callers that intentionally sample many variants and expect some
    to be unparseable (build_detection's candidate loop) must catch this and
    skip; callers processing a single already-gated variant (AutoPVS1Tool.run)
    should let it surface as a typed pipeline error instead of returning
    silent None.
    """

    def ok(v: str) -> bool:
        return bool(v) and v.strip().upper() not in ("NA", "", ".")

    # ── Strategy 1: explicit Ref_seq / Var_seq columns ────────────────────────
    if ok(ref_field) and ok(alt_field) and ok(chrom_field) and ok(pos_field):
        return chrom_field, pos_field, ref_field, alt_field

    # ── Strategy 2: parse the Variant column string ───────────────────────────
    v = variant_str.strip()

    m = _SNV_RE.match(v)
    if m:
        return f"chr{m.group(1)}", m.group(2), m.group(3).upper(), m.group(4).upper()

    m = _DEL_RE.match(v)
    if m:
        # AutoPVS1 accepts the deleted bases as REF and "-" as ALT for simple deletions
        return f"chr{m.group(1)}", m.group(2), m.group(3).upper(), "-"

    m = _INS_RE.match(v)
    if m:
        return f"chr{m.group(1)}", m.group(2), "-", m.group(3).upper()

    m = _DELINS_RE.match(v)
    if m:
        return f"chr{m.group(1)}", m.group(2), m.group(3).upper(), m.group(4).upper()

    # ── Strategy 3: chrom+pos known, try loose base extraction ────────────────
    if ok(chrom_field) and ok(pos_field):
        bases = re.findall(r"[ACGTacgt]+", v)
        if len(bases) >= 2:
            return chrom_field, pos_field, bases[0].upper(), bases[1].upper()

    raise ValueError(
        f"parse_variant_coords: could not extract coordinates — "
        f"variant_str={variant_str!r} chrom={chrom_field!r} pos={pos_field!r} "
        f"ref={ref_field!r} alt={alt_field!r}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AutoPVS1 fetch + HTML parse
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_build(build: str) -> str:
    """Normalise any genome build alias to the two tokens AutoPVS1 accepts in its URL."""
    if build in ("hg19", "19", "37", "GRCh37", "b37"):
        return "hg19"
    return "hg38"


_AUTOPVS1_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "User-Agent": "Mozilla/5.0 (compatible; PythonScript/1.0)",
}

# Per-process rate limiter: autopvs1.bgi.com is a free public service.
# 1 req/s per process → 6 parallel jobs = ~6 req/s total.
_AUTOPVS1_RATE_LOCK      = threading.Lock()
_AUTOPVS1_LAST_CALL_TIME = 0.0
_AUTOPVS1_DELAY          = 1.0


def _autopvs1_rate_limit() -> None:
    global _AUTOPVS1_LAST_CALL_TIME
    with _AUTOPVS1_RATE_LOCK:
        elapsed = time.time() - _AUTOPVS1_LAST_CALL_TIME
        wait = _AUTOPVS1_DELAY - elapsed
        if wait > 0:
            time.sleep(wait)
        _AUTOPVS1_LAST_CALL_TIME = time.time()


def _is_valid_hgvs(s: str) -> bool:
    """True if s looks like a transcript HGVS cDNA string (NM_xxx:c.xxx or NR_xxx:c.xxx)."""
    return bool(s) and bool(re.match(r"^(NM_|NR_|ENST)\d+", s, re.IGNORECASE)) and ":c." in s


_CLEAN_HGVS_RE = re.compile(
    r"(N[MR]_\d+(?:\.\d+)?|ENST\d+(?:\.\d+)?)[^|]*?(c\.[^\s|]+)"
)


def _clean_transcript_hgvs(raw: str) -> str:
    """
    Extract a bare 'TRANSCRIPT:c.CHANGE' string from a raw HGVS field.

    The canonical HGVS field is not always already in that bare form — it can be
    gene-prefixed ('GENE:NM_xxx:exonN:c.xxx', as produced by some ANNOVAR-style
    annotators), carry a trailing protein annotation ('... p.(Xxx)'), or list
    multiple transcripts pipe-separated ('GENE:NM_1:...|GENE:NM_2:...'). AutoPVS1's
    /search endpoint only accepts the bare 'transcript:c.change' form and only
    needs one transcript, so this strips gene prefix / exon segment / protein
    suffix / extra transcripts down to that.

    Returns "" if no transcript:c. pattern is found anywhere in raw.
    """
    if not raw:
        return ""
    m = _CLEAN_HGVS_RE.search(raw)
    if not m:
        return ""
    return f"{m.group(1)}:{m.group(2)}"


_TRANSCRIPT_ACCESSION_RE = re.compile(r"(N[MR]_\d+|ENST\d+)")


def _transcript_base(hgvs_or_accession: str) -> str:
    """Bare transcript accession with version stripped, e.g. 'NM_000000.1' or
    'NM_000000.1:c.100A>T' -> 'NM_000000'. Returns "" if none found."""
    if not hgvs_or_accession:
        return ""
    m = _TRANSCRIPT_ACCESSION_RE.search(hgvs_or_accession)
    return m.group(1) if m else ""


def _is_transcript_mismatch(queried_hgvs: str, returned_chgvs: str | None) -> bool:
    """
    True when AutoPVS1's returned cHGVS is pinned to a materially different
    transcript than the one we queried (version suffix ignored).

    Some genes have multiple RefSeq transcripts with divergent exon/intron
    structure in a hotspot region (an exon present in transcript NM_A but
    absent/renumbered in the "canonical" transcript NM_B).
    AutoPVS1's /search endpoint can silently resolve a query against the wrong
    one of these and return exon/intron boundaries that do not correspond to the
    transcript actually reported in the input data — e.g. a query against
    NM_A's hotspot exon came back as "NM_B:c.1000+331_1000+332del
    (Intron)", which is a real position on a different transcript, not evidence
    that the queried variant is intronic. Silently trusting the gene-level match
    (GENE_X == GENE_X) alone is not enough cross-validation; the transcript accession
    itself must also match.
    """
    queried = _transcript_base(queried_hgvs)
    returned = _transcript_base(returned_chgvs or "")
    return bool(queried and returned and queried != returned)


def _is_intergenic_result(d: dict) -> bool:
    """True when AutoPVS1 returned an intergenic hit or no gene — unusable result."""
    return (d.get("variant_type") or "").lower() == "intergenic" or d.get("gene") is None


def fetch_variant_data(variant_id: str, hg: str = "hg19") -> dict | None:
    """Query AutoPVS1 by VCF-style ID (e.g. '1-47882707-C-A'). Returns parsed dict or None."""
    url = f"https://autopvs1.bgi.com/variant/{_norm_build(hg)}/{variant_id}"
    print(f"[AutoPVS1] VCF fetch: {url}")
    try:
        _autopvs1_rate_limit()
        response = requests.get(url, headers=_AUTOPVS1_HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[AutoPVS1] VCF request failed: {e}")
        return None
    return _parse_autopvs1_html(response.text)


def fetch_variant_data_search(
    hgvs_str: str,
    hg: str = "hg19",
) -> tuple[dict, str] | tuple[None, None]:
    """
    Query AutoPVS1 via the /search endpoint using a transcript HGVS string.

    AutoPVS1 resolves the HGVS to its canonical VCF coordinates internally
    and redirects to /variant/{build}/{resolved_id}. This path is robust to
    stale or wrong genomic coordinates in the source CSV.

    Returns (parsed_dict, resolved_variant_id) on success, (None, None) on failure.
    The resolved_variant_id comes from the final redirect URL (e.g. '1-47882707-C-A').
    """
    params = {"q": hgvs_str, "genome_version": _norm_build(hg)}
    url = "https://autopvs1.bgi.com/search?" + urlencode(params)
    print(f"[AutoPVS1] HGVS search: {url}")
    try:
        _autopvs1_rate_limit()
        response = requests.get(url, headers=_AUTOPVS1_HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[AutoPVS1] HGVS search failed: {e}")
        return None, None
    # Extract the resolved variant_id from the final redirect URL
    resolved_id = response.url.rstrip("/").split("/")[-1]
    return _parse_autopvs1_html(response.text), resolved_id


def _parse_autopvs1_html(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    data = {}

    h3 = soup.find("h3")
    if h3:
        h3_text = h3.get_text(strip=True)
        m = re.match(r"^([\w\-]+):\s*([\d\w\-]+)$", h3_text)
        if m:
            data["variant_type"] = m.group(1)
            data["variant"]      = m.group(2)
        else:
            data["variant_type"] = None
            data["variant"]      = h3_text

    def get_field(label: str) -> str | None:
        tag = soup.find("b", string=re.compile(rf"^\s*{re.escape(label)}\s*$"))
        if not tag:
            return None
        val = ""
        for sib in tag.next_siblings:
            text = sib.get_text(strip=True) if hasattr(sib, "get_text") else str(sib).strip()
            if text:
                val = text
                break
        return None if val in ("", "-", "na", "N/A") else val

    data["gene"]               = get_field("Gene:")
    data["chgvs"]              = get_field("cHGVS:")
    data["phgvs"]              = get_field("pHGVS:")
    data["exon"]               = get_field("Exon:")
    data["intron"]             = get_field("Intron:")
    data["pli"]                = get_field("pLI:")
    data["haploinsufficiency"] = get_field("Haploinsufficiency:")

    _key_fields = ("gene", "chgvs", "phgvs", "pli", "haploinsufficiency")
    _none_count = sum(1 for f in _key_fields if data.get(f) is None)
    if _none_count > 2:
        logger.warning(
            "[AutoPVS1] %d/%d key fields are None — HTML structure may have changed",
            _none_count, len(_key_fields),
        )

    for btn in soup.select("a.btn-primary"):
        label = btn.get_text(strip=True)
        href  = btn.get("href", "")
        if "clinvar" in label.lower():
            m = re.search(r"/variation/(\w+)", href)
            data["clinvar_id"]  = m.group(1) if m and m.group(1) != "na" else None
            data["clinvar_url"] = href if "na" not in href else None
        elif "gnomad" in label.lower():
            data["gnomad_url"] = href
        elif "omim" in label.lower():
            data["omim_url"] = href

    flowchart = soup.find("figure")
    if flowchart:
        caption = flowchart.find("figcaption")
        data["pvs1_path"] = caption.get_text(strip=True) if caption else None
        steps = [
            li.find("code").get_text(" ", strip=True)
            for li in flowchart.find_all("li")
            if li.find("code")
        ]
        data["pvs1_decision_steps"] = steps
        data["pvs1_strength"]       = steps[-1] if steps else None
    else:
        data["pvs1_path"]           = None
        data["pvs1_decision_steps"] = []
        data["pvs1_strength"]       = None

    table = soup.find("table", class_="table-bordered")
    if table:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = []
        for tr in table.find("tbody").find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells and len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))
        data["disease_mechanism"] = rows if rows else None
    else:
        data["disease_mechanism"] = None

    full_text = soup.get_text()
    # pvs1_applicable: True only when a flowchart is present AND no "incompatible" message.
    # Defaulting to True for empty pages (old behaviour) was incorrect — an empty page
    # means AutoPVS1 found nothing, not that PVS1 applies.
    # Note: `flowchart` is already bound above from soup.find("figure").
    data["pvs1_applicable"] = (
        flowchart is not None
        and not bool(re.search(r"incompatible with recommendations", full_text, re.I))
    )
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# High-level function  (HGVS-first strategy + guards)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_and_format_autopvs1(
    chrom: str,
    pos:   str,
    ref:   str,
    alt:   str,
    hgvs:  str = "",
    hg:    str = "hg19",
    expected_gene: str = "",
) -> str | None:
    """
    Full pipeline: fetch AutoPVS1 evidence and format it as a string block.

    Query strategy:
      1. HGVS search path (/search?q=) — preferred; resolves the transcript to
         AutoPVS1's canonical coordinates, immune to stale CSV coords.
         Only used when hgvs contains a valid transcript HGVS (NM_xxx:c.xxx).
      2. VCF coords path (/variant/...) — fallback when HGVS is absent or the
         search path fails.

    Results are discarded (returns None) if:
      - AutoPVS1 returns variant_type="Intergenic" (coordinates not in any gene)
      - The returned gene does not match expected_gene (when provided)

    Args:
        chrom         : chromosome, e.g. "chr18" or "18"
        pos           : 1-based position string, e.g. "44580785"
        ref           : reference allele, e.g. "A"  (use "-" for pure insertions)
        alt           : alternate allele, e.g. "T"  (use "-" for pure deletions)
        hgvs          : HGVS notation from the variant dict, e.g.
                        "NM_000000:c.100C>A p.C34X" (space-split: first token used)
        hg            : genome build ("hg19" or "hg38")
        expected_gene : variant's Gene field; used for cross-validation

    Returns a multi-line string ready to embed in the augmented context,
    or None if the fetch failed or returned unusable data.
    """
    hgvs_clean = _clean_transcript_hgvs(hgvs)      # "NM_000000:c.100C>A"
    d: dict | None = None
    variant_id: str = vcf_to_autopvs1_id(chrom, pos, ref, alt)  # fallback label

    # ── Strategy 1: HGVS search (preferred) ──────────────────────────────────
    if _is_valid_hgvs(hgvs_clean):
        d, resolved_id = fetch_variant_data_search(hgvs_clean, hg=hg)
        if d is not None:
            if _is_intergenic_result(d):
                logger.warning(
                    "[AutoPVS1] HGVS search returned intergenic for %s — discarding",
                    hgvs_clean,
                )
                d = None
            elif _is_transcript_mismatch(hgvs_clean, d.get("chgvs")):
                logger.warning(
                    "[AutoPVS1] Transcript mismatch: queried %s, AutoPVS1 resolved "
                    "against %s — discarding HGVS-search result, falling back to "
                    "VCF coords",
                    hgvs_clean, d.get("chgvs"),
                )
                d = None
            else:
                variant_id = resolved_id  # use the coordinate AutoPVS1 resolved to

    # ── Strategy 2: VCF coords (fallback) ────────────────────────────────────
    if d is None:
        vcf_id = vcf_to_autopvs1_id(chrom, pos, ref, alt)
        print(f"[AutoPVS1] Falling back to VCF coords: {vcf_id}")
        d = fetch_variant_data(vcf_id, hg=hg)
        if d is not None:
            if _is_intergenic_result(d):
                logger.warning(
                    "[AutoPVS1] VCF coords also returned intergenic (%s) — discarding. "
                    "Possible coordinate error or AutoPVS1 annotation gap for this gene.",
                    vcf_id,
                )
                return None

    if d is None:
        return None

    # ── Gene cross-validation ─────────────────────────────────────────────────
    if expected_gene:
        returned_gene = (d.get("gene") or "").upper()
        if returned_gene and returned_gene != expected_gene.upper():
            logger.warning(
                "[AutoPVS1] Gene mismatch: expected %s, got %s (%s) — discarding",
                expected_gene, d.get("gene"), variant_id,
            )
            return None

    # ── Format output ─────────────────────────────────────────────────────────
    def _val(v) -> str:
        return str(v) if v not in (None, "", [], {}) else "—"

    steps_str   = " -> ".join(d.get("pvs1_decision_steps") or []) or "—"
    disease_str = "—"
    if d.get("disease_mechanism"):
        disease_str = "; ".join(
            ", ".join(f"{k}: {v}" for k, v in row.items())
            for row in d["disease_mechanism"]
        )

    label = hgvs_clean if hgvs_clean else variant_id
    lines = [
        f"AUTOPVS1 EVIDENCE ({label}):",
        f"  Variant        : {variant_id}  ({_val(d.get('variant_type'))})",
        f"  Gene           : {_val(d.get('gene'))}",
        f"  cHGVS          : {_val(d.get('chgvs'))}",
        f"  pHGVS          : {_val(d.get('phgvs'))}",
        f"  Exon           : {_val(d.get('exon'))}   Intron: {_val(d.get('intron'))}",
        f"  pLI            : {_val(d.get('pli'))}",
        f"  Haploinsuff.   : {_val(d.get('haploinsufficiency'))}",
        f"  PVS1 path      : {_val(d.get('pvs1_path'))}",
        f"  PVS1 steps     : {steps_str}",
        f"  PVS1 strength  : {_val(d.get('pvs1_strength'))}",
        f"  PVS1 applicable: {d.get('pvs1_applicable', '—')}",
        f"  Disease mech.  : {disease_str}",
        f"  ClinVar        : {_val(d.get('clinvar_url'))}",
        f"  gnomAD         : {_val(d.get('gnomad_url'))}",
        f"  OMIM           : {_val(d.get('omim_url'))}",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PVS1 GATE CONSTANTS  (ported from server_main._pvs1_decision)
# ═══════════════════════════════════════════════════════════════════════════════

# Variant types that unambiguously warrant PVS1.
# "stoploss" is intentionally excluded: stop-loss variants extend the protein
# rather than truncate it, so PVS1 does not apply and AutoPVS1 would give
# misleading output for that class.
_CLEAR_LOF_TYPES = {
    "frameshift", "nonsense", "stopgain",
    "splice-3", "splice-5", "splice_site", "splicing",
    "start_lost", "initiator_codon", "deletion", "indel",
}

# Variant types that unambiguously do NOT warrant PVS1
_CLEAR_NON_LOF_TYPES = {
    "missense", "synonymous", "silent", "utr", "utr3", "utr5",
    "intergenic", "upstream", "downstream",
}

# Minimum SpliceAI max-delta to override the synonymous/silent fast-NO below.
# ACMG/ClinGen SVI: a synonymous change is not exempt from PVS1 — it can
# disrupt an exonic splicing enhancer/silencer or create a cryptic splice
# site without altering the encoded residue. 0.5 matches SpliceAI's own
# "MODERATE splice impact" band (see spliceai.py::_interpret).
_SPLICE_OVERRIDE_THRESHOLD = 0.5

_SPLICEAI_MAX_DELTA_RE = re.compile(r"Max delta\s*:\s*([\d.]+)")


def _spliceai_max_delta(context: "ToolContext") -> float | None:
    """
    Best-effort max SpliceAI delta score for the current variant, checked
    before letting a synonymous/silent Type call PVS1 not-applicable.
    Prefers a precomputed canonical SpliceAI_score field (from the input
    CSV); falls back to parsing this run's own spliceai tool output.
    """
    raw = context.field("SpliceAI_score")
    if raw and raw != "NA":
        try:
            return float(raw)
        except ValueError:
            pass

    spliceai_output = context.all_outputs.get("spliceai")
    if spliceai_output:
        m = _SPLICEAI_MAX_DELTA_RE.search(spliceai_output)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE TOOL CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class AutoPVS1Tool(NetworkTool):
    """Fetches ACMG PVS1 criterion evaluation from AutoPVS1 for LoF variants (hg19)."""

    name        = "autopvs1"
    description = "Fetches ACMG PVS1 criterion evaluation from AutoPVS1 for LoF variants (hg19)."

    def gate(self, variant: dict, context: ToolContext) -> bool:
        """
        Three-way hybrid gate — ported from server_main._pvs1_decision():
          - Clearly LoF        → True  (no LLM call)
          - Clearly not LoF    → False (no LLM call)
          - Ambiguous          → ask LLM (max_tokens=5, very cheap)

        Uses Type field first, then HGVS clues, then LLM.
        """
        type_val = variant.get("Type", "").lower()
        hgvs_val = variant.get("HGVS", "").lower()

        # ── Fast YES ─────────────────────────────────────────────────────────
        if any(t in type_val for t in _CLEAR_LOF_TYPES):
            print(f"[PVS1 gate] YES (rule) — type: {type_val}")
            return True

        # HGVS frameshift / stop clues
        if any(c in hgvs_val for c in ["fs", "ter", "del", "dup", "ins", "ext*"]):
            # but exclude if it's clearly synonymous p.(X=)
            if "=" not in hgvs_val:
                print(f"[PVS1 gate] YES (hgvs clue) — hgvs: {hgvs_val}")
                return True

        # ── Synonymous/silent override ──────────────────────────────────────
        # A synonymous change does not preclude PVS1: it can still disrupt a
        # splice site or exonic splicing enhancer. Check SpliceAI evidence
        # before letting the type-based fast-NO below silently exempt it.
        is_synonymous = (
            any(t in type_val for t in ("synonymous", "silent"))
            or re.search(r"p\.\([a-z]+\d+=\)", hgvs_val)
        )
        if is_synonymous:
            max_ds = _spliceai_max_delta(context)
            if max_ds is not None and max_ds >= _SPLICE_OVERRIDE_THRESHOLD:
                print(
                    f"[PVS1 gate] YES (SpliceAI override, delta={max_ds:.2f}) "
                    f"— type: {type_val} hgvs: {hgvs_val}"
                )
                return True

        # ── Fast NO ──────────────────────────────────────────────────────────
        if any(t in type_val for t in _CLEAR_NON_LOF_TYPES):
            print(f"[PVS1 gate] NO (rule) — type: {type_val}")
            return False

        # Synonymous: protein annotation ends with = e.g. p.(Ser47=)
        if re.search(r"p\.\([a-z]+\d+=\)", hgvs_val):
            print(f"[PVS1 gate] NO (synonymous hgvs) — hgvs: {hgvs_val}")
            return False

        # Deep intronic: offset > 10 bp from exon boundary
        deep = re.search(r"[+\-](\d+)[acgt]", hgvs_val)
        if deep and int(deep.group(1)) > 10:
            print(f"[PVS1 gate] NO (deep intronic) — hgvs: {hgvs_val}")
            return False

        # UTR variants
        if re.match(r"c\.\*", hgvs_val) or re.match(r"c\.-", hgvs_val):
            print(f"[PVS1 gate] NO (UTR) — hgvs: {hgvs_val}")
            return False

        # ── Ambiguous → LLM judge ─────────────────────────────────────────────
        print(f"[PVS1 gate] Ambiguous — asking LLM. type={type_val} hgvs={hgvs_val}")
        answer = context.llm.generate(
            system=(
                "You are a clinical genetics assistant. "
                "Answer with a single word: YES or NO. No explanation."
            ),
            user=(
                "Should the PVS1 ACMG criterion be evaluated for this variant?\n"
                f"Type: {type_val}\n"
                f"HGVS: {hgvs_val}\n\n"
                "Answer YES only if this is a loss-of-function variant: frameshift, "
                "nonsense/stop-gain, canonical splice site (±1 or ±2 bp from exon), "
                "start-loss, or exon-spanning deletion/duplication.\n"
                "Answer NO for missense, synonymous, deep intronic, or UTR variants."
            ),
            max_tokens=5,
        )
        decision = "yes" in answer.strip().lower()
        print(f"[PVS1 gate] LLM answered: {answer.strip()!r} → {decision}")
        return decision

    def run(self, variant: dict, context: ToolContext) -> str | None:
        """
        Extract VCF coordinates from the canonical variant dict,
        then call fetch_and_format_autopvs1().
        Returns the formatted evidence block, or None if AutoPVS1 has no
        usable result (intergenic, gene mismatch, fetch failure).
        Raises ToolParseError only when BOTH VCF-coord extraction AND a bare
        transcript HGVS extraction fail — the gate already confirmed this
        variant is LoF-eligible, so having neither usable input is a data
        problem, not a skip. When coords are unparseable but the HGVS field
        yields a clean 'transcript:c.change' string (see
        `_clean_transcript_hgvs`), that alone is enough: AutoPVS1's
        HGVS-search path resolves canonical coordinates itself, so the VCF
        coords below only matter as a fallback query/label if that search
        also fails downstream.
        """
        hgvs_raw = variant.get("HGVS", "")
        clean_hgvs = _clean_transcript_hgvs(hgvs_raw)

        try:
            coords = parse_variant_coords(
                variant_str=variant.get("Variant", ""),
                chrom_field=variant.get("Chromosome", ""),
                pos_field=variant.get("Position", ""),
                ref_field=variant.get("Ref_seq", ""),
                alt_field=variant.get("Var_seq", ""),
            )
        except ValueError as e:
            if not _is_valid_hgvs(clean_hgvs):
                raise ToolParseError(f"AutoPVS1: {e}") from e
            chrom = variant.get("Chromosome", "") or "NA"
            pos   = variant.get("Position", "") or "NA"
            ref   = variant.get("Ref_seq", "") or "N"
            alt   = variant.get("Var_seq", "") or "N"
        else:
            chrom, pos, ref, alt = coords

        try:
            result = fetch_and_format_autopvs1(
                chrom, pos, ref, alt,
                hgvs=hgvs_raw,
                hg=context.genome_build,
                expected_gene=variant.get("Gene", ""),
            )
        except requests.exceptions.RequestException as e:
            raise ToolFetchError(
                f"AutoPVS1 request failed for {variant.get('Variant', '?')}: {e}"
            ) from e

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Quick test  (python -m pipeline.tools.autopvs1)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) not in (5, 6):
        sys.exit("usage: python -m pipeline.tools.autopvs1 <chrom> <pos> <ref> <alt> [transcript:c.HGVS]")
    chrom, pos, ref, alt = sys.argv[1:5]
    hgvs = sys.argv[5] if len(sys.argv) == 6 else None

    print(f"=== vcf_to_autopvs1_id ===\n  {vcf_to_autopvs1_id(chrom, pos, ref, alt)}")
    print(f"\n=== Live fetch ({chrom}:{pos} {ref}>{alt}) ===")
    result = fetch_and_format_autopvs1(chrom, pos, ref, alt, hgvs=hgvs)
    print(result or "No result")

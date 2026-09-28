"""
pipeline/tools/spliceai.py — SpliceAI splice impact scores via Broad Institute API.

Coordinates are extracted from canonical variant dict fields
(Chromosome, Position, Ref_seq, Var_seq, Variant) using the same
parse_variant_coords() helper as AutoPVS1Tool.

Gate logic:
  - Runs on any variant with resolvable genomic coordinates
  - Skips synonymous and intergenic variants (no splice impact possible)
  - Skips if Ref_seq or Var_seq is NA and Variant string cannot be parsed

Public pipeline class:
  SpliceAITool — NetworkTool
"""

import threading
import time
import requests

from pipeline.tools.base import NetworkTool
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError

# Reuse the coord extractor already written for AutoPVS1
from pipeline.tools.autopvs1 import parse_variant_coords


# ── Config ────────────────────────────────────────────────────────────────────

SPLICEAI_API = {
    "38": "https://spliceai-38-xwkwwwxdwq-uc.a.run.app/spliceai/",
    "19": "https://spliceai-37-xwkwwwxdwq-uc.a.run.app/spliceai/",
    "37": "https://spliceai-37-xwkwwwxdwq-uc.a.run.app/spliceai/",
}

# Per-process rate limiter: 1 req/s → 6 parallel jobs = ~6 req/s total on the API.
# Without this, 32 workers per job burst all requests simultaneously → 429s.
_SPLICEAI_RATE_LOCK      = threading.Lock()
_SPLICEAI_LAST_CALL_TIME = 0.0
_SPLICEAI_DELAY          = 1.0   # seconds between requests within this process


def _spliceai_rate_limit() -> None:
    global _SPLICEAI_LAST_CALL_TIME
    with _SPLICEAI_RATE_LOCK:
        elapsed = time.time() - _SPLICEAI_LAST_CALL_TIME
        wait = _SPLICEAI_DELAY - elapsed
        if wait > 0:
            time.sleep(wait)
        _SPLICEAI_LAST_CALL_TIME = time.time()


# Circuit breaker: the public SpliceAI-lookup API bans the caller's IP for an
# "extended period" (undocumented duration) on the first 429 — it is not a
# soft per-request throttle. Once tripped, every subsequent variant in this
# process skips the network call entirely for _SPLICEAI_BREAKER_COOLDOWN
# seconds instead of burning 4 retries x exponential backoff each, which
# would otherwise stall all 32 workers for ~15s apiece against a ban that
# retries cannot clear.
_SPLICEAI_BREAKER_LOCK       = threading.Lock()
_SPLICEAI_BREAKER_OPENED_AT  = 0.0
_SPLICEAI_BREAKER_COOLDOWN   = 600.0  # seconds before retrying the API again


def _breaker_is_open() -> bool:
    with _SPLICEAI_BREAKER_LOCK:
        return (time.time() - _SPLICEAI_BREAKER_OPENED_AT) < _SPLICEAI_BREAKER_COOLDOWN


def _breaker_trip() -> None:
    global _SPLICEAI_BREAKER_OPENED_AT
    with _SPLICEAI_BREAKER_LOCK:
        _SPLICEAI_BREAKER_OPENED_AT = time.time()


SCORE_KEYS   = ["DS_AG", "DS_AL", "DS_DG", "DS_DL"]
POS_KEYS     = ["DP_AG", "DP_AL", "DP_DG", "DP_DL"]
SCORE_LABELS = {
    "DS_AG": "Acceptor Gain",
    "DS_AL": "Acceptor Loss",
    "DS_DG": "Donor Gain",
    "DS_DL": "Donor Loss",
}

ABERRATION_LABELS = {
    "exon_skip":        "Exon skipping",
    "intron_retention": "Intron retention",
    "alt_donor":        "Cryptic donor site activation",
    "alt_acceptor":     "Cryptic acceptor site activation",
    "frameshift":       "Frameshift (predicted)",
}

# Variant types that cannot affect splicing — skip entirely.
# Synonymous/silent variants are intentionally NOT excluded: they can disrupt
# exonic splicing enhancers/silencers or create pseudo-splice sites, and
# SpliceAI scores them reliably.
_NO_SPLICE_TYPES = {
    "intergenic", "upstream", "downstream",
    "utr", "utr3", "utr5",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _max_ds(entry: dict) -> float:
    return max(_safe_float(entry.get(k, 0.0)) for k in SCORE_KEYS)


def _interpret(max_ds: float) -> str:
    if max_ds >= 0.8:
        return "HIGH splice impact (delta >= 0.8)"
    elif max_ds >= 0.5:
        return "MODERATE splice impact (delta >= 0.5)"
    elif max_ds >= 0.2:
        return "LOW splice impact (delta >= 0.2) — may warrant further review"
    else:
        return "NO predicted splice impact (delta < 0.2)"


def _pick_mane_select(scores: list) -> dict:
    """Return MANE Select transcript entry, or first entry as fallback."""
    for e in scores:
        if e.get("t_priority") == "MS":
            return e
    return scores[0]


def _build_variant_id(chrom: str, pos: str, ref: str, alt: str) -> str:
    """Build chrom-pos-ref-alt string expected by the SpliceAI API."""
    c = chrom.strip()
    if not c.startswith("chr"):
        c = f"chr{c}"
    return f"{c}-{pos.strip()}-{ref.strip()}-{alt.strip()}"


# ── Fetch + format ────────────────────────────────────────────────────────────

def fetch_and_format_spliceai(
    chrom: str,
    pos: str,
    ref: str,
    alt: str,
    hg: str = "hg19",
    distance: int = 500,
    mask: int = 0,
    timeout: int = 60,
) -> str | None:
    """
    Query the SpliceAI API and return a formatted evidence block,
    or None if no scores are returned (variant not splice-altering).

    Raises ToolFetchError on network failure, ToolParseError on bad response.
    """
    # Insertions/deletions with "-" as ref/alt (AutoPVS1 convention) are not
    # supported by SpliceAI — need at least one real nucleotide on each side
    if ref.strip() in ("-", "NA", "") or alt.strip() in ("-", "NA", ""):
        return None

    variant_id = _build_variant_id(chrom, pos, ref, alt)

    if _breaker_is_open():
        raise ToolFetchError(
            f"SpliceAI circuit breaker open (rate-limited by API earlier this run) — "
            f"skipping {variant_id} without calling the API"
        )

    hg_key     = "19" if hg in ("19", "37", "hg19", "hg37", "GRCh37") else "38"
    base_url   = SPLICEAI_API[hg_key]
    hg_param   = "37" if hg_key == "19" else "38"

    params = {
        "hg":       hg_param,
        "variant":  variant_id,
        "distance": distance,
        "mask":     mask,
        "gencode":  "basic",
    }

    print(f"[SpliceAI] Querying: {variant_id} (hg{hg_param})")

    try:
        _spliceai_rate_limit()
        r = requests.get(base_url, params=params, timeout=timeout)
        if r.status_code == 429:
            # Not a soft per-request throttle — Broad's API bans the caller's
            # IP outright ("loss of access ... for an extended period").
            # Retrying with backoff cannot clear this, so trip the breaker
            # immediately instead of burning 4 attempts.
            _breaker_trip()
            raise requests.exceptions.RequestException(
                f"SpliceAI returned 429 (rate limit banned) for {variant_id} — circuit breaker tripped"
            )
        r.raise_for_status()
    except requests.exceptions.Timeout as e:
        raise ToolFetchError(f"SpliceAI request timed out for {variant_id}") from e
    except requests.exceptions.RequestException as e:
        raise ToolFetchError(f"SpliceAI request failed for {variant_id}: {e}") from e

    try:
        data = r.json()
    except Exception as e:
        raise ToolParseError(f"Could not parse SpliceAI response for {variant_id}: {e}") from e

    # API-level error (e.g. unparseable variant)
    if "error" in data:
        raise ToolParseError(f"SpliceAI API error for {variant_id}: {data['error']}")

    scores = data.get("scores", [])
    if not scores:
        return None  # valid response — variant has no predicted splice impact

    entry  = _pick_mane_select(scores)
    gene   = entry.get("g_name") or "?"
    t_id   = entry.get("t_id") or "?"
    refseq = ", ".join(entry.get("t_refseq_ids") or []) or "—"
    n_tx   = len(scores)
    max_ds = _max_ds(entry)

    lines = [
        f"SPLICEAI SPLICE IMPACT SCORES ({variant_id}):",
        f"  Gene           : {gene}",
        f"  Transcript     : {t_id}  RefSeq: {refseq}",
        f"  (showing MANE Select out of {n_tx} transcripts)",
        f"  Max delta      : {max_ds:.4f}  ->  {_interpret(max_ds)}",
        f"  (Delta scores range 0–1; >= 0.2 = possible impact, >= 0.5 = moderate, >= 0.8 = high)",
        f"",
        f"  {'Type':<20} {'D_score':>8}  {'Position':>10}  {'REF':>8}  {'ALT':>8}",
        f"  {'-'*60}",
    ]

    for sk, pk in zip(SCORE_KEYS, POS_KEYS):
        label = SCORE_LABELS[sk]
        ds    = _safe_float(entry.get(sk, 0.0))
        dp    = _safe_int(entry.get(pk), default="?")
        ref_s = _safe_float(entry.get(sk + "_REF", 0.0))
        alt_s = _safe_float(entry.get(sk + "_ALT", 0.0))
        flag  = "  <--" if ds >= 0.2 else ""
        lines.append(
            f"  {label:<20} {ds:>8.4f}  {str(dp):>10}  {ref_s:>8.4f}  {alt_s:>8.4f}{flag}"
        )

    # Clinical interpretation block (sai10kPredictions)
    sai10k = data.get("sai10kPredictions", {})
    aberrations = sai10k.get("aberrations", [])
    if aberrations:
        lines.append("")
        lines.append("  Predicted aberrations:")
        for ab in aberrations:
            ab_type = ABERRATION_LABELS.get(
                ab.get("aberration_type", ""),
                ab.get("aberration_type", "?").replace("_", " ").title()
            )
            conf    = ab.get("confidence", "?")
            ds_ab   = _safe_float(ab.get("max_delta_score", 0))
            fs_desc = ab.get("frameshift_description", "")
            region  = ab.get("affected_region", {})
            intron  = region.get("region_number", "?")
            dist    = region.get("distance_to_boundary", "?")
            bound   = region.get("nearest_boundary", "?")
            lines.append(f"    [{ab_type}]  confidence={conf}  max_ds={ds_ab:.2f}")
            lines.append(f"      Intron {intron}, {dist}bp from {bound}")
            if fs_desc:
                lines.append(f"      {fs_desc}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE TOOL CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SpliceAITool(NetworkTool):
    """
    Fetches SpliceAI splice impact scores from the Broad Institute API.

    Uses the same coordinate extraction as AutoPVS1Tool (parse_variant_coords).
    Only runs on variants with resolvable genomic coordinates and skips
    variant types that cannot affect splicing (synonymous, intergenic, UTR).

    Output includes MANE Select delta scores (DS_AG/AL/DG/DL), REF/ALT
    raw scores, and the sai10kPredictions clinical interpretation block
    (aberration type, frameshift description, confidence).
    """

    name        = "spliceai"
    description = "Fetches SpliceAI splice impact scores from the Broad Institute API."

    distance: int = 500
    mask: int = 0

    def gate(self, variant: dict, context: ToolContext) -> bool:
        # Skip variant types that cannot have splice impact
        type_val = variant.get("Type", "").lower().strip()
        if any(t in type_val for t in _NO_SPLICE_TYPES):
            print(f"[SpliceAI gate] SKIP — type: {type_val!r}")
            return False

        # Require resolvable coordinates
        try:
            parse_variant_coords(
                variant_str=variant.get("Variant", ""),
                chrom_field=variant.get("Chromosome", ""),
                pos_field=variant.get("Position", ""),
                ref_field=variant.get("Ref_seq", ""),
                alt_field=variant.get("Var_seq", ""),
            )
        except ValueError:
            print(f"[SpliceAI gate] SKIP — cannot resolve coordinates")
            return False

        return True

    def run(self, variant: dict, context: ToolContext) -> str | None:
        try:
            coords = parse_variant_coords(
                variant_str=variant.get("Variant", ""),
                chrom_field=variant.get("Chromosome", ""),
                pos_field=variant.get("Position", ""),
                ref_field=variant.get("Ref_seq", ""),
                alt_field=variant.get("Var_seq", ""),
            )
        except ValueError:
            return None

        chrom, pos, ref, alt = coords

        return fetch_and_format_spliceai(
            chrom, pos, ref, alt,
            hg=context.genome_build,
            distance=self.distance,
            mask=self.mask,
            timeout=self.timeout,
        )

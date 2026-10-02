"""
pipeline/tools/websearch.py — shared NCBI E-utilities helpers.

Contains _ncbi_get (process-wide rate-limited GET with retry) and _clean_xml_text,
imported by the NCBI-backed tools.
"""

import os
import re
import time
import threading
import logging
import requests


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

NCBI_BASE      = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Set NCBI_API_KEY env var (free key from https://www.ncbi.nlm.nih.gov/account/)
# to raise the anonymous 3 req/s cap to 10 req/s — same key litvar2.py already
# supports. This module previously ignored it entirely, hardcoding every NCBI
# call in this file (ClinGen allele resolution, ClinVar gene/variant stats) to
# the anonymous rate regardless of whether a key was configured.
NCBI_API_KEY   = os.environ.get("NCBI_API_KEY", "")
# NCBI_MAX_RPS env var caps this process's E-utilities request rate (default: 10 req/s
# with a key, 3 without — NCBI's own limits). NCBI enforces the limit per key, so when
# several servers share one key, give each a share (e.g. 4 servers → NCBI_MAX_RPS=2.5).
NCBI_MAX_RPS   = float(os.environ.get("NCBI_MAX_RPS") or (10 if NCBI_API_KEY else 3))
if NCBI_MAX_RPS <= 0:
    raise ValueError(f"NCBI_MAX_RPS must be > 0, got {NCBI_MAX_RPS}")
NCBI_MIN_DELAY = 1.0 / NCBI_MAX_RPS
DEFAULT_TIMEOUT  = 15          # seconds
DEFAULT_MAX_CHARS = 3000       # safe default — main.py can override per tool instance

# Global rate limiter — enforces the minimum interval per process across all threads
# and all NCBI-calling tools (litvar2.py uses it too). With 32 workers, per-thread
# sleeps are ineffective: all threads sleep independently and fire simultaneously.
# This lock serialises the request slots instead.
_NCBI_WS_RATE_LOCK      = threading.Lock()
_NCBI_WS_LAST_CALL_TIME = 0.0


def _ncbi_ws_rate_limit() -> None:
    global _NCBI_WS_LAST_CALL_TIME
    with _NCBI_WS_RATE_LOCK:
        elapsed = time.time() - _NCBI_WS_LAST_CALL_TIME
        wait = NCBI_MIN_DELAY - elapsed
        if wait > 0:
            time.sleep(wait)
        _NCBI_WS_LAST_CALL_TIME = time.time()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _ncbi_get(endpoint: str, params: dict, timeout: int = DEFAULT_TIMEOUT) -> requests.Response:
    """
    Wrapper around NCBI E-utilities GET requests.
    Uses a global rate limiter (not per-thread sleep) to stay under the
    per-process request-rate cap (3 req/s anonymous, 10 req/s with
    NCBI_API_KEY set — see NCBI_MIN_DELAY above). Attaches api_key
    automatically when configured, same as litvar2.py already does.
    Retries up to 4 times with exponential backoff on 429 AND on transient
    network failures (timeout, connection error, 5xx) — a real past failure:
    this previously retried ONLY on 429, so a plain timeout or connection
    reset (which is common under the multi-process load this pipeline
    generates against NCBI — several server processes on this cluster share
    one outbound IP, each independently rate-limiting itself but combining
    to exceed NCBI's real server-side limit) raised immediately with zero
    retry, and callers that swallow exceptions locally (e.g.
    clinvar_gene_stats.py's _resolve_variation_id) then reported a
    definitive-sounding "not resolvable" for a variant that was never
    actually queried.
    """
    url = f"{NCBI_BASE}/{endpoint}"
    if NCBI_API_KEY:
        params = {**params, "api_key": NCBI_API_KEY}
    logger.debug("NCBI GET %s params=%s", endpoint, params)
    last_exc: Exception | None = None
    for attempt in range(4):
        _ncbi_ws_rate_limit()
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            wait = 2.0 ** attempt
            logger.warning(
                "NCBI request failed (%s) — retrying in %.0fs (attempt %d/4)",
                type(e).__name__, wait, attempt + 1,
            )
            time.sleep(wait)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 2.0 ** attempt
            logger.warning(
                "NCBI %d response — backing off %.0fs (attempt %d/4)",
                resp.status_code, wait, attempt + 1,
            )
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    if last_exc is not None:
        raise last_exc
    resp.raise_for_status()
    return resp


def _clean_xml_text(raw: str) -> str:
    """Strip XML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", text).strip()

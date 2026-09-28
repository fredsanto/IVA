"""
pipeline/tools/websearch.py — Web search and fetch tools for the ReAct agent.

Contains:
  - Shared helpers used by NCBI and web tools
  - WebSearchTool  — DuckDuckGo discovery (always the first ReAct step)
  - WebFetchTool   — Generic HTML extraction for non-NCBI URLs

Note: These tools are sub-tools consumed by WebSearchAgentTool's internal ReActAgent.
They are NOT pipeline-level tools (they do not implement the run(variant, context)
contract). They inherit from NetworkTool for the HTTP helpers and the shared
name/description fields, but override run(query: str) for ReActAgent compatibility.
"""

import os
import re
import time
import threading
import logging
import requests
import urllib.request
import xml.etree.ElementTree as ET

from ddgs import DDGS
from pipeline.tools.base import NetworkTool
from urllib.parse import urlparse

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
NCBI_MIN_DELAY = 0.11 if NCBI_API_KEY else 0.34   # 10 req/s with key, else 3 req/s
DEFAULT_TIMEOUT  = 15          # seconds
DEFAULT_MAX_CHARS = 3000       # safe default — main.py can override per tool instance

# Global rate limiters — enforce minimum intervals per process across all threads.
# With 32 workers, per-thread sleeps are ineffective: all threads sleep independently
# and fire simultaneously.  These locks serialise the request slots instead.
_NCBI_WS_RATE_LOCK      = threading.Lock()
_NCBI_WS_LAST_CALL_TIME = 0.0

_DDG_RATE_LOCK      = threading.Lock()
_DDG_LAST_CALL_TIME = 0.0
_DDG_DELAY          = 2.0   # DuckDuckGo: 1 search per 2s per process


def _ncbi_ws_rate_limit() -> None:
    global _NCBI_WS_LAST_CALL_TIME
    with _NCBI_WS_RATE_LOCK:
        elapsed = time.time() - _NCBI_WS_LAST_CALL_TIME
        wait = NCBI_MIN_DELAY - elapsed
        if wait > 0:
            time.sleep(wait)
        _NCBI_WS_LAST_CALL_TIME = time.time()


def _ddg_rate_limit() -> None:
    global _DDG_LAST_CALL_TIME
    with _DDG_RATE_LOCK:
        elapsed = time.time() - _DDG_LAST_CALL_TIME
        wait = _DDG_DELAY - elapsed
        if wait > 0:
            time.sleep(wait)
        _DDG_LAST_CALL_TIME = time.time()

# Country-code TLDs whose content is predominantly non-English
_NON_ENGLISH_TLDS = {
    ".ru", ".ua", ".by", ".kz",          # Russian-speaking
    ".ar", ".ir",                          # Arabic / Farsi
    ".cn", ".jp", ".kr", ".vn",           # CJK + Vietnamese
    ".pl", ".cz", ".sk", ".hu", ".ro",   # Eastern European
    ".bg", ".rs", ".hr",                   # Balkan
    ".tr",                                 # Turkish
}

NON_ASCII_THRESHOLD = 0.3     # reject page if >30 % of chars are non-ASCII


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


def _is_likely_english(url: str) -> bool:
    """Return False for URLs whose TLD indicates a non-English-primary domain."""
    try:
        hostname = urlparse(url).hostname or ""
        return not any(hostname.endswith(tld) for tld in _NON_ENGLISH_TLDS)
    except Exception:
        return True


def _extract_body_text(html: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """
    Extract readable body text from HTML using xml.etree (fast, no deps).
    Falls back to regex stripping if the HTML is malformed.
    Skips non-content tags: script, style, nav, footer, header, aside, menu.
    Returns empty string if nothing useful is found.
    """
    SKIP = {"script", "style", "nav", "footer", "header", "aside", "menu"}

    # Fast path: try stdlib XML parser on well-formed pages
    try:
        # wrap in a root tag to handle fragment HTML
        root = ET.fromstring(f"<root>{html}</root>")
        parts = []
        for elem in root.iter():
            tag = elem.tag.lower() if isinstance(elem.tag, str) else ""
            if tag in SKIP:
                continue
            if elem.text and elem.text.strip():
                parts.append(elem.text.strip())
            if elem.tail and elem.tail.strip():
                parts.append(elem.tail.strip())
        text = " ".join(parts)
    except ET.ParseError:
        # Fallback: brute-force tag stripping
        text = re.sub(r"<(script|style|nav|footer|header|aside|menu)[^>]*>.*?</\1>",
                      " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 100:
        logger.warning("Extracted text is suspiciously short (%d chars) — page may be JS-rendered or empty", len(text))

    return text[:max_chars]


# ─────────────────────────────────────────────
# TOOL 1 — WEB SEARCH
# ─────────────────────────────────────────────

class WebSearchTool(NetworkTool):
    """
    Executes a DuckDuckGo web search and returns structured result snippets.
    Always use this tool FIRST for any research question.
    Follow up with ncbi_fetch (for NCBI URLs) or web_fetch (for everything else).
    """
    name        = "web_search"
    description = (
        "Search the web for a query and return structured snippets with titles, "
        "summaries, and source URLs. Always use this first to discover candidate pages. "
        "Then use ncbi_fetch or web_fetch to read the most relevant results in full."
    )
    input_system = (
        "You are a biomedical search query optimizer. "
        "Output ONLY 2-5 keywords that best retrieve relevant web results.\n\n"
        "RULES:\n"
        "- For variant-specific searches (ClinVar, functional data, case reports): "
        "include the gene symbol AND the HGVS notation (e.g. GENE_X c.100+5G>A) or "
        "protein change (e.g. GENE_X p.Lys34Thr). HGVS notation gives precise results "
        "on ClinVar and literature databases.\n"
        "- For gene-disease searches: use gene symbol + disease or phenotype name.\n"
        "- NEVER use rsIDs (rs...) — they return poor results in general web searches.\n"
        "- NEVER use chromosomal coordinates or transcript IDs (NM_...).\n"
        "- No quotes, no punctuation, no explanation, no tool names.\n\n"
        "Example outputs:\n"
        "  GENE_X c.100+5G>A ClinVar\n"
        "  GENE_X obesity intellectual disability"
    )
    input_template = "Convert to a plain keyword search query: {user_input}"

    # Matches tool-call wrapping: tool_name("actual content") or tool_name('actual content')
    _TOOL_CALL_RE = re.compile(r"^\w+\s*\(\s*[\"']?(.*?)[\"']?\s*\)\s*$", re.DOTALL)

    def __init__(self, max_results: int = 6):
        self.max_results = max_results

    def run(self, query: str) -> str:
        query = query.strip()

        # Defensive: strip accidental tool-call wrapping e.g. web_fetch("KSR2 obesity")
        m = self._TOOL_CALL_RE.match(query)
        if m:
            logger.warning("WebSearchTool received tool-call syntax — stripping wrapper: %r", query)
            query = m.group(1).strip()

        if not query:
            return "Error: empty query."

        logger.debug("WebSearchTool query=%r max_results=%d", query, self.max_results)

        try:
            _ddg_rate_limit()
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results, region="us-en"))
        except Exception as e:
            logger.error("DuckDuckGo search failed: %s", e)
            return f"Search error: {e}"

        if not results:
            return f"No results found for: {query!r}"

        # Drop results from predominantly non-English domains
        results = [r for r in results if _is_likely_english(r.get("href", ""))]
        if not results:
            return f"No English-language results found for: {query!r}"

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r['title']}\n"
                f"    {r['body']}\n"
                f"    URL: {r['href']}"
            )
        return "\n\n".join(lines)


# ─────────────────────────────────────────────
# TOOL 3 — WEB FETCH
# ─────────────────────────────────────────────

class WebFetchTool(NetworkTool):
    """
    Fetches a URL and returns its readable text content.
    Use for any non-NCBI URL discovered via web_search.
    For ncbi.nlm.nih.gov URLs, use ncbi_fetch instead.
    """
    name        = "web_fetch"
    description = (
        "Retrieves and extracts readable text from any non-NCBI web page. "
        "Use this to read pages discovered by web_search in full. "
        "Do NOT use this for ncbi.nlm.nih.gov URLs — use ncbi_fetch instead."
    )
    input_system = (
        "You are a URL extractor. "
        "Extract the single most relevant non-NCBI URL from the context. "
        "Reject any URL containing ncbi.nlm.nih.gov. "
        "NEVER include tool names, function calls, parentheses, or code syntax. "
        "Output ONLY the bare URL string. No explanation, no punctuation."
    )
    input_template = "Extract the most relevant non-NCBI URL from: {user_input}"

    _NCBI_RE = re.compile(r"ncbi\.nlm\.nih\.gov")

    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS, timeout: int = DEFAULT_TIMEOUT):
        self.max_chars = max_chars
        self.timeout   = timeout

    def run(self, url: str) -> str:
        url = url.strip()

        if not url.startswith("http"):
            return "Error: URL must start with http or https."

        if self._NCBI_RE.search(url):
            return "Error: use ncbi_fetch for ncbi.nlm.nih.gov URLs."

        logger.debug("WebFetchTool url=%s", url)

        try:
            parsed = urlparse(url)
            if not parsed.hostname:
                return "Error: could not parse hostname from URL."

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content_type = resp.headers.get_content_type()
                if "html" not in content_type:
                    return f"Unsupported content type: {content_type}. Only HTML pages are supported."
                html = resp.read().decode("utf-8", errors="ignore")

        except urllib.error.HTTPError as e:
            return f"HTTP {e.code} error fetching {url}: {e.reason}"
        except urllib.error.URLError as e:
            return f"URL error fetching {url}: {e.reason}"
        except Exception as e:
            logger.error("Unexpected fetch error for %s: %s", url, e)
            return f"Unexpected error fetching {url}: {e}"

        text = _extract_body_text(html, self.max_chars)

        if not text:
            return "Page fetched but no readable content could be extracted (page may be JS-rendered)."

        non_ascii = sum(1 for c in text if ord(c) > 127)
        if len(text) > 100 and non_ascii / len(text) > NON_ASCII_THRESHOLD:
            return f"Non-English page detected at {url} — skipping."

        hostname = parsed.hostname or url
        return f"Content from {hostname}:\n\n{text}"

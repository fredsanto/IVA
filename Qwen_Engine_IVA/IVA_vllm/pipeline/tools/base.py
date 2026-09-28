"""
pipeline/tools/base.py — Base classes for all pipeline tools.

Hierarchy:
    Tool          — pure deterministic logic, no network, no SLM
    NetworkTool   — HTTP fetch helpers (_get, _post); no SLM
    SLMTool       — may call context.llm.generate()
    ReActTool     — full ReAct loop with an internal tool registry
    RAGTool       — vector DB retrieval via context.db
    BotTool       — browser automation via context.browser

All pipeline-facing tools must implement:
    gate(variant, context) -> bool     (optional — default: always run)
    run(variant, context)  -> str | None

Note: ReAct sub-tools (WebSearchTool, NCBIFetchTool, WebFetchTool) also inherit
from NetworkTool for the HTTP helpers, but they keep a different run(query: str)
signature because they are used internally by ReActAgent, not by the pipeline executor.
"""

from __future__ import annotations

import requests
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.core.context import ToolContext


class Tool:
    """
    Base class for all pipeline tools.

    Subclasses must set:
        name        — unique identifier matching the YAML manifest name
        description — one sentence, shown in logs and TOOL_DEVELOPMENT.md
    """

    name: str = ""
    description: str = ""

    def gate(self, variant: dict, context: "ToolContext") -> bool:
        """
        Return False to skip this tool for this variant.
        Default: always run.
        Override in subclasses for conditional execution.
        """
        return True

    def run(self, variant: dict, context: "ToolContext") -> str | None:
        """
        Execute the tool for one variant.

        Returns a string block to include in the mini-doc, or None if no result.
        Must raise a PipelineError subclass on failure — never return an error string.
        """
        raise NotImplementedError


class NetworkTool(Tool):
    """
    Base class for tools that make HTTP requests.

    Provides _get() and _post() convenience wrappers that enforce the per-tool
    timeout and call raise_for_status() automatically.
    """

    timeout: int = 15

    def _get(self, url: str, timeout: int | None = None, **kwargs) -> requests.Response:
        """GET request with automatic raise_for_status."""
        t = timeout if timeout is not None else self.timeout
        resp = requests.get(url, timeout=t, **kwargs)
        resp.raise_for_status()
        return resp

    def _post(self, url: str, timeout: int | None = None, **kwargs) -> requests.Response:
        """POST request with automatic raise_for_status."""
        t = timeout if timeout is not None else self.timeout
        resp = requests.post(url, timeout=t, **kwargs)
        resp.raise_for_status()
        return resp


class SLMTool(NetworkTool):
    """
    Base class for tools that call the SLM via context.llm.generate().
    Inherits network helpers from NetworkTool.
    """
    pass


class ReActTool(SLMTool):
    """
    Base class for tools that run a full ReAct loop with an internal tool registry.
    The loop is implemented in the concrete subclass's run() method.
    """
    pass


class RAGTool(SLMTool):
    """
    Base class for tools that retrieve from a vector database.
    context.db is a VectorStore instance (Qdrant by default).

    VectorStore interface:
        context.db.search(query, collection, top_k, filters) -> list[Chunk]
        context.db.collections() -> list[str]

    Chunk fields: text, score, metadata (dict with source, gene, rsid, etc.)

    Always pass gene and/or rsid as filters to avoid semantic drift on short identifiers.
    """
    pass


class BotTool(Tool):
    """
    Base class for tools that use browser automation.
    context.browser is a BrowserSession instance.

    BrowserSession interface:
        context.browser.get(url)               -> str   (returns page text)
        context.browser.click(selector)
        context.browser.fill(selector, value)
        context.browser.wait_for(selector, timeout)

    Underlying implementation (Playwright or Selenium) is injected at startup.
    BotTool subclasses must never import Playwright or Selenium directly.
    """
    pass

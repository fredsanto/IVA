"""
pipeline/core/context.py — ToolContext dataclass.

Every tool receives a ToolContext as its second argument.
This is the only way tools access shared resources (LLM, DB, browser).
Tools must never instantiate models or open connections directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient


@dataclass
class ToolContext:
    # ── Variant being processed ───────────────────────────────────────────────
    variant: dict                        # normalized variant dict (canonical schema)

    # ── Patient information ───────────────────────────────────────────────────
    patient_phenotype: str               # free-text phenotype from user

    # ── Pipeline state ────────────────────────────────────────────────────────
    all_outputs: dict[str, str | None]   # {tool_name: output} for tools that ran before this one
    variant_report: str                  # accumulated mini-doc so far (for compression tools)
    variant_index: int                   # 0-based index of this variant in the run
    total_variants: int                  # total number of variants in this run

    # ── Shared resources (injected at startup, never instantiated by tools) ───
    llm: "LLMClient"                     # call context.llm.generate(system, user, max_tokens)
    db: Any | None = None                # VectorStore | None — None if no DB configured
    browser: Any | None = None           # BrowserSession | None — None if no bot configured

    # ── Run-level metadata ────────────────────────────────────────────────────
    genome_build: str = "hg19"
    raw_fields: dict = field(default_factory=dict)

    # ── Convenience ───────────────────────────────────────────────────────────

    def field(self, name: str) -> str:
        """Extract a field from the variant dict. Returns 'NA' if missing or falsy."""
        return self.variant.get(name, "NA") or "NA"

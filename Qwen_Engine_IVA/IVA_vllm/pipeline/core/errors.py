"""
pipeline/core/errors.py — Typed pipeline exceptions.

All pipeline errors inherit from PipelineError.
Tools must raise these; they must never return error strings or raise bare Exception.
"""


class PipelineError(Exception):
    """Base class for all pipeline errors."""


class ToolFetchError(PipelineError):
    """Network request failed. Include URL and status code."""


class ToolParseError(PipelineError):
    """Could not parse response from external source."""


class ToolGateError(PipelineError):
    """Gate evaluation failed (not a skip — an actual error)."""


class SLMError(PipelineError):
    """SLM call failed or returned unparseable output."""


class ManifestError(PipelineError):
    """Manifest file is missing, malformed, or references unknown tool."""


class NormalizationError(PipelineError):
    """Input CSV/Excel could not be parsed or normalized."""

"""
pipeline/core/manifest.py — Manifest loading, validation, and gate resolution.

Loads all enabled YAML manifests from pipeline/manifests/.
Manifests declare metadata, gate conditions, execution order, and compression config
for each pipeline tool.

Public API:
    ManifestLoader          — load all / one manifest from the manifests/ directory
    eval_manifest_gate()    — evaluate a manifest gate expression against a variant dict
                              Returns True (run), False (skip), or None (SLM gate — caller handles)
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from pipeline.core.errors import ManifestError

if TYPE_CHECKING:
    from pipeline.core.context import ToolContext


# Path resolved relative to this file so it works regardless of cwd
MANIFESTS_DIR = Path(__file__).parent.parent / "manifests"

REQUIRED_FIELDS = {"name", "enabled"}

VALID_OPERATORS = {
    "equals", "not_equals", "in", "not_in", "not_na", "na", "slm",
    "raw_not_na", "raw_na",      # check context.raw_fields
    "context_equals",            # check a ToolContext attribute by name
}


class ManifestLoader:
    """
    Loads and validates tool manifests from the manifests/ directory.

    Usage:
        loader = ManifestLoader()
        manifests = loader.load_all()   # sorted list of enabled manifest dicts
        m = loader.load_one("autopvs1") # single manifest by tool name
    """

    def __init__(self, manifests_dir: Optional[Path] = None):
        self._dir = manifests_dir or MANIFESTS_DIR

    # ── public ────────────────────────────────────────────────────────────────

    def load_all(self) -> list[dict]:
        """
        Load every enabled manifest from the manifests directory.

        Returns a list of manifest dicts sorted by:
          1. 'order' field (ascending, default 99 if absent)
          2. 'name' field (alphabetical, for deterministic tie-breaking)

        Raises ManifestError if:
          - The directory does not exist
          - No YAML files are found
          - Any file is malformed or missing required fields
        """
        if not self._dir.exists():
            raise ManifestError(f"Manifests directory not found: {self._dir}")

        yaml_files = sorted(self._dir.glob("*.yaml"))
        if not yaml_files:
            raise ManifestError(f"No YAML manifests found in {self._dir}")

        manifests: list[dict] = []
        for path in yaml_files:
            try:
                manifest = self._load_one(path)
            except ManifestError:
                raise
            except Exception as e:
                raise ManifestError(
                    f"Failed to load manifest {path.name}: {e}"
                ) from e

            if not manifest.get("enabled", True):
                continue

            manifests.append(manifest)

        manifests.sort(key=lambda m: (m.get("order", 99), m.get("name", "")))
        return manifests

    def load_one(self, name: str) -> dict:
        """
        Load a single manifest by tool name (looks for <name>.yaml).

        Raises ManifestError if the file does not exist or is invalid.
        """
        path = self._dir / f"{name}.yaml"
        if not path.exists():
            raise ManifestError(f"Manifest file not found: {path}")
        return self._load_one(path)

    # ── private ───────────────────────────────────────────────────────────────

    def _load_one(self, path: Path) -> dict:
        """Load and validate a single YAML file. Raises ManifestError on any problem."""
        with path.open(encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ManifestError(
                    f"YAML parse error in {path.name}: {e}"
                ) from e

        if not isinstance(data, dict):
            raise ManifestError(
                f"Manifest {path.name} must be a YAML mapping, got {type(data).__name__}."
            )

        for field in REQUIRED_FIELDS:
            if field not in data:
                raise ManifestError(
                    f"Manifest {path.name} is missing required field: {field!r}"
                )

        # Validate gate config if present
        gate = data.get("gate")
        if gate is not None and isinstance(gate, dict):
            operator = gate.get("operator")
            if operator is not None and operator not in VALID_OPERATORS:
                raise ManifestError(
                    f"Manifest {path.name}: invalid gate operator {operator!r}. "
                    f"Valid operators: {sorted(VALID_OPERATORS)}"
                )

        return data


# ── Gate evaluation ───────────────────────────────────────────────────────────


def eval_manifest_gate(
    manifest: dict,
    variant: dict,
    context: "ToolContext | None" = None,
) -> Optional[bool]:
    """
    Evaluate the manifest-level gate expression against a variant dict.

    Args:
        manifest: A manifest dict loaded by ManifestLoader.
        variant:  A normalized variant dict (canonical schema).

    Returns:
        True   — gate passes; tool should run
        False  — gate fails; tool should be skipped
        None   — SLM gate; the executor must call the LLM to decide

    Raises ManifestError if the gate config is structurally invalid.
    """
    gate = manifest.get("gate")

    # No gate → always run
    if gate is None:
        return True

    # Null YAML value (gate: ~) → always run
    if not isinstance(gate, dict):
        return True

    # SLM gate — executor must call LLM
    if gate.get("type") == "slm" or gate.get("operator") == "slm":
        return None

    field_name = gate.get("field")
    operator   = gate.get("operator")
    value      = gate.get("value")

    if not field_name or not operator:
        raise ManifestError(
            f"Manifest gate for {manifest.get('name')!r} requires both "
            f"'field' and 'operator' keys."
        )

    field_val = variant.get(field_name, "NA")

    if operator == "not_na":
        return field_val not in (None, "NA", "", "na", "Na")

    if operator == "na":
        return field_val in (None, "NA", "", "na", "Na")

    if operator == "equals":
        return str(field_val) == str(value)

    if operator == "not_equals":
        return str(field_val) != str(value)

    if operator == "in":
        return str(field_val) in [str(v) for v in (value or [])]

    if operator == "not_in":
        return str(field_val) not in [str(v) for v in (value or [])]

    if operator == "raw_not_na":
        if context is None:
            return True  # no context — default to run
        val = context.raw_fields.get(field_name, "NA")
        return val not in (None, "NA", "", "na", "Na", ".", "nan")

    if operator == "raw_na":
        if context is None:
            return False  # no context — default to skip
        val = context.raw_fields.get(field_name, "NA")
        return val in (None, "NA", "", "na", "Na", ".", "nan")

    if operator == "context_equals":
        if context is None:
            return True
        ctx_val = getattr(context, field_name, None)
        return str(ctx_val) == str(value)

    raise ManifestError(
        f"Unknown gate operator {operator!r} in manifest {manifest.get('name')!r}"
    )

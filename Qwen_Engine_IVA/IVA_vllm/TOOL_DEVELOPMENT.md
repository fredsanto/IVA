# TOOL_DEVELOPMENT.md — Writing New Pipeline Tools

This guide explains how to add a new evidence-gathering tool to the variant
analysis pipeline. Read CLAUDE.md for architecture context first.

---

## What a tool is

A tool is a self-contained unit that receives one normalized variant (as a dict)
plus a `ToolContext` (shared resources), performs some work, and returns a
string block — or `None` if it has nothing to contribute.

The framework handles everything else: ordering, gating, compression, error
catching, and mini-doc assembly. You do not touch any of that.

---

## Tool base classes

Choose the one that matches your tool's requirements:

| Class | Use when |
|---|---|
| `Tool` | Pure deterministic logic, no network, no SLM |
| `NetworkTool` | HTTP fetch, no SLM |
| `SLMTool` | Calls the SLM for summarization, filtering, or judgment |
| `ReActTool` | Full ReAct loop with tool registry |
| `RAGTool` | Vector DB retrieval via `context.db` |
| `BotTool` | Browser automation via `context.browser` |

All inherit from `Tool`. Subclasses add capabilities — they do not change the
`gate()` / `run()` contract.

---

## Minimum implementation

```python
# pipeline/tools/my_tool.py
from pipeline.tools.base import SLMTool          # or Tool, NetworkTool, etc.
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError


class MyTool(SLMTool):
    name        = "my_tool"              # must match manifest name field
    description = "One sentence."        # shown in logs

    def gate(self, variant: dict, context: ToolContext) -> bool:
        """Return False to skip for this variant. Default: always run."""
        return context.field("RS_ID") != "NA"

    def run(self, variant: dict, context: ToolContext) -> str | None:
        rsid = context.field("RS_ID")

        # Network call
        try:
            response = self._get(f"https://example.com/api/{rsid}", timeout=self.timeout)
        except Exception as e:
            raise ToolFetchError(f"Failed to fetch {rsid}: {e}") from e

        # Parse
        try:
            data = response.json()
        except Exception as e:
            raise ToolParseError(f"Could not parse response for {rsid}: {e}") from e

        if not data:
            return None

        # Optional SLM call (only in SLMTool subclasses)
        summary = context.llm.generate(
            system="Summarize this evidence in 2 sentences.",
            user=str(data),
            max_tokens=200,
        )

        return f"MY_TOOL EVIDENCE ({rsid}):\n{summary}"
```

---

## Tool contract — rules that must be respected

1. **Stateless.** All state lives in `ToolContext`. Never store results on `self`
   between calls — a new context is passed for each variant.

2. **Never load models.** Always use `context.llm.generate()`. Never import
   torch, transformers, or vllm directly in a tool file.

3. **Never open DB connections.** Use `context.db` (available when configured).

4. **Raise typed exceptions.** Always raise from `pipeline/core/errors.py`.
   Never return an error string. Never raise bare `Exception`.

5. **Prompts in files.** Multi-line SLM prompts belong in `prompts/`.
   Load with `(Path(__file__).parent.parent.parent / "prompts" / "my_prompt.txt").read_text()`.
   This anchors the path to the source file location, not the working directory. Do not hardcode.

6. **Return free-form string or None.** No structured output required.
   The SLM handles synthesis in stages 3 and 4.

7. **`enable_thinking` is always False.** Never pass `enable_thinking=True`
   to `context.llm.generate()`. Enforced at the registry level too.

---

## ToolContext — what you have access to

```python
context.variant          # dict: the canonical variant (all TARGET_COLUMNS)
context.patient_phenotype # str: free-text phenotype from the request
context.all_outputs      # dict: {tool_name: output} for tools that ran before this one
context.variant_report   # str: accumulated mini-doc for compression tools
context.variant_index    # int: 0-based index of this variant
context.total_variants   # int: total variants in this run
context.llm              # LLMClient: call .generate(system, user, max_tokens)
context.db               # VectorStore | None
context.browser          # BrowserSession | None

# Convenience accessor
context.field("RS_ID")   # returns "NA" if missing or falsy
```

---

## Error types

Import from `pipeline.core.errors`:

| Exception | When to raise |
|---|---|
| `ToolFetchError` | Network request failed (include URL + status) |
| `ToolParseError` | Could not parse external source response |
| `ToolGateError` | Gate evaluation itself errored |
| `SLMError` | SLM call failed or returned unparseable output |
| `ManifestError` | Manifest file missing, malformed, or invalid |
| `NormalizationError` | Input CSV/Excel cannot be parsed |

The executor catches all `PipelineError` subclasses per tool, logs the error,
and writes a failure note into the mini-doc so the SLM can reason from it.

---

## Manifest

Create `pipeline/manifests/my_tool.yaml`:

```yaml
name: my_tool            # must match Tool.name
description: "One sentence."
enabled: true

order: 3                 # lower = runs earlier; ties broken alphabetically
parallel: true           # run concurrently with other tools at same order level

gate:                    # optional — omit to always run
  field: RS_ID
  operator: not_na       # operators: equals, not_equals, in, not_in, not_na, na, slm

compress:                # optional per-tool output compression
  enabled: true
  threshold: auto        # compress when total_variants > this (auto = 5)
  strategy: slm          # slm | truncate | first_n_lines
  max_tokens: 200

timeout: 15              # seconds before ToolFetchError is raised
retry:
  attempts: 2
  backoff: 2.0           # seconds between retries
```

### Gate operators

| Operator | Meaning |
|---|---|
| `not_na` | field value is not `"NA"` / empty |
| `na` | field value is `"NA"` / empty |
| `equals` | field value == value |
| `not_equals` | field value != value |
| `in` | field value is in the list |
| `not_in` | field value is not in the list |
| `slm` | ask the SLM — provide a `prompt:` string |

The Python `gate()` method always takes precedence over the manifest gate when
both are defined.

### Execution order

Tools at the same `order` share the same execution wave. Within a wave, tools
run concurrently if `parallel: true`. Waves run sequentially (order 1 completes
before order 2 starts).

Use this to express dependencies: a tool at order 3 sees the outputs of all
order 1 and order 2 tools via `context.all_outputs`.

---

## Registering the tool

Add the import to `pipeline/tools/__init__.py`:

```python
from pipeline.tools.my_tool import MyTool
__all__ = [..., "MyTool"]
```

Add the instance to the `self._tools` list in `pipeline/pipeline.py`:

```python
self._tools = [
    LitVar2SummaryTool(),
    AutoPVS1Tool(),
    WebSearchAgentTool(),
    MyTool(),            # ← add here, in execution order
]
```

No other files need to change.

---

## Interface-ready base classes (no concrete implementation yet)

### RAGTool

```python
class RAGTool(SLMTool):
    """
    context.db is a VectorStore instance (Qdrant by default).

    VectorStore interface:
        context.db.search(query: str, collection: str, top_k: int, filters: dict) -> list[Chunk]
        context.db.collections() -> list[str]

    Chunk fields: text, score, metadata (dict with source, gene, rsid, etc.)

    Always pass gene and/or rsid as filters to avoid semantic drift on short identifiers.
    """
```

### BotTool

```python
class BotTool(Tool):
    """
    context.browser is a BrowserSession instance.

    BrowserSession interface:
        context.browser.get(url: str) -> str          # returns page text
        context.browser.click(selector: str)
        context.browser.fill(selector: str, value: str)
        context.browser.wait_for(selector: str, timeout: int)

    Never import Playwright or Selenium directly in a BotTool.
    """
```

---

## Testing a new tool standalone

Every tool file should have an `if __name__ == "__main__"` block:

```python
if __name__ == "__main__":
    from pipeline.core.context import ToolContext
    from pipeline.llm.vllm_client import VLLMClient

    llm   = VLLMClient(base_url="http://localhost:8001")
    tool  = MyTool()
    ctx   = ToolContext(
        variant={"RS_ID": "rs121913527", "Gene": "BRCA1", ...},
        patient_phenotype="Breast cancer",
        all_outputs={},
        variant_report="",
        variant_index=0,
        total_variants=1,
        llm=llm,
    )
    print(tool.run(ctx.variant, ctx))
```

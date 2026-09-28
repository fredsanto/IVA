> **Archived.** Migration is complete. This document is archived for historical reference only.
> The constraints, task list, and in-progress notes below no longer apply to the current codebase.

# MIGRATION.md — Refactoring Instructions for Claude Code

This document tells Claude Code exactly what to do with the existing source files.
Read CLAUDE.md first for architecture. This document is about the mechanical migration only.
Do not rewrite logic unless explicitly told to below.

---

## Ground rule

**Preserve existing logic unless told otherwise.**
The goal is to reorganize and wrap, not rewrite.
If a function works today, it must work identically after migration.

---

## File-by-file instructions

---

### `csv_normalizer.py` → `pipeline/core/normalizer.py`

**Action: move and rename only.**

- Copy the file verbatim to `pipeline/core/normalizer.py`
- No logic changes
- Update the module docstring path reference
- The public function `normalize_upload(raw, filename)` stays identical

---

### `autopvs1_tool.py` → `pipeline/tools/autopvs1.py`

**Action: move + wrap in new Tool base class.**

- Copy all existing functions verbatim:
  - `vcf_to_autopvs1_id()`
  - `parse_variant_coords()`
  - `fetch_variant_data()`
  - `_parse_autopvs1_html()`
  - `fetch_and_format_autopvs1()`
- Keep them as module-level functions — they are used internally by the tool class
- Add a new class at the bottom:

```python
class AutoPVS1Tool(NetworkTool):
    name        = "autopvs1"
    description = "Fetches ACMG PVS1 criterion evaluation from AutoPVS1 for LoF variants (hg19)."

    def gate(self, variant: dict, context: "ToolContext") -> bool:
        # Exact same logic as _pvs1_decision() in server_main.py
        # adapted to receive variant dict instead of variant string
        # SLM fallback calls context.llm.generate() instead of _chat()
        pass

    def run(self, variant: dict, context: "ToolContext") -> str | None:
        # Call parse_variant_coords() then fetch_and_format_autopvs1()
        # exactly as done in run_browsing() in server_main.py
        pass
```

- The `if __name__ == "__main__"` block stays for standalone testing.

---

### `litvar2_tool.py` → `pipeline/tools/litvar2.py`

**Action: move + wrap in new Tool base class.**

- Copy all existing code verbatim
- `LitVar2SummaryTool` inherits from `SLMTool` instead of `Tool` from retrieval_agent
- Replace `slm` constructor argument with `ToolContext` access:
  - Old: `_call_slm(self.slm, system, user)` → New: `context.llm.generate(system, user, max_tokens)`
  - Remove `__init__(self, slm, ...)` — LLM comes from context, not constructor
  - Keep `max_pmids`, `top_n`, `max_chars`, `timeout` as class-level attributes with defaults
- `gate()` returns `context.field("RS_ID") != "NA"`
- All private methods unchanged

---

### `tools.py` → split into two files

**Action: split by tool class, no logic changes.**

**`pipeline/tools/websearch.py`** contains:
- All shared helpers: `NCBI_BASE`, `NCBI_MIN_DELAY`, `DEFAULT_TIMEOUT`, `DEFAULT_MAX_CHARS`,
  `_ncbi_get()`, `_clean_xml_text()`, `_extract_body_text()`
- `WebSearchTool` → inherits from `NetworkTool`
- `WebFetchTool` → inherits from `NetworkTool`

**`pipeline/tools/ncbi.py`** contains:
- Imports shared helpers from `websearch.py`
- `NCBIFetchTool` → inherits from `NetworkTool`
- All private methods `_fetch_pubmed`, `_fetch_pmc`, `_fetch_clinvar` unchanged

---

### `retrieval_agent.py` → split into three locations

**Part 1: `LLM` class → `pipeline/llm/hf_client.py`**

- Extract `LLM` class, rename to `HFClient`
- Inherit from `LLMClient` (abstract base in `pipeline/llm/base.py`)
- Implement abstract `generate(self, system, user, max_tokens, temperature, enable_thinking)`
- Internal logic (tokenizer, pipeline, apply_chat_template) stays identical
- `enable_thinking` is always forced to `False` — silent override, never passed through

**Part 2: `AgentConfig`, `ToolRegistry`, prompt builders, parsers → `pipeline/core/executor.py`**

- `AgentConfig` → keep as dataclass
- `ToolRegistry` → keep identical
- Old `Tool` base class from `retrieval_agent.py` → DELETED, replaced by `pipeline/tools/base.py`
- Prompt builder functions (`thought_messages`, `action_pick_messages`, etc.) → keep verbatim
- Parser functions (`parse_thought`, `parse_action`, `parse_yes_no`) → keep verbatim

**Part 3: `ReActAgent` → `pipeline/tools/websearch_agent.py`**

- Copy `ReActAgent` verbatim into the file
- Add a wrapper class:

```python
class WebSearchAgentTool(ReActTool):
    name        = "websearch_agent"
    description = "ReAct agent that searches the web and NCBI for variant-phenotype evidence."

    def run(self, variant: dict, context: ToolContext) -> str | None:
        # Instantiate ReActAgent with context.llm and the web tools
        # Run with the retrieval prompt template
        # Return output string
        pass
```

- The monkey-patching block (`SharedLLM`, `_patched_react_init`) in `server_main.py`
  is removed entirely — it was a workaround and is no longer needed.

---

### `server_main.py` → split into four locations

**Part 1: Model loading → `pipeline/llm/hf_client.py`**

The tokenizer/model/pipeline loading block moves into `HFClient.__init__()`.
`generate_text_from_prompt()` and `generate_text_from_messages()` become
private methods of `HFClient`. No longer at module level.

**Part 2: Prompt templates → `prompts/` directory**

Extract verbatim (do not change a single character):

- `RETRIEVAL_TEMPLATE` → `prompts/retrieval.txt`
- `REASONING_TEMPLATE` → `prompts/reasoning.txt`
- `CONCLUSION_TEMPLATE` → `prompts/conclusion.txt`

Create new placeholder at `prompts/compression.txt`:
```
Compress the following variant evidence to {max_tokens} tokens maximum.
Preserve: key findings, ACMG criteria, evidence strength, URLs.
Remove: repetition, verbose explanations, raw data already interpreted.

Evidence:
{variant_report}
```

**Part 3: Business logic → pipeline stages**

- `run_browsing()` → `pipeline/stages/retrieval.py`
  The per-variant loop logic moves into `pipeline/core/executor.py`.
  `retrieval.py` calls the executor per variant and concatenates results.
- `run_reasoning()` → `pipeline/stages/reasoning.py` as `run(augmented_context, llm) -> str`
- `run_conclusion()` → `pipeline/stages/conclusion.py` as `run(augmented_context, reasoning, llm) -> str`
- `_pvs1_decision()` → moves into `AutoPVS1Tool.gate()`
- `_extract_rsid()`, `_field_from()`, `_extract_pvs1_strength()` → `pipeline/core/executor.py` as helpers
- `_chat()` → removed, replaced by `context.llm.generate()`

**Part 4: FastAPI routes → `server.py`**

Keep only the FastAPI app setup and routes. The `/analyze` route becomes:

```python
@app.post("/analyze")
async def analyze(csv_file: UploadFile, patient_report: str = Form(...)):
    raw = await csv_file.read()
    variants = normalize_upload(raw, csv_file.filename)
    result = await pipeline.run(variants, patient_report)
    return PlainTextResponse(content=result)
```

All logic delegated to `pipeline/pipeline.py`.

---

## New files to create from scratch

| File | What it contains |
|---|---|
| `pipeline/llm/base.py` | `LLMClient` abstract base class |
| `pipeline/llm/vllm_client.py` | `VLLMClient` via `httpx` to vLLM OpenAI endpoint |
| `pipeline/llm/registry.py` | `MODELS` dict + `get_client(model_name)` factory |
| `pipeline/tools/base.py` | `Tool`, `NetworkTool`, `SLMTool`, `ReActTool`, `RAGTool`, `BotTool` |
| `pipeline/core/context.py` | `ToolContext` dataclass |
| `pipeline/core/errors.py` | All typed exceptions |
| `pipeline/core/manifest.py` | Manifest loader, validator, gate resolver |
| `pipeline/core/executor.py` | Tool fan-out, compression, mini-doc assembly |
| `pipeline/stages/compression.py` | Global context compression stage |
| `pipeline/pipeline.py` | Top-level orchestrator, calls stages in order |
| `pipeline/manifests/litvar2.yaml` | Manifest for LitVar2 |
| `pipeline/manifests/autopvs1.yaml` | Manifest for AutoPVS1 |
| `pipeline/manifests/websearch_agent.yaml` | Manifest for ReAct agent |
| `launch.sh` | SLURM sbatch script (see CLAUDE.md deployment section) |
| `client.py` | CLI client for login node submission |
| `README.md` | User-facing quickstart |
| `TOOL_DEVELOPMENT.md` | Tool writing guide (expand from CLAUDE.md tool section) |

---

## What NOT to do

- Do not rewrite `ReActAgent` internals
- Do not modify prompt template strings — extract verbatim
- Do not change the normalizer logic
- Do not add new tools — framework must work end-to-end with three existing tools first
- Do not add tests
- Do not change FastAPI route signatures — existing frontend must keep working
- Do not remove `if __name__ == "__main__"` blocks in tool files

---

## Build order

Follow strictly. Each step must be importable before moving to the next.

1. `pipeline/core/errors.py`
2. `pipeline/llm/base.py`
3. `pipeline/core/context.py`
4. `pipeline/tools/base.py`
5. `pipeline/llm/hf_client.py`
6. `pipeline/llm/vllm_client.py`
7. `pipeline/llm/registry.py`
8. `pipeline/core/normalizer.py`
9. `pipeline/tools/websearch.py` + `pipeline/tools/ncbi.py`
10. `pipeline/tools/autopvs1.py`
11. `pipeline/tools/litvar2.py`
12. `pipeline/tools/websearch_agent.py`
13. `pipeline/core/manifest.py`
14. `pipeline/manifests/*.yaml`
15. `pipeline/core/executor.py`
16. `pipeline/stages/*.py`
17. `pipeline/pipeline.py`
18. `server.py`
19. `prompts/*.txt`
20. `launch.sh`, `client.py`, `README.md`, `TOOL_DEVELOPMENT.md`

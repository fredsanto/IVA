# CLAUDE.md — Variant Analysis Pipeline

This document is the authoritative reference for Claude Code when working on this project.
Read it fully before writing, editing, or refactoring any file.

---

## Project Overview

This is a clinical genomics pipeline that takes a list of genetic variants and a patient
phenotype description, gathers evidence from multiple sources per variant, and produces
a structured clinical report using a Small Language Model (SLM).

The pipeline is designed to run on a SLURM HPC cluster (CURNAGL, A100 40GB GPU node)
and is exposed as a FastAPI server — making it accessible interactively from a browser,
programmatically via HTTP, or as an sbatch job. The server is the single universal
entry point regardless of how compute resources were allocated.

**Core philosophy:**
- The framework handles orchestration. Tools handle one thing each.
- Adding a new tool = one `.py` file + one `.yaml` manifest + one import line in `pipeline/tools/__init__.py` + one instance line in `Pipeline._tools` (`pipeline/pipeline.py`).
- The SLM is a shared resource. Tools request it via context; they do not own it.
- Errors are informative, not silent. Failures appear in the report with context.

---

## Repository Structure

```
variant-pipeline/
│
├── CLAUDE.md                        ← this file
├── MIGRATION.md                     ← migration notes from original codebase
├── README.md                        ← user-facing quickstart and architecture
├── TOOL_DEVELOPMENT.md              ← guide for writing new tools
├── launch.sh                        ← SLURM sbatch script
├── batch.sh                         ← SLURM sbatch script for batch folder processing
├── client.py                        ← CLI client for submitting jobs from login node
├── batch.py                         ← batch runner: process a folder of variant files
├── server.py                        ← FastAPI entry point (thin, no business logic)
│
├── pipeline/
│   ├── __init__.py
│   ├── pipeline.py                  ← top-level orchestrator
│   ├── config.py                    ← global pipeline configuration (GENOME_BUILD, EXECUTION_MODE)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── manifest.py              ← manifest loading, validation, gate resolution
│   │   ├── context.py               ← ToolContext dataclass
│   │   ├── executor.py              ← tool fan-out, error handling, mini-doc assembly
│   │   ├── normalizer.py            ← CSV/Excel ingestion and schema normalization
│   │   └── errors.py                ← typed pipeline exceptions
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                  ← LLMClient abstract base class
│   │   ├── vllm_client.py           ← vLLM HTTP client (primary, for A100 node)
│   │   ├── hf_client.py             ← HuggingFace pipeline client (fallback/dev)
│   │   └── registry.py              ← model registry, instantiate by name
│   │
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── retrieval.py             ← per-variant tool fan-out stage
│   │   ├── first_triage.py          ← fast SLM keep/discard decision per variant
│   │   ├── reasoning.py             ← two-call SLM: step-by-step reasoning + second triage (INCLUDE/EXCLUDE)
│   │   ├── cross_analysis.py        ← gene-level compound-het / interaction analysis
│   │   ├── conclusion.py            ← per-variant structured report (included variants only)
│   │   └── final_conclusion.py      ← overall clinical conclusion paragraph
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                  ← Tool, NetworkTool, SLMTool, RAGTool, BotTool
│   │   ├── litvar2.py               ← literature search: gene-first three-track PubMed + LitVar2 rsID
│   │   ├── autopvs1.py              ← AutoPVS1 PVS1 criterion evaluation (LoF variants)
│   │   ├── spliceai.py              ← SpliceAI splice impact delta scores (Broad API)
│   │   ├── gnomad_constraint.py     ← gnomAD gene constraint (pLI, LOEUF) — gene-scoped, class-cached
│   │   ├── clinvar_gene_stats.py    ← ClinVar gene-level P/LP missense vs. nonsense/frameshift counts
│   │   ├── clingen_allele.py        ← ClinGen Allele Registry variant resolution (CAid, cross-refs)
│   │   ├── genereviews.py           ← GeneReviews clinical-description fetch (NCBI Bookshelf) — gene-scoped, class-cached
│   │   ├── websearch.py             ← WebSearchTool + WebFetchTool (sub-tools for ReAct agent)
│   │   ├── websearch_agent.py       ← WebSearchAgentTool: full ReAct web-search loop
│   │   └── ncbi.py                  ← NCBIFetchTool (sub-tool for ReAct agent)
│   │
│   └── manifests/
│       ├── litvar2.yaml
│       ├── autopvs1.yaml
│       ├── spliceai.yaml
│       ├── gnomad_constraint.yaml
│       ├── clinvar_gene_stats.yaml
│       ├── clingen_allele.yaml
│       ├── genereviews.yaml
│       └── websearch_agent.yaml
│
└── prompts/
    ├── retrieval.txt
    ├── compression.txt
    ├── first_triage.txt
    ├── reasoning.txt
    ├── second_triage.txt
    ├── conclusion.txt
    ├── cross_analysis.txt
    └── final_conclusion.txt
```

---

## Variant Schema

The pipeline normalizes any input CSV or Excel file to the following canonical schema.
Unknown columns are ignored. Missing columns are filled with `"NA"`.

### Canonical fields

| Field | Type | Example | Notes |
|---|---|---|---|
| `Variant` | str | `chr6:100896130 T>C` | Free-text genomic identifier |
| `Chromosome` | str | `chr6` | With or without `chr` prefix |
| `Position` | str | `100896130` | 1-based, hg19 |
| `RS_ID` | str | `rs28936388` | `NA` if not in dbSNP |
| `Ref_seq` | str | `T` | Reference allele; `NA` for complex indels |
| `Var_seq` | str | `C` | Alternate allele |
| `Type` | str | `SNV` | SNV, Insertion, Deletion, Indel, CNV… |
| `HGVS` | str | `NM_005068.3:c.744-2A>G p.?` | Full HGVS notation |
| `Zygosity` | str | `Heterozygous` | Heterozygous / Homozygous / Hemizygous |
| `Gene` | str | `SIM1` | HGNC gene symbol |
| `OMIM_phenotype` | str | `Gitelman syndrome` | Associated OMIM disease name |
| `OMIM_inheritance` | str | `Autosomal recessive` | Full inheritance string |
| `Inheritance` | str | `AR` | Short code: AR, AD, XL, … |
| `ClinVar_class` | str | `Pathogenic` | ClinVar clinical significance |
| `Allelic_balance` | float | `0.5393` | VAF / allele balance |
| `Frequency` | float | `1.19E-05` | Population allele frequency (gnomAD) |
| `CADD_score` | float | `26` | CADD PHRED score |

### Real example rows (from test dataset)

```
Variant=chr19:40730680_1 insG, Chromosome=chr19, Position=40730681, RS_ID=NA,
Ref_seq=NA, Var_seq=G, Type=Insertion, HGVS=NM_024877.4:c.305dup p.(Val103CysfsTer25),
Zygosity=Heterozygous, Gene=CCNP, OMIM_phenotype=NA, OMIM_inheritance=NA,
Inheritance=NA, ClinVar_class=NA, Allelic_balance=0.5417, Frequency=0, CADD_score=NA

Variant=chr6:100896130 T>C, Chromosome=chr6, Position=100896130, RS_ID=NA,
Ref_seq=T, Var_seq=C, Type=SNV, HGVS=NM_005068.3:c.744-2A>G p.?,
Zygosity=Heterozygous, Gene=SIM1, OMIM_phenotype=NA, OMIM_inheritance=NA,
Inheritance=AD, ClinVar_class=NA, Allelic_balance=0.383, Frequency=0, CADD_score=34
```

### Schema extensibility

The normalizer maps any incoming column name to a canonical field via `COLUMN_ALIASES`
in `core/normalizer.py`. To support a new input format, add aliases there.
The canonical schema is the single source of truth — tools always receive a normalized
variant dict, never raw CSV rows.

Additional computed or externally-fetched fields (e.g. gnomAD v4 AF, SpliceAI score)
can be injected by a tool and will appear in `context.all_outputs` for downstream tools.

---

## LLM Layer

### Design

All SLM access goes through a single abstract `LLMClient`. Tools never instantiate
models directly. The client is created once at server startup and injected into
`ToolContext`.

```python
# pipeline/llm/base.py
from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,   # always False for Qwen3.5 in this pipeline
    ) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...
```

### Implementations

**`vllm_client.py`** — primary client for A100 node.
Calls the vLLM OpenAI-compatible HTTP endpoint (`/v1/chat/completions`).
Supports async via `httpx.AsyncClient`. Used in production.

Every request includes `"chat_template_kwargs": {"enable_thinking": False}` — this
is the vLLM equivalent of the `enable_thinking=False` flag that the HF tokenizer
applies via `apply_chat_template`. Without it, Qwen3.5-9B generates plain-text
"Thinking Process:" preambles in non-thinking mode that contaminate downstream
context and break output parsers. A `_strip_thinking_header()` regex is also applied
to every response as a defensive fallback.

**`hf_client.py`** — fallback for local dev without GPU.
Wraps `transformers.pipeline("text-generation")`.
Same interface, slower. Useful for testing tool logic without a running vLLM server.
Applies `enable_thinking=False` via `tokenizer.apply_chat_template` — this is what
prevents thinking preambles in HF mode and was the reference for the vLLM fix.

### Model registry

```python
# pipeline/llm/registry.py
MODELS = {
    "qwen3.5-9b": {
        "client": "hf",            # currently "hf" (dev default); change to "vllm" for production
        "model_id": "Qwen/Qwen3.5-9B",
        "enable_thinking": False,  # MUST be False for Qwen3.5 in this pipeline
        "max_tokens_default": 1024,
    },
    # add new models here
}

def get_client(model_name: str, **kwargs) -> LLMClient:
    ...
```

**Qwen3.5 note:** `enable_thinking` must always be `False` in this pipeline.
The thinking mode produces chain-of-thought tokens that bloat the context and
break downstream parsing. For the HF client this is enforced via
`tokenizer.apply_chat_template(..., enable_thinking=False)`. For the vLLM client
it is enforced via `"chat_template_kwargs": {"enable_thinking": False}` in the
request payload — the two are equivalent but use different APIs.

---

## Tool System

### Tool types

| Class | Use when |
|---|---|
| `Tool` | Pure deterministic logic, no network, no SLM |
| `NetworkTool` | HTTP fetch, no SLM |
| `SLMTool` | Calls the SLM for summarization, filtering, or judgment |
| `ReActTool` | Full ReAct loop with tool registry (current websearch agent) |
| `RAGTool` | Vector DB retrieval (interface ready, implementation pluggable) |
| `BotTool` | Browser automation (interface ready, Playwright/Selenium pluggable) |

All inherit from `Tool`. Subclasses add capabilities; they do not change the contract.

### Tool contract

Every tool must implement:

```python
class MyTool(Tool):                      # or NetworkTool, SLMTool, etc.
    name        = "my_tool"              # matches manifest name
    description = "One sentence."        # shown in logs and TOOL_DEVELOPMENT.md

    def gate(self, variant: dict, context: "ToolContext") -> bool:
        """Return False to skip this tool for this variant. Default: always run."""
        return True

    def run(self, variant: dict, context: "ToolContext") -> str | None:
        """
        Execute the tool for one variant.
        Return a string block to include in the mini-doc, or None if no result.
        Raise a PipelineError subclass on failure (never return an error string).
        """
        raise NotImplementedError
```

### ToolContext

Every tool receives a `ToolContext` instance. This is the only way tools access
shared resources.

```python
@dataclass
class ToolContext:
    # Variant being processed
    variant: dict                        # normalized variant dict

    # Patient information
    patient_phenotype: str               # free-text phenotype from user

    # Pipeline state
    all_outputs: dict[str, str | None]   # {tool_name: output} for tools that ran before this one
    variant_report: str                  # accumulated mini-doc so far (for compression tools)
    variant_index: int                   # 0-based index of this variant
    total_variants: int                  # total number of variants in this run

    # Shared resources (injected at startup, never instantiated by tools)
    llm: LLMClient                       # call context.llm.generate(system, user, max_tokens)
    db: VectorStore | None = None        # None if no DB configured
    browser: BrowserSession | None = None  # None if no bot configured

    # Run-level metadata
    genome_build: str = "hg19"           # reference genome build for this run
    raw_fields: dict = field(default_factory=dict)  # full original CSV row for this variant

    # Convenience
    def field(self, name: str) -> str:
        """Extract a field from the variant dict. Returns 'NA' if missing or falsy."""
        return self.variant.get(name, "NA") or "NA"
```

### Gate system

Gates control whether a tool runs for a given variant. They are evaluated by the
executor before calling `run()`. A gate can be:

**1. Python method** (most flexible):
```python
def gate(self, variant: dict, context: ToolContext) -> bool:
    return context.field("Type").lower() in {"insertion", "deletion", "frameshift"}
```

**2. Manifest expression** (simple cases, no custom code needed):
```yaml
gate:
  field: Type
  operator: in
  value: [Insertion, Deletion, frameshift, nonsense, splice_site]
```

**3. SLM gate** (ambiguous cases — calls SLM with max_tokens=5):
```yaml
gate:
  type: slm
  prompt: "Should PVS1 be evaluated for this variant? Type={Type} HGVS={HGVS}. Answer YES or NO."
```

The Python method always takes precedence over the manifest gate if both are defined.

Manifest gates only support single-field conditions with a single operator. For OR logic
or multi-field conditions a Python `gate()` method is required, and the `gate:` block
should be omitted from the manifest entirely. `litvar2_summary` is the current example:
it gates on RS_ID valid **or** Gene present — this cannot be expressed in a single YAML
expression, so `litvar2.yaml` has no `gate:` block and the logic lives entirely in the
Python `gate()` method.

### Compression

A tool can declare that its output should be compressed when the variant count
exceeds a threshold. This is configured in the manifest:

```yaml
compress:
  enabled: true
  threshold: 8          # compress if total_variants > 8
  strategy: slm         # slm | truncate | first_n_lines
  max_tokens: 300       # target length after compression
```

Alternatively, the threshold can be inferred from the variant table at runtime
by setting `threshold: auto` — the executor will compute a per-tool token budget
based on total variant count and available context window.

Compression is applied by the executor after `run()` returns, before appending
to the mini-doc. The tool itself does not need to handle this.

### Error handling

Tools must raise typed exceptions from `pipeline/core/errors.py`.
Never return error strings. Never raise bare `Exception`.

```python
# pipeline/core/errors.py

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
```

The executor catches all `PipelineError` subclasses per tool, logs the error,
and writes a failure note into the mini-doc:

```
AUTOPVS1 [FAILED — ToolFetchError]: Connection timeout after 15s (https://autopvs1.bgi.com/...)
```

This failure note is visible to the SLM in the reasoning stage, which can then
note the missing evidence in the report rather than reasoning from a gap.

### Implemented tools

| Tool | Class | Order | Gate |
|---|---|---|---|
| `litvar2_summary` | `SLMTool` | 1 (parallel) | RS_ID valid **or** Gene present (Python gate) |
| `spliceai` | `NetworkTool` | 1 (parallel) | Python gate — skips synonymous, intergenic, UTR, unresolvable coords |
| `gnomad_constraint` | `NetworkTool` | 1 (parallel) | Gene present (manifest gate) |
| `clinvar_gene_stats` | `NetworkTool` | 1 (parallel) | Gene present (manifest gate) |
| `clingen_allele` | `NetworkTool` | 1 (parallel) | Python gate — a usable query can be built (transcript+cDNA, clean HGVS, or genomic SNV coordinates) |
| `genereviews` | `NetworkTool` | 1 (parallel) | Gene present (manifest gate) |
| `autopvs1` | `NetworkTool` | 2 (parallel) | Python gate — LoF/frameshift/splice variants only |
| `websearch_agent` | `ReActTool` | 3 (serial) | No gate — always runs; the ReAct agent's own pre-loop checkpoint (sees prior tool outputs + full variant record, including ClinVar_class/Frequency) decides whether search is needed |

**`litvar2_summary`** runs a gene-first three-track search:

- **Track 1 — Gene + patient phenotype (primary — always runs when Gene is present)** — queries PubMed `esearch` with `({gene}[tiab] OR {gene}[Gene]) AND {disease_query}`. If no disease co-occurrence is found, retries with gene-only to distinguish "no literature at all" from "gene unrelated to disease". Returns an explicit `"NO DISEASE LINK"` message on zero relevant hits — providing a clear negative signal for triage. Output header includes PubMed's `querytranslation` and pool counts for downstream traceability. When total results exceed `MAX_PMIDS`, a second call sorted by publication date adds recent papers not in the relevance window.
- **Track 2 — Gene + known disease term (supplemental — runs when a disease term is available)** — bridges the terminology gap between how the patient's condition is described and how OMIM/CGD name the canonical disease for that gene. Disease term source priority: (a) `OMIM_phenotype` from the variant dict; (b) NHGRI Clinical Genomic Database (CGD) conditions for the gene — downloaded once per process and cached at the class level (`_cgd_table`). Relevance filtering anchors on the known condition; summarisation is framed on the patient phenotype so output stays clinically grounded. Track silently skipped when no disease term is available or no relevant papers are found.
- **Track 3 — rsID variant-level search (supplemental — runs when RS_ID is valid)** — queries the LitVar2 variant publications endpoint for variant-specific literature. Omitted from output if no relevant papers are found. Output header identifies LitVar2 as source, distinguished from the gene-level PubMed blocks.

All tracks that yield evidence are combined in the output separated by `---`. `_disease_query` is resolved once per pipeline run via a single SLM call (`_resolve_disease_query`), cached as an instance attribute, and reused for all variants. The resolver returns 2–4 PubMed-compatible disease terms as a PubMed OR expression, preferring MeSH-indexed terms for broad recall.

**`autopvs1`** uses a HGVS-first query strategy: when a valid transcript HGVS is available (`NM_xxx:c.xxx`), it queries AutoPVS1 via `/search?q={hgvs}` so AutoPVS1 resolves coordinates internally — immune to stale CSV coords. Falls back to VCF-style coords (`/variant/{build}/{chrom}-{pos}-{ref}-{alt}`) when HGVS is absent or the search path fails. Results are discarded when AutoPVS1 returns `variant_type="Intergenic"` or the returned gene does not match the expected gene. `pvs1_applicable` is `True` only when a flowchart is present in the response and no "incompatible with recommendations" message is found.

**`spliceai`** queries the Broad Institute SpliceAI API and returns delta scores
(DS_AG, DS_AL, DS_DG, DS_DL) for acceptor/donor gain and loss, along with the
`sai10kPredictions` clinical interpretation block (aberration type, confidence,
frameshift description). Only the MANE Select transcript is reported. Coordinates
and genome build are sourced from the canonical variant dict fields; build is read
from `pipeline/config.py` (`DEFAULT_GENOME_BUILD`). Gate is two-layer: Python gate
skips synonymous, intergenic, UTR, and variants with unresolvable coordinates;
manifest gate additionally skips the API call when the canonical `SpliceAI_score`
field is already populated — i.e. the SLM header-interpretation step (see
`core/normalizer.py`) recognized a precomputed SpliceAI column in the input,
regardless of its raw name (`SpliceAI_v13`, `spliceai_concat`, or any other
annotation-tool naming). Compound annotation strings bundling multiple
delta/position values (e.g. ANNOVAR-style pipe-delimited output) are collapsed
to a single max delta score by `_parse_spliceai_value()` based on the value's
shape, not the column's name. Output labels are enriched for SLM readability
(plain-English aberration names, explicit delta score scale).

**`gnomad_constraint`** fetches pLI and LOEUF for the variant's gene from the public
gnomAD GraphQL API. Gene-scoped, not variant-scoped — one call per gene per run,
shared across all variants in that gene via a class-level cache. Used to judge
whether loss-of-function is a plausible disease mechanism for the gene (PVS1
supporting context) and, more generally, whether a gene tolerates LoF at all.

**`clinvar_gene_stats`** fetches gene-level ClinVar P/LP variant counts, split by
missense vs. nonsense/frameshift, via NCBI esearch (count-only, no per-variant
fetch). Grounds PP2 (missense-predominant genes) and BP1 (truncating-predominant
genes) — these criteria may only be applied when this block is present with the
matching verdict, never from gene name or general plausibility alone.

**`clingen_allele`** resolves the variant against the public ClinGen Allele Registry
(`reg.clinicalgenome.org`, no API key) to its canonical allele ID (CAid) and
cross-references ClinVar/dbSNP/gnomAD/ExAC. Query strategy, in order: (1) Transcript
field + cDNA token extracted from HGVS, (2) HGVS field itself if already a clean
versioned `transcript:c.` string, (3) genomic-coordinate SNV fallback — tried against
*both* hg19 and hg38 RefSeq accessions (declared build first), because the pipeline's
declared `genome_build` is not always right for a given upload and a wrong build
reads back from ClinGen as an `IncorrectReferenceAllele` mismatch rather than a
clean failure. An allele with zero cross-references in any indexed database is
itself a meaningful signal (novel/unreported variant), surfaced explicitly rather
than as a blank result.

**`genereviews`** fetches the gene's canonical GeneReviews chapter(s) directly from
NCBI Bookshelf (`esearch` gene symbol → gene ID, `elink` gene→books, `esummary` to
resolve chapter accessions, then a plain page fetch + `Clinical Description` /
`Clinical Characteristics` / `Suggestive Findings` section extraction) and returns
that curated phenotype text verbatim. Gene-scoped, class-cached like
`gnomad_constraint`. Exists because `litvar2_summary`'s Track 1 sorts by `pub_date`
to surface newly-characterized gene-disease links (see its own note above) — for a
gene with a large, unrelated publication volume (e.g. a common cancer gene that also
causes a rare syndrome) that can bury the syndrome's own defining phenotype
description outside the `max_pmids`-capped pool entirely, and the reasoning stage
then correctly summarizes an incomplete pool as "no phenotype link" from evidence
that never actually contained the relevant paper. Concrete case that motivated this
tool: SMAD4/Myhre syndrome — LitVar2's pool surfaced a narrow 2026 hepatic
case report and concluded no neurological association, while GeneReviews' own Myhre
Syndrome chapter explicitly lists "developmental delay/intellectual disability" and
"epilepsy" as part of the syndrome. A gene with no GeneReviews entry returns an
explicit "no chapter found" message rather than `None`, consistent with the
"errors are informative, not silent" rule.

**`websearch_agent`** runs a ReAct loop with three sub-tools (WebSearchTool,
WebFetchTool, NCBIFetchTool). No gate — always runs. Before starting the loop, a
pre-loop checkpoint prompt receives all pre-fetched evidence from earlier tools
(LitVar2, AutoPVS1, SpliceAI, gnomAD constraint, ClinVar gene stats, ClinGen allele,
GeneReviews) plus the full variant record (including `ClinVar_class`/`Frequency`)
and decides whether any primary gaps remain (OMIM, ClinVar details, functional data,
recent case reports — GeneReviews itself is now pre-fetched, so the agent should not
need to re-search for it) or whether the variant needs search at all — this replaced
an earlier Python `gate()` that hard-skipped ClinVar benign/likely-benign and common
(AF > 1%) variants before the checkpoint ever ran; that decision now lives entirely
in the checkpoint's own judgment, which has full visibility into those same fields.
The loop runs up to `max_steps=4` iterations; after each observation a mid-loop
checkpoint decides whether to continue or stop.

---

## Manifest System

Each tool has a corresponding YAML manifest in `pipeline/manifests/`.
The manifest declares metadata, gate, ordering, and compression config.
The executor loads all manifests at startup and resolves tool instances by `name`.

### Manifest schema

```yaml
# pipeline/manifests/example.yaml

name: my_tool                    # must match Tool.name in the Python class
description: "One sentence."     # used in logs and documentation
enabled: true                    # set false to disable without deleting

order: 2                         # execution order (lower = earlier); ties broken alphabetically
parallel: true                   # whether this tool can run concurrently with others at same order

gate:                            # optional; omit to always run, or when gate() is in Python
  field: RS_ID                   # variant field to evaluate
  operator: not_na               # operators: equals, not_equals, in, not_in, not_na, na, slm
  value: ~                       # not needed for not_na / na

compress:                        # optional compression of this tool's output
  enabled: false
  threshold: auto                # integer or "auto"
  strategy: slm                  # slm | truncate | first_n_lines
  max_tokens: 300

timeout: 15                      # seconds before ToolFetchError is raised

retry:
  attempts: 2
  backoff: 2.0                   # seconds between retries
```

### Execution order

Tools at the same `order` value run concurrently if `parallel: true` and the
executor is in async mode. Tools at different order values run sequentially
(i.e. order 1 tools all complete before order 2 tools start).

This allows expressing dependencies: a compression tool at order 99 sees the
full accumulated `variant_report` from all earlier tools.

### Parallelism

All pipeline stages use a `ThreadPoolExecutor` with `MAX_WORKERS = 32` (set in
both `pipeline/stages/retrieval.py` and `pipeline/pipeline.py`). Variants are
processed concurrently; tools within each variant still run in manifest order.

vLLM's continuous batching handles concurrent HTTP requests natively — parallel
threads translate directly into higher GPU throughput without any special async
plumbing in the pipeline.

Thread-safety requirements:
- `executor.run_variant()` returns `(mini_doc, log_entries)` — no shared state mutated.
- `LitVar2SummaryTool._disease_query` and `_cgd_table` are initialized with
  double-checked locking (`threading.Lock()`).
- `WebSearchAgentTool._last_trace` uses `threading.local()` to avoid cross-thread
  contamination.

GPU KV cache math for the A100 40 GB (Qwen3.5-9B bfloat16):
- Model weights: ~18 GB → KV cache pool: ~18 GB (at `--gpu-memory-utilization 0.90`)
- ~112 KB per token; 32 workers × ~3000 tokens each ≈ 10.7 GB — well within budget.
- Beyond ~48 workers vLLM starts queuing rather than crashing; latency increases.

To change worker count, edit `MAX_WORKERS` in both files (one line each).

---

## Pipeline Stages

The full flow is: retrieval → first triage → reasoning + second triage → cross-analysis → conclusion → final conclusion.

### Stage 1 — Retrieval (`stages/retrieval.py`)

For each variant, calls `executor.run_variant()` and collects the returned context slice.
`retrieval.py` is a thin loop — manifest loading, gate evaluation, tool ordering,
execution, and per-tool compression all happen inside the Executor.

```
retrieval.py (per variant)
    → executor.run_variant(variant, ...)
          → gate evaluation (skip if False)
          → tool.run(variant, context)
          → compression (if configured and threshold met)
          → append to mini-doc
    ← context slice string
```

All context slices are kept independently (one per variant) for downstream stages.

### First Triage (`stages/first_triage.py`)

One small SLM call per variant immediately after retrieval. Decides whether a variant
can be discarded with high confidence given the patient phenotype.
Prompt loaded from `prompts/first_triage.txt`.

Only runs when `n > TRIAGE_ENABLED_THRESHOLD` (configured in `pipeline/config.py`;
current value: 12). When skipped, all variants are implicitly KEEP.

Compound heterozygous candidates (two or more variants in the same gene) are
automatically exempted from DISCARD — a DISCARD decision on any such variant is
overridden to KEEP.

SLM output format:
```
Keep-case: <strongest reason to keep, ≤10 words>
Discard-case: <strongest reason to discard, ≤10 words>
Decision: KEEP or DISCARD
```

Results:
- **KEEP** variants proceed to reasoning.
- **DISCARD** variants are excluded from all subsequent stages and appear in the
  appendix of the final report with their justification.

### Stage 2 — Reasoning + Second Triage (`stages/reasoning.py`)

Two SLM calls per kept variant, executed sequentially:

**Call 1 — reasoning** (`prompts/reasoning.txt`): produces step-by-step clinical
reasoning grounded in cited evidence only. The model is explicitly instructed not
to produce a score or classification and not to introduce external knowledge without
a reference in the provided context. Steps: phenotype fit → variant description →
molecular mechanism → inheritance coherence → flags and uncertainties.

**Call 2 — second triage** (`prompts/second_triage.txt`): takes the variant context
and the reasoning from call 1 and emits a structured inclusion decision. Decoupling
the decision from the narrative prevents the model from forcing a favourable outcome
to justify speculative reasoning. `MAX_NEW_TOKENS_SCORING = 400` — budget is large
enough to accommodate any residual thinking preamble before the three decision lines.

SLM output format for second triage:
```
Include-case: <strongest reason to include, ≤15 words>
Exclude-case: <strongest reason to exclude, ≤15 words>
Decision: INCLUDE or EXCLUDE
```

The two outputs are concatenated (separated by `SECOND TRIAGE:`) before being stored,
so the REASONING display section in the final output is self-contained.

Results:
- **INCLUDE** variants proceed to cross-analysis and conclusion.
- **EXCLUDE** variants are listed in the report appendix with their reasoning available
  in the REASONING section.

### Stage 3 — Cross-analysis (`stages/cross_analysis.py`)

Runs after reasoning, for each gene that has two or more included variants. Presents the
per-variant contexts and reasonings to the SLM and asks it to evaluate potential
compound heterozygous or other gene-level interactions.
Prompt loaded from `prompts/cross_analysis.txt`.

The cross-analysis output is passed into `conclusion.run_one()` for the relevant variants.

### Stage 4 — Conclusion (`stages/conclusion.py`)

One SLM call per included variant. Takes the variant context slice, its reasoning block,
and (if available) the cross-analysis for its gene. Produces the structured variant
report block. Prompt loaded from `prompts/conclusion.txt`.

After the SLM call, a `_validate_citations` post-processing step strips any inline
citations that are not grounded in the variant context: `(PMID:XXXXXXXX)` references
are kept only if that PMID appears in the context; `(https://...)` references are kept
only if the URL appears verbatim in the context. This prevents hallucinated citations
from propagating into the final report.

Output format per variant:
```
# Variant [N] — [GENE] ([HGVS])
**Molecular mechanism:** ...
**Phenotype fit:** match | mismatch | uncertain — ...
**Inheritance check:** consistent | inconsistent | insufficient data
**Evidence strength:** strong | moderate | weak | absent
**ACMG criteria**: ...
**Comment:** ...
```

### Stage 5 — Final Conclusion (`stages/final_conclusion.py`)

Single SLM call. Takes all included variants' conclusion blocks and the patient phenotype.
Prompt loaded from `prompts/final_conclusion.txt`.

Output format (two sections):
```
# Reasoning
[3–6 sentences comparing variants, drawing on what each conclusion states]

# Clinical Conclusion
[2–3 sentences naming the primary causative variant and why it fits]
```

This stage is separate from Stage 4 so that each variant's block can be generated
individually (reducing SLM memory pressure) while the overall summary still has
visibility across all variants.

The combined conclusions text is truncated to 50 000 characters before the LLM
call (`_MAX_CONCLUSIONS_CHARS`) to prevent a `400 Bad Request` from vLLM when a
large number of variants are included and the combined text exceeds
`--max-model-len 16384` tokens.

---

## Deployment

### SLURM (production)

`launch.sh` starts two processes in the same job allocation:
1. vLLM server on the GPU (serves the model via OpenAI-compatible HTTP)
2. FastAPI server on CPU (pipeline logic, calls vLLM for SLM steps)

```bash
#!/bin/bash
#SBATCH --job-name=variant_batch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

# Start vLLM server
vllm serve Qwen/Qwen3.5-9B \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --port 8001 \
    --enable-prefix-caching \
    > vllm_startup.log 2>&1 &

# Poll until ready (Qwen3.5 Mamba architecture takes several minutes to initialize)
for i in $(seq 1 60); do
    curl -sf http://localhost:8001/v1/models > /dev/null 2>&1 && break
    sleep 5
done

# Run batch pipeline
python batch.py "$JOBS_FOLDER" --backend vllm
```

`--max-model-len 16384` — keeps KV cache within the 40 GB A100 budget.
`--gpu-memory-utilization 0.90` — reserves headroom to avoid OOM during concurrent requests.
`--enable-prefix-caching` — reuses KV cache for shared prompt prefixes across requests.
Startup polling uses 60 attempts × 5 s = 5 minutes; Qwen3.5's hybrid Mamba
architecture needs this much time on first load.

### Local / dev

Set `LLM_BACKEND=hf` to use `hf_client`. No vLLM needed.
Slower but identical pipeline behavior.

### Client (login node)

```bash
python client.py \
    --server http://<node>:8000 \
    --csv variants.csv \
    --phenotype "Obesity, intellectual disability, hyperphagia" \
    --output report.txt
```

---

## Adding a New Tool — Quick Reference

Full guide in `TOOL_DEVELOPMENT.md`. Minimum steps:

1. Create `pipeline/tools/my_tool.py` inheriting from the appropriate base class
2. Implement `gate()` (optional) and `run()`
3. Create `pipeline/manifests/my_tool.yaml` with `name`, `order`, `enabled`
4. Register in `pipeline/tools/__init__.py`

No other files need to change.

### Tool template

```python
# pipeline/tools/my_tool.py
from pipeline.tools.base import SLMTool          # or Tool, NetworkTool, RAGTool, BotTool
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError


class MyTool(SLMTool):
    name        = "my_tool"
    description = "One sentence describing what this tool fetches or computes."

    def gate(self, variant: dict, context: ToolContext) -> bool:
        # Return False to skip. Access variant fields via context.field("FieldName").
        return context.field("RS_ID") != "NA"

    def run(self, variant: dict, context: ToolContext) -> str | None:
        rsid = context.field("RS_ID")

        # Network call example
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

        # Optional SLM call (only available in SLMTool subclasses)
        summary = context.llm.generate(
            system="Summarize this evidence in 2 sentences.",
            user=str(data),
            max_tokens=200,
        )

        return f"MY_TOOL EVIDENCE ({rsid}):\n{summary}"
```

### Manifest template

```yaml
# pipeline/manifests/my_tool.yaml
name: my_tool
description: "One sentence."
enabled: true
order: 3
parallel: true
gate:
  field: RS_ID
  operator: not_na
compress:
  enabled: true
  threshold: auto
  strategy: slm
  max_tokens: 200
timeout: 15
retry:
  attempts: 2
  backoff: 2.0
```

---

## Interface-Ready Tool Types

These base classes exist and are importable but have no concrete implementations yet.
They define the interface contract so tools can be written without knowing the
underlying library.

### RAGTool

```python
class RAGTool(SLMTool):
    """
    Base class for tools that retrieve from a vector database.
    context.db is a VectorStore instance (Qdrant by default).
    
    The VectorStore interface:
        context.db.search(query: str, collection: str, top_k: int, filters: dict) -> list[Chunk]
        context.db.collections() -> list[str]
    
    Chunk fields: text, score, metadata (dict with source, gene, rsid, etc.)
    
    For variant queries, always pass gene and/or rsid as filters to avoid
    pure semantic drift on short identifiers.
    """
```

### BotTool

```python
class BotTool(Tool):
    """
    Base class for tools that use browser automation.
    context.browser is a BrowserSession instance.
    
    The BrowserSession interface:
        context.browser.get(url: str) -> str          # returns page text
        context.browser.click(selector: str)
        context.browser.fill(selector: str, value: str)
        context.browser.wait_for(selector: str, timeout: int)
    
    Underlying implementation (Playwright or Selenium) is injected at startup.
    BotTool subclasses never import Playwright or Selenium directly.
    
    Example use case: Silicon (https://silicon.com) or any JS-rendered
    clinical database that does not expose an API.
    """
```

---

## Key Design Rules

These rules must be respected in all new code:

1. **Tools are stateless with respect to variant data.** Tools must never store
   per-variant results on `self` between calls — all variant-scoped state lives in
   `ToolContext`. Two accepted exceptions exist: (a) instance-level caching of
   expensive one-time computations that are identical across all variants (e.g.
   `_disease_query` in `litvar2_summary`, derived solely from `patient_phenotype`) —
   initialised in `__init__` as `None`; (b) class-level caching of data fetched once
   per process and shared across all instances and variants (e.g. `_cgd_table` in
   `litvar2_summary`). Both forms must be pipeline-run-scoped and variant-independent.

2. **Tools never load models.** The LLM client is always accessed via `context.llm`.

3. **Tools never open DB connections.** Always via `context.db`.

4. **Errors are typed.** Always raise from `pipeline/core/errors.py`.
   Never return error strings. Never raise bare `Exception`.

5. **Prompts live in files.** All SLM prompts longer than one sentence belong in
   `prompts/`. Load them with:
   `Path(__file__).parent.parent.parent / "prompts" / "my_prompt.txt"`.
   This resolves relative to the source file, not the working directory.
   Do not hardcode multi-line prompts in Python.

6. **Manifests are the source of truth for ordering and gating.**
   Do not hardcode tool order or conditions in stage code.

7. **`enable_thinking` is always False.** Enforced at the LLM registry level.
   No tool or stage should pass `enable_thinking=True`.

8. **Output is free-form string or None.** Tools do not need to produce structured
   output. The SLM in the reasoning and conclusion stages handles synthesis.

9. **Compression is the executor's job.** Tools return their full output.
   The executor applies compression based on the manifest config.

10. **The server knows nothing about variants.** `server.py` parses the request,
    calls `pipeline.run()`, and returns the result. All logic is in `pipeline/`.

11. **No real gene/variant identifiers in prompts — ever.** Prompts in `prompts/*.txt`
    and any inline prompt text in `pipeline/tools/*.py` (e.g. `websearch_agent.py`'s
    message builders) must never name a real gene symbol, real HGVS/protein notation,
    real ClinVar Variation ID, or exact point value/arithmetic copied from an actual
    evaluation case (the CHUV trio set or the singleton visual-impairment set). This
    applies to "a real past failure" worked examples too — describe the *shape* of the
    failure generically (e.g. "GENE_X", "a dominant-relevant gene", "c.100A>C",
    "cluster A ... cluster B") instead of the real identifiers from the debugging
    session that found it. Baking a real gene/variant into a prompt is training the
    pipeline on its own eval set — it fixes that one case while overfitting the model's
    attention to a name it will pattern-match on, not the general rule. Before
    finalizing any prompt edit, grep the new text for gene symbols and exact numbers
    pulled from the current session's actual report output, every time — this has
    already slipped through unnoticed more than once. See project memory
    `feedback_no_test_data_in_prompts`.

---

## Dependencies

### Core runtime
```
torch>=2.0
transformers>=4.40
fastapi
uvicorn
pydantic
httpx           # vLLM async client
vllm            # production inference (A100 node)
```

### Pipeline tools
```
requests
beautifulsoup4
ddgs            # DuckDuckGo search (was duckduckgo-search)
pandas
openpyxl        # Excel support in normalizer
```

### Optional (future tools)
```
qdrant-client   # RAG / vector DB
sentence-transformers  # embeddings for RAG
playwright      # BotTool browser automation
```

### Notes
- `torch` and `transformers` are only required by `hf_client.py` (dev/fallback mode)
- In production (vLLM mode), the FastAPI server process does NOT load torch directly —
  the model runs in the separate vLLM subprocess
- `vllm` must be installed in the same environment as the model server, not necessarily
  the FastAPI server environment

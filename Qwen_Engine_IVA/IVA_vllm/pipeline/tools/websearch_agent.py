"""
pipeline/tools/websearch_agent.py — ReAct agent wrapper for web + NCBI literature search.

Contains:
  - AgentConfig        — dataclass for ReAct loop hyper-parameters
  - ToolRegistry       — lightweight tool name→instance mapping
  - Prompt builders    — thought_messages, action_pick_messages, etc.
  - Parsers            — parse_thought, parse_action, parse_yes_no
  - ReActAgent         — copied verbatim from retrieval_agent.py; constructor
                         extended with optional llm= parameter to accept
                         an externally injected LLM (removes monkey-patching workaround)
  - _LLMAdapter        — adapts LLMClient.generate(system, user, max_tokens)
                         to the ReActAgent's generate(messages, max_new_tokens) interface
  - WebSearchAgentTool — ReActTool pipeline wrapper; instantiates ReActAgent with
                         context.llm and runs the retrieval prompt

Note: The monkey-patching block (SharedLLM, _patched_react_init) from server_main.py
is removed entirely — it was a workaround, now superseded by the llm= constructor param.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.tools.base import ReActTool
from pipeline.core.context import ToolContext
from pipeline.tools.websearch import WebSearchTool, WebFetchTool
from pipeline.tools.ncbi import NCBIFetchTool
from pipeline.core.acmg_sf import is_pathogenic_clinvar
from pipeline.core.protein_change import extract_residue_queries as _extract_residue_queries


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

@dataclass
class AgentConfig:
    model_id:            str   = "Qwen/Qwen3.5-9B"
    temperature:         float = 0.7
    top_k:               int   = 5
    top_p:               float = 0.8
    do_sample:           bool  = True
    max_steps:           int   = 4
    tokens_thought:      int   = 50
    tokens_action_pick:  int   = 15
    tokens_action_input: int   = 60
    tokens_checkpoint:   int   = 15
    tokens_final:        int   = 300


# ─────────────────────────────────────────────
# TOOL REGISTRY  (internal — for ReAct sub-tools)
# ─────────────────────────────────────────────

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, object] = {}

    def register(self, tool):
        self._tools[tool.name] = tool

    def get(self, name: str):
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def description_block(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())


# ─────────────────────────────────────────────
# LLM ADAPTER
# ─────────────────────────────────────────────

class _LLMAdapter:
    """
    Adapts pipeline LLMClient.generate(system, user, max_tokens) to the
    ReActAgent's internal generate(messages, max_new_tokens) interface.

    ReActAgent builds fresh message lists for every sub-call (thought, action,
    checkpoint, final answer). The lists always follow [system, ...turns, user]
    with conversation always reset to [] — so messages are effectively [system, user].
    This adapter extracts system and concatenates remaining turns as user.
    """

    def __init__(self, client):
        self._client = client

    def generate(self, messages: list[dict], max_new_tokens: int) -> str:
        system = ""
        user_parts = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_parts.append(m["content"])
        user = "\n\n".join(user_parts)
        return self._client.generate(system=system, user=user, max_tokens=max_new_tokens)


# ─────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────

def _scratchpad_str(scratchpad: list[str]) -> str:
    return ("\n\nSteps so far:\n" + "\n".join(scratchpad)) if scratchpad else ""


def _prefetched_str(prefetched: str) -> str:
    """Format the pre-fetched block for inclusion in prompts."""
    return (
        f"\n\nPre-fetched evidence (prior tools):\n{prefetched}"
        if prefetched.strip() else ""
    )


def thought_messages(user_input: str, tools: ToolRegistry,
                     conversation: list[dict], scratchpad: list[str],
                     prefetched: str = "") -> list[dict]:
    return [
        {"role": "system", "content": (
            f"You are a reasoning assistant.\n\nAvailable tools:\n{tools.description_block()}\n\n"
            "Decide what must happen next.\n\n"
            "RULES:\n"
            "1. If a tool is strictly necessary, name exactly one tool in your sentence.\n"
            "2. Do NOT use a tool unless truly required.\n"
            "3. Check the pre-fetched evidence below to understand what has already been retrieved. "
            "Avoid searching for information already present there — focus on genuine gaps.\n"
            "4. Output ONE sentence starting with 'Thought:' explaining your next step.\n\n"
            "OUTPUT: One sentence starting with 'Thought:' — nothing before it, nothing after it."
        )},
        *conversation,
        {"role": "user", "content": (
            f"Question: {user_input}"
            f"{_prefetched_str(prefetched)}"
            f"{_scratchpad_str(scratchpad)}"
        )},
    ]


def action_pick_messages(user_input: str, tools: ToolRegistry,
                         scratchpad: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": (
            f"You are a tool selector.\n\nAvailable tools:\n{tools.description_block()}\n\n"
            f"Valid choices: {', '.join(tools.names())}\n\n"
            "Output ONLY the tool name. One word. No punctuation."
        )},
        {"role": "user", "content": f"Question: {user_input}{_scratchpad_str(scratchpad)}"},
    ]


def action_input_messages(user_input: str, tool,
                           scratchpad: list[str]) -> list[dict]:
    user_content = tool.input_template.format(user_input=user_input)
    if scratchpad:
        user_content += "\n\nContext so far:\n" + "\n".join(scratchpad)
    return [
        {"role": "system", "content": tool.input_system},
        {"role": "user",   "content": user_content},
    ]


def checkpoint_messages(user_input: str, scratchpad: list[str],
                        prefetched: str = "") -> list[dict]:
    """
    Full-context checkpoint: sees both pre-fetched evidence (LitVar2 / AutoPVS1)
    AND the accumulated scratchpad observations as two distinct labelled blocks.

    Called both:
      - Before the ReAct loop (scratchpad=[]) to check if pre-fetched data alone suffices.
      - After each tool observation inside the loop to check if further search is needed.
    """
    pre_block = (
        f"Pre-fetched evidence (prior tools):\n{prefetched.strip()}\n\n"
        if prefetched.strip() else ""
    )
    obs_block = (
        f"Search observations:\n{chr(10).join(scratchpad)}"
        if scratchpad else "Search observations: none yet."
    )
    return [
        {"role": "system", "content": (
            "You are a search decision checker for a clinical genomics pipeline. "
            "Given the available evidence, decide whether further web search is needed.\n\n"
            "GeneReviews for this gene, if one exists, is already in the pre-fetched "
            "evidence below (GENEREVIEWS block) — read it before deciding anything is "
            "missing; do not treat 'GeneReviews entry' as an open gap if that block is "
            "present, even if it only partially covers the patient's phenotype.\n\n"
            "Primary gaps to check first (prefer these over PubMed):\n"
            "- OMIM entry for this gene (GeneReviews is covered above)\n"
            "- Mode of inheritance for this gene (autosomal dominant/recessive, "
            "X-linked, or a dominant-negative mechanism) — check OMIM's "
            "'Inheritance' field, or the pre-fetched GeneReviews block\n"
            "- ClinVar submitter details or review status\n"
            "- Functional or experimental data for this specific variant\n"
            "- Recent preprints or clinical case reports\n\n"
            "Secondary gap (only if primary gaps are resolved):\n"
            "- Critical gene-disease or variant-phenotype literature not already "
            "present in the pre-fetched evidence\n\n"
            "Say YES (sufficient) if primary gaps are filled or unresolvable by web search, "
            "and no critical secondary gap remains.\n"
            "Say NO (search needed) if any concrete gap above is still open.\n"
            "When uncertain, say NO.\n"
            "Your entire response must be a single word: yes or no. No other text whatsoever."
        )},
        {"role": "user", "content": (
            f"Question: {user_input}\n\n"
            f"{pre_block}"
            f"{obs_block}\n\n"
            "Is the evidence sufficient to assess this variant without further searches? (yes/no)"
        )},
    ]


def final_answer_messages(user_input: str, conversation: list[dict],
                           scratchpad: list[str], prefetched: str = "") -> list[dict]:
    """
    Final answer sees both pre-fetched evidence and scratchpad observations
    as distinct labelled blocks, ensuring nothing is silently omitted.
    """
    pre_block = (
        f"Pre-fetched evidence (prior tools):\n{prefetched.strip()}\n\n"
        if prefetched.strip() else ""
    )
    obs_block = (
        "\n".join(scratchpad) if scratchpad else "No additional searches were performed."
    )
    return [
        {"role": "system", "content": (
            "You are a clinical genomics evidence synthesiser. "
            "Write a clear, direct evidence summary using the sources provided below. "
            "Synthesise the findings that link (or do not link) the variant or gene to the disease. "
            "Include URLs for any sources cited. "
            "Do NOT mention tools, search steps, or internal reasoning. "
            "Begin directly with the evidence.\n\n"
            "NAMED-SYNDROME SCAN (mandatory, do this BEFORE writing anything else): "
            "re-read every search-result title and snippet in the observations below — "
            "not just pages you fetched in full — for a disease/syndrome name that "
            "itself contains the gene's name (e.g. '<GENE>-related syndrome', "
            "'<GENE> syndrome', '<GENE>-related disorder', a GeneReviews/OMIM/Orphanet/"
            "Simons Searchlight gene-guide title). A hit like this is itself a "
            "first-class gene-disease link — it counts as evidence even if you never "
            "fetched the full page, and it must be named explicitly in your phenotype-"
            "relevance answer. A real past failure to not repeat: two separate "
            "web_search calls for a gene both returned a result titled "
            "'Simons Searchlight | <GENE>-related syndrome ... also called "
            "<GENE>-related neurodevelopmental disorder' — a direct, named hit "
            "matching the patient's developmental phenotype — yet the final answer "
            "still wrote 'No direct evidence links <GENE> to the patient's phenotype', "
            "simply re-echoing the earlier pre-fetched PubMed track's negative framing "
            "and never mentioning the syndrome name sitting in its own search results. "
            "Do not let a large volume of irrelevant search results (unrelated disease "
            "associations, cancer/metabolic papers, etc.) bury a small but directly "
            "on-point named-syndrome hit — surface it, don't average it away.\n\n"
            "Whenever the sources support it, explicitly answer these two questions "
            "before finishing your answer: (1) Phenotype relevance — is this gene "
            "affirmatively linked to the patient's phenotype, and how specifically; "
            "(2) Mode of inheritance — is this gene's disease mechanism autosomal "
            "dominant, autosomal recessive, X-linked, or dominant-negative "
            "(name the mechanism explicitly, e.g. 'dominant-negative — mutant "
            "subunit interferes with the wild-type protein', if the sources "
            "describe one), citing the source for each. If a source does not "
            "address one of these questions, say so plainly rather than omitting "
            "the question entirely — do not guess or infer either answer beyond "
            "what the sources state.\n\n"
            "PARTIAL PHENOTYPE OVERLAP IS NOT A MISMATCH: the patient's phenotype "
            "text often lists findings from more than one distinct system/cluster "
            "(e.g. a neurodevelopmental cluster plus an unrelated renal or "
            "oncologic finding). If the sources affirmatively link the gene to ONE "
            "cluster but say nothing about a separate, unrelated cluster, this is a "
            "PARTIAL match, not a mismatch — write your Phenotype relevance answer "
            "as 'affirmatively linked to [cluster A]; sources are silent on "
            "[cluster B]', with the affirmative link as the lead clause and the "
            "silence as a trailing caveat. Do NOT title or frame this as "
            "'Phenotype Mismatch', 'no link to the patient's phenotype', or any "
            "other net-negative verdict — silence on a secondary, unrelated "
            "cluster is not evidence against the gene, and downstream triage "
            "treats a labeled 'Mismatch' section as a strong exclude signal even "
            "when the underlying finding is genuinely partial. A real past "
            "failure: a gene's literature affirmatively confirmed one clinical "
            "cluster matching a patient feature, but the final answer added a "
            "separate 'Phenotype Mismatch:' section over the patient's other, "
            "unrelated findings — which caused a genuinely causative variant to "
            "be excluded downstream. Only call something a mismatch when the "
            "sources are silent on, or actively contradict, EVERY "
            "cluster in the patient's phenotype, not merely a secondary one."
        )},
        *conversation,
        {"role": "user", "content": (
            f"Question: {user_input}\n\n"
            f"{pre_block}"
            f"Search observations:\n{obs_block}\n\n"
            "Final answer:"
        )},
    ]


# ─────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────

def parse_thought(raw: str) -> str:
    for line in raw.splitlines():
        if line.strip().lower().startswith("thought:"):
            return line.split(":", 1)[1].strip()
    return next((l.strip() for l in raw.splitlines() if l.strip()), raw.strip())


def parse_action(raw: str, valid_names: list[str]) -> Optional[str]:
    for word in raw.lower().split():
        clean = re.sub(r"[^a-z_]", "", word)
        if clean in valid_names:
            return clean
    return None


def parse_yes_no(raw: str) -> Optional[bool]:
    for word in raw.lower().split():
        clean = re.sub(r"[^a-z]", "", word)
        if clean == "yes": return True
        if clean == "no":  return False
    return None


# ─────────────────────────────────────────────
# REACT AGENT  (verbatim from retrieval_agent.py;
#               constructor extended with optional llm= parameter)
# ─────────────────────────────────────────────

class ReActAgent:
    def __init__(self, cfg: AgentConfig, tools: list, llm=None):
        """
        Args:
            cfg:   AgentConfig with loop hyper-parameters.
            tools: List of ReAct sub-tool instances to register.
            llm:   Optional pre-built LLM object with .generate(messages, max_new_tokens).
                   If None, an HFClient is instantiated (standalone / dev mode).
        """
        self.cfg = cfg
        if llm is not None:
            self.llm = llm
        else:
            # Standalone / dev fallback — deferred import avoids loading torch at import time
            from pipeline.llm.hf_client import HFClient
            _client = HFClient(model_id=cfg.model_id)
            self.llm = _LLMAdapter(_client)

        self.registry = ToolRegistry()
        for t in tools:
            self.registry.register(t)
        self._last_trace: list[str] = []
        self.last_scratchpad: list[str] = []

    # ── public entry point ──────────────────

    def run(self, user_input: str, conversation: list[dict] = None,
            prefetched: str = "") -> str:
        """
        Run the ReAct loop.

        Args:
            user_input:   The task / question for the agent.
            conversation: Prior conversation turns (currently unused / reset).
            prefetched:   Pre-fetched evidence string (LitVar2 + AutoPVS1 blocks).
                          Passed to every checkpoint and the final answer so the
                          model always has full context when deciding whether to
                          keep searching or synthesize a conclusion.
        """
        self._last_trace = []
        conversation = []
        scratchpad: list[str] = []

        # ── Pre-loop checkpoint ───────────────────────────────────────────────
        # If pre-fetched evidence alone is sufficient, skip web searches entirely.
        if prefetched.strip():
            print("\n[Pre-loop checkpoint] Evaluating pre-fetched evidence...")
            self._last_trace.append("[Pre-loop checkpoint] Evaluating pre-fetched evidence...")
            if self._check(user_input, scratchpad, prefetched):
                print("[Pre-loop checkpoint] Pre-fetched evidence sufficient — skipping ReAct loop.")
                self._last_trace.append("[Pre-loop checkpoint] Sufficient — ReAct loop SKIPPED")
                self.last_scratchpad = scratchpad
                return self._final(user_input, conversation, scratchpad, prefetched)
            print("[Pre-loop checkpoint] Pre-fetched evidence insufficient — proceeding with search.")
            self._last_trace.append("[Pre-loop checkpoint] Insufficient — proceeding with ReAct loop")

        # ── ReAct loop ────────────────────────────────────────────────────────
        for step in range(self.cfg.max_steps):
            print(f"\n{'='*44}\n[Step {step + 1}]")

            # 1 · Thought
            thought = self._thought(user_input, conversation, scratchpad, prefetched)
            scratchpad.append(f"Thought: {thought}")

            # 2 · Action selection
            action_name = self._pick_action(user_input, scratchpad)

            if action_name is None:
                print("[Warn] No valid tool selected — finalizing.")
                self.last_scratchpad = scratchpad
                return self._final(user_input, conversation, scratchpad, prefetched)

            scratchpad.append(f"Action: {action_name}")
            tool = self.registry.get(action_name)

            # 3 · Action input
            query = self._action_input(user_input, tool, scratchpad)
            scratchpad.append(f"Action Input: {query}")

            # 4 · Execute tool
            print(f"[Tool] {action_name}({query!r})")
            obs = tool.run(query)
            print(f"[Obs]  {obs}")
            scratchpad.append(f"Observation ({action_name}): {obs}")

            # 5 · Post-step checkpoint (full context: prefetched + scratchpad)
            if self._check(user_input, scratchpad, prefetched):
                self.last_scratchpad = scratchpad
                return self._final(user_input, conversation, scratchpad, prefetched)

        print("[Warn] Max steps reached.")
        self.last_scratchpad = scratchpad
        return self._final(user_input, conversation, scratchpad, prefetched)

    # ── private helpers ─────────────────────

    def _thought(self, user_input, conversation, scratchpad, prefetched: str = "") -> str:
        raw = self.llm.generate(
            thought_messages(user_input, self.registry, conversation, scratchpad, prefetched),
            self.cfg.tokens_thought,
        )
        thought = parse_thought(raw)
        print(f"[Thought] {thought}")
        return thought

    def _pick_action(self, user_input, scratchpad) -> Optional[str]:
        raw = self.llm.generate(
            action_pick_messages(user_input, self.registry, scratchpad),
            self.cfg.tokens_action_pick,
        )
        print(f"[Action raw] {raw}")
        action = parse_action(raw, self.registry.names())
        print(f"[Action] {action}")
        return action

    def _action_input(self, user_input, tool, scratchpad) -> str:
        if tool.input_system is None:
            return ""
        raw = self.llm.generate(
            action_input_messages(user_input, tool, scratchpad),
            self.cfg.tokens_action_input,
        )
        result = next((l.strip() for l in raw.splitlines() if l.strip()), "")
        print(f"[Input] {result}")
        return result

    def _check(self, user_input, scratchpad, prefetched: str = "") -> bool:
        raw = self.llm.generate(
            checkpoint_messages(user_input, scratchpad, prefetched),
            self.cfg.tokens_checkpoint,
        ).lower()
        decision = parse_yes_no(raw)
        print(f"[Checkpoint] {raw!r} → {decision}")
        return decision is True

    def _final(self, user_input, conversation, scratchpad,
               prefetched: str = "") -> str:
        answer = self.llm.generate(
            final_answer_messages(user_input, conversation, scratchpad, prefetched),
            self.cfg.tokens_final,
        )
        print(f"[Answer] {answer}")
        return answer


# ─────────────────────────────────────────────
# PIPELINE TOOL WRAPPER
# ─────────────────────────────────────────────

# Retrieval prompt template path — populated at step 19 (prompts/*.txt)
_RETRIEVAL_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "retrieval.txt"

# Inline fallback used only when the prompt file has not been created yet
_RETRIEVAL_TEMPLATE_FALLBACK = (
    "PATIENT PHENOTYPE:\n{patient_report}\n\n"
    "VARIANT DATA:\n{variant}\n{prior_evidence_block}\n"
    "TASK: Search for evidence linking this variant to the patient phenotype."
)


def _load_retrieval_template() -> str:
    if _RETRIEVAL_PROMPT_PATH.exists():
        return _RETRIEVAL_PROMPT_PATH.read_text(encoding="utf-8")
    return _RETRIEVAL_TEMPLATE_FALLBACK


class WebSearchAgentTool(ReActTool):
    """ReAct agent that searches the web and NCBI for variant-phenotype evidence."""

    name        = "websearch_agent"
    description = "ReAct agent that searches the web and NCBI for variant-phenotype evidence."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tl = threading.local()   # thread-local storage for _last_trace

    @property
    def _last_trace(self) -> list[str]:
        return getattr(self._tl, "trace", [])

    @_last_trace.setter
    def _last_trace(self, value: list[str]) -> None:
        self._tl.trace = value

    # No gate() override — always runs (inherits Tool.gate()'s default True).
    # Previously hard-skipped ClinVar benign/likely-benign and common (AF > 1%)
    # variants in Python before the ReAct loop ever started. Removed: that
    # skip fired on the raw CSV ClinVar_class / Frequency fields alone, with
    # no judgment call — the ReActAgent's pre-loop checkpoint already decides
    # whether prefetched evidence is sufficient and web search is needed, and
    # is the right place for this decision (it sees all prior tool outputs and
    # has a purpose-built prompt), not a Python-side hard skip upstream of it.

    def run(self, variant: dict, context: ToolContext) -> str | None:
        """
        Instantiate ReActAgent with context.llm (wrapped in _LLMAdapter) and
        the standard web tools, then run the retrieval prompt for this variant.

        Pre-fetched evidence from all earlier tools (LitVar2, AutoPVS1, SpliceAI,
        and any future tools at order < 3) is passed to the agent via the
        prefetched= argument so it can decide whether to skip web search entirely.
        Tool outputs are included generically from context.all_outputs — no
        hardcoded tool names.
        """
        self._last_trace = []

        # ── Build prefetched block from earlier tools ─────────────────────────
        prefetched_parts = []
        for tool_name, output in context.all_outputs.items():
            if output:
                label = tool_name.upper().replace("_", " ")
                prefetched_parts.append(f"{label} EVIDENCE:\n{output}")

        # ── Hardcoded variant-specific search (residue, not nucleotide) ──────
        # A nucleotide-level query (e.g. "c.100A>C") can only ever find THIS
        # exact allele. Both PS1 and PM5 need the opposite: OTHER established
        # pathogenic variants at the SAME residue — PS1 via a DIFFERENT
        # nucleotide producing the SAME amino acid change, PM5 via a DIFFERENT
        # missense change altogether — neither of which a nucleotide-exact or
        # full-change-exact query can surface by construction. Search by
        # residue only (3-letter form, e.g. "p.Lys34", matching ClinVar's
        # naming convention, destination amino acid dropped) so one query
        # returns every substitution reported at that position. Downstream in
        # the conclusion stage: same amino acid change + different nucleotide
        # → PS1; different amino acid change (any nucleotide) → PM5; same
        # nucleotide as this variant → it's this variant's own record, not a
        # separate precedent (check PS3 instead).
        gene = (variant.get("Gene") or "").strip()
        hgvs = (variant.get("HGVS") or "").strip()
        if gene and gene != "NA" and hgvs and hgvs != "NA":
            import logging as _logging
            _logger = _logging.getLogger(__name__)
            _ws = WebSearchTool()
            residue_queries = _extract_residue_queries(hgvs)
            if residue_queries:
                for seed in residue_queries[:4]:   # bounded — multi-transcript HGVS rarely exceeds this
                    variant_query = f"{gene} {seed['query']} ClinVar"
                    _logger.info(
                        "[WebSearchAgent] Residue-level search: %r (candidate change: %s)",
                        variant_query, seed["own_change"],
                    )
                    result = _ws.run(variant_query)
                    if result and not result.startswith(("Error", "No results")):
                        prefetched_parts.append(
                            f"VARIANT WEB SEARCH ({variant_query} — candidate's own change is "
                            f"{seed['own_change']}; for PS1/PM5, compare each hit's amino acid "
                            f"and nucleotide change against this):\n{result}"
                        )
            else:
                # No parseable protein change (e.g. intronic/splice variant) —
                # PS1/PM5 don't apply to these anyway; fall back to a
                # nucleotide-level query for general ClinVar visibility.
                hgvs_short = hgvs.split(":")[-1] if ":" in hgvs else hgvs
                variant_query = f"{gene} {hgvs_short} ClinVar"
                _logger.info("[WebSearchAgent] Nucleotide-level search (no protein change parsed): %r", variant_query)
                result = _ws.run(variant_query)
                if result and not result.startswith(("Error", "No results")):
                    prefetched_parts.append(f"VARIANT WEB SEARCH ({variant_query}):\n{result}")

        # ── Forced ClinVar submission-level check (P/LP variants) ───────────
        # ClinVar_class P/LP in the input CSV is just an aggregate label — it
        # doesn't say how many submitters agree, nor what evidence backs it
        # (needed to ground PS3). The generic pre-loop checkpoint below only
        # "prefers" checking ClinVar details as one of several primary gaps and
        # can be satisfied by other evidence without ever pulling ClinVar's own
        # per-submission records, so this fetch is forced unconditionally
        # whenever ClinVar_class reads Pathogenic/Likely pathogenic, pulling:
        # (1) the classification tally across all individual submitters
        #     (P/LP/VUS/LB/B counts), and
        # (2) for each P/LP submission, its rationale and source (submitter +
        #     SCV accession + cited PMIDs) — so a P/LP call can be traced back
        #     to why and by whom, not just accepted as an aggregate label.
        clinvar_class = (variant.get("ClinVar_class") or "").strip()
        if is_pathogenic_clinvar(clinvar_class) and gene and gene != "NA":
            _ncbi = NCBIFetchTool()
            variation_id = _ncbi.resolve_clinvar_id(gene, hgvs)
            if variation_id:
                clinvar_detail = _ncbi._fetch_clinvar(variation_id)
                prefetched_parts.append(
                    f"CLINVAR SUBMISSION-LEVEL CHECK (forced — ClinVar_class={clinvar_class}):\n"
                    f"{clinvar_detail}"
                )
            else:
                prefetched_parts.append(
                    f"CLINVAR SUBMISSION-LEVEL CHECK (forced — ClinVar_class={clinvar_class}): "
                    "could not resolve a ClinVar variation ID for gene+HGVS — classification "
                    "breakdown and PS3 evidence cannot be pulled from ClinVar for this variant."
                )

        prefetched = "\n\n".join(prefetched_parts)

        # ── Build retrieval prompt ────────────────────────────────────────────
        template = _load_retrieval_template()

        # Convert variant dict back to the key=value string format the template expects
        from pipeline.core.normalizer import TARGET_COLUMNS
        variant_str = ", ".join(
            f"{k}={variant.get(k, 'NA')}" for k in TARGET_COLUMNS
        )

        # Prior evidence is passed via prefetched= to the agent — not duplicated in the prompt
        prompt = (template
            .replace("{patient_report}", context.patient_phenotype)
            .replace("{variant}", variant_str)
            .replace("{prior_evidence_block}", ""))

        # ── Instantiate and run agent ─────────────────────────────────────────
        cfg   = AgentConfig(model_id="Qwen/Qwen3.5-9B", max_steps=4)
        tools = [WebSearchTool(), WebFetchTool(max_chars=2000), NCBIFetchTool()]
        adapted_llm = _LLMAdapter(context.llm)
        agent = ReActAgent(cfg, tools, llm=adapted_llm)

        output = agent.run(prompt, conversation=[], prefetched=prefetched)

        # Capture ReAct trace from agent
        self._last_trace.extend(agent._last_trace)
        if hasattr(agent, 'last_scratchpad') and agent.last_scratchpad:
            for entry in agent.last_scratchpad:
                self._last_trace.append(entry)

        if not output:
            return None
        gene = context.field("Gene")
        header = f"Web search evidence for {gene}:" if gene != "NA" else "Web search evidence:"
        return f"{header}\n{output}"

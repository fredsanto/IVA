"""
pipeline/stages/final_conclusion.py — Cross-MOI clinical conclusion stage (Layer 8).

The last synthesis stage. Reads the per-MOI-layer variant blocks produced by
stages/moi_denovo.py, moi_dominant.py, moi_recessive.py, moi_xlinked.py (each
already carrying its own base+delta ACMG total), and generates the short "# Clinical Conclusion" paragraph that
closes the final report — now reasoning across MOI layers, not just across a
flat variant list, since a patient can have independently-explanatory
findings under different inheritance mechanisms (e.g. an AD_AR gene's variant
appearing in both the dominant and recessive layers).

This is separated from stages/conclusion.py (Layer 2) so each variant's
structured block can be generated individually (reducing SLM memory pressure)
while the overall summary still has visibility across everything.

Prompt loaded from prompts/clinical_conclusion.txt.

Public API:
    run(layer_outputs, patient_phenotype, llm) -> str
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH           = Path(__file__).parent.parent.parent / "prompts" / "clinical_conclusion.txt"
_LAYER_PROMPT_PATH     = Path(__file__).parent.parent.parent / "prompts" / "layer_conclusion.txt"
_REVISE_PROMPT_PATH    = Path(__file__).parent.parent.parent / "prompts" / "final_conclusion_revise.txt"
_RECONCILE_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "final_conclusion_reconcile.txt"

MAX_NEW_TOKENS_FINAL_CONCLUSION = 3000
MAX_NEW_TOKENS_LAYER            = 1500
MAX_WORKERS_LAYER               = 5

# With --max-model-len 16384, leave 2000 tokens for prompt template + output.
# ~4 chars per token → 14384 × 4 = 57536 chars safe budget for the conclusions block.
_MAX_CONCLUSIONS_CHARS = 50_000

# A real past failure: the model burned its whole token budget on rambling
# self-revision in STEP 2 (e.g. "*Wait*, looking closer... Re-reading the
# rule...") and got cut off mid-sentence in section 2, before ever reaching
# section 5 — but still contained the "# Clinical Conclusion" header, so the
# header-only check below let the truncated output through silently. Requiring
# every numbered section marker catches this: a genuinely truncated response
# is missing at least "5)" (the last section written).
_REQUIRED_SECTION_MARKERS = ("1)", "2)", "3)", "4)", "5)")


def _is_well_formed(text: str) -> bool:
    return "# Clinical Conclusion" in text and all(m in text for m in _REQUIRED_SECTION_MARKERS)


def _load_prompt(path: Path = _PROMPT_PATH) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt not found at {path}.")


# ── Stage A — MAP: per-layer parallel synthesis ─────────────────────────────
#
# A real observed failure: a patient with an unusually large evidence
# volume (many compound-het VUS blocks in one gene bloating the De
# Novo/Dominant-Inherited layers) produced a combined layer text of
# ~95-98k chars against this stage's 50k-char budget. The blind
# `conclusions_text[:_MAX_CONCLUSIONS_CHARS]` front-slice in run() silently
# dropped the ENTIRE Recessive layer (third in the fixed key order below) —
# a real causative compound-het finding — before the LLM ever saw it, and
# the synthesis model then correctly, but misleadingly, reported "no
# Recessive layer blocks were provided" (true of what it was given, false of
# what actually existed). The deterministic completeness check downstream
# caught the omission, but only as a mechanical footnote, not a real fix.
#
# Synthesizing each MOI layer independently, in parallel, before ever
# building one combined prompt fixes this structurally rather than by
# raising the budget number: (1) a layer's own map call cannot misreport
# another layer's contents, since it never sees them; (2) each map call's
# input is bounded by ONE layer's own evidence, not the sum across all
# layers, so the common case of "one large layer, others small" no longer
# risks starving a small-but-critical layer; (3) only causative (>=6pt) and
# notable-VUS (4-6pt) content survives into the reduce prompt — the bulk of
# a layer's raw text (VUS-below-4/benign blocks the SCOPE RULE forbids
# mentioning anywhere in the report regardless) is dropped at the source
# instead of eating budget it was never going to use downstream.
#
# This does not replace _qualifying_causative_findings /
# _enforce_causative_completeness / _reconcile_missing_causative below —
# those still scan the ORIGINAL, full, untruncated layer_outputs directly
# (never the map stage's output) and remain the deterministic safety net of
# last resort if a map call itself still drops something.

def _build_layer_prompt(layer_name: str, blocks: list[str], patient_phenotype: str) -> str:
    template = _load_prompt(_LAYER_PROMPT_PATH)
    return (template
        .replace("{patient_phenotype}", patient_phenotype)
        .replace("{layer_name}", layer_name)
        .replace("{layer_blocks}", "\n\n---\n\n".join(blocks)))


def _synthesize_layer(layer_name: str, blocks: list[str], patient_phenotype: str, llm: "LLMClient") -> str:
    """One MAP call, scoped to a single layer's own blocks. Falls back to the
    raw block text (pre-restructuring behavior for this layer only) on any
    LLM failure, so a single layer's map error never loses that layer's
    evidence entirely — it just reaches the reduce prompt unsummarized."""
    try:
        return llm.generate(
            system=(
                "You are an expert clinical geneticist, extracting causative "
                "and notable-VUS findings from one MOI layer's own evidence."
            ),
            user=_build_layer_prompt(layer_name, blocks, patient_phenotype),
            max_tokens=MAX_NEW_TOKENS_LAYER,
        )
    except Exception as exc:
        logger.warning(
            "[FinalConclusion] Layer synthesis failed for %s (%s) — "
            "falling back to raw block text for this layer.",
            layer_name, exc,
        )
        return "\n\n---\n\n".join(blocks)


def _synthesize_layers_parallel(
    layer_outputs: dict[str, list[str]],
    patient_phenotype: str,
    llm: "LLMClient",
) -> dict[str, str]:
    """Fan out one MAP call per non-empty layer, concurrently. Returns {layer_name: synthesized
    text}; a layer with no blocks at all is simply absent from the result —
    never represented by an empty or fabricated entry."""
    jobs: dict[str, list[str]] = {name: blocks for name, blocks in layer_outputs.items() if blocks}
    if not jobs:
        return {}

    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS_LAYER, len(jobs))) as pool:
        futures = {
            pool.submit(_synthesize_layer, name, blocks, patient_phenotype, llm): name
            for name, blocks in jobs.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
    return results


def _build_layers_text_from_synth(layer_synth: dict[str, str]) -> str:
    """Wrap each layer's already-synthesized (MAP-stage) text under the same
    "=== NAME LAYER ===" headers the previous raw-block concatenation used,
    so the reduce prompt (clinical_conclusion.txt) and the revise/reconcile
    prompts — which all pattern-match on these exact header strings — see an
    unchanged shape, just pre-filtered content."""
    parts = []
    for name, text in layer_synth.items():
        if not text or not text.strip():
            continue
        parts.append(f"=== {name.upper()} LAYER ===\n\n{text.strip()}")
    return "\n\n---\n\n".join(parts)


def _build_actionable_text(actionable_variants: list[dict] | None) -> str:
    """Render the ACMG SF actionable-variant list (ground truth for the
    Clinical Conclusion's own "Actionable findings" section) as plain lines
    of gene/HGVS/condition/classification — or an explicit "None" line so the
    model never has to guess whether the list was simply omitted."""
    if not actionable_variants:
        return "None — no variant in this run met ACMG SF actionable-gene criteria."
    lines = []
    for v in actionable_variants:
        lines.append(
            f"- {v['gene']} ({v['hgvs']}) — condition: {v['condition']}; "
            f"zygosity: {v['zygosity']}; classification: {v['classification']}"
        )
    return "\n".join(lines)


# ── Deterministic causative-list completeness check ─────────────────────────
#
# recompute_and_fix_totals() (pipeline.core.acmg_points) guarantees a stated
# total is arithmetically correct. It says nothing about whether a qualifying
# variant was silently dropped from section 2 entirely — a real observed
# failure: the synthesis model picked a single "winner" gene across layers
# (e.g. a De Novo finding) and demoted a separately-qualifying
# Dominant-Inherited finding in a different gene to an "also present, possibly
# coincidental" footnote, even though prompts/clinical_conclusion.txt's
# CAUSATIVE THRESHOLD RULE explicitly forbids treating this as a competition.
# This check re-derives, independently of the LLM, which variants the rule
# requires in section 2, and mechanically appends any the model dropped.

_CAUSATIVE_THRESHOLD = 6.0

_JOINT_STATUS_RE = re.compile(
    r"\*\*Joint compound-het classification:\*\*\s*(CAUSATIVE|COMPOUND VUS)"
)


def _qualifying_causative_findings(
    layer_outputs: dict[str, list[str]]
) -> list[dict]:
    """
    Collect every variant the CAUSATIVE THRESHOLD RULE requires in section 2:
    any block (De Novo / Dominant-Inherited / X-Linked / a
    homozygous-solo Recessive block) whose own final total is >= 6 points, or
    both variants of a compound-het PAIR block whose own "Joint compound-het
    classification" line says CAUSATIVE — never a COMPOUND VUS pair, even
    when one partner's individual total alone is >= 6 (see
    moi_recessive.py's _joint_compound_het_status).
    """
    findings: list[dict] = []

    def _process(block: str, layer_name: str) -> None:
        joint_m = _JOINT_STATUS_RE.search(block)
        if joint_m:
            if joint_m.group(1) != "CAUSATIVE":
                return  # COMPOUND VUS pair — neither partner is causative here
            # Only the per-variant "## Variant A/B — ..." headers pair with
            # the two totals; the block's own top "# Recessive Analysis —"
            # header also matches _HEADER_LINE_RE and, left in, shifted every
            # pairing by one (A's total under the top header, B's under A's).
            headers = [h for h in _HEADER_LINE_RE.finditer(block) if h.group(0).startswith("##")]
            totals = list(_ANY_TOTAL_RE.finditer(block))
            for header_m, total_m in zip(headers, totals):
                try:
                    points = float(total_m.group(1))
                except ValueError:
                    continue
                findings.append({
                    # Pair labels read "Variant N — GENE (HGVS)" — drop the prefix.
                    "gene":   header_m.group("gene").split("—")[-1].strip(),
                    "detail": header_m.group("detail").strip(),
                    "points": points,
                    "label":  total_m.group(2).strip(),
                    "layer":  layer_name,
                    "block":  block,
                })
            return
        f = _extract_variant_finding(block)
        if f and f["points"] >= _CAUSATIVE_THRESHOLD:
            f["layer"] = layer_name
            f["block"] = block
            findings.append(f)

    for layer_name, blocks in layer_outputs.items():
        for block in blocks:
            _process(block, layer_name)
    return findings


def _missing_from_section_2(text: str, findings: list[dict]) -> list[dict]:
    """
    The subset of `findings` whose gene name is not mentioned in the
    "2) Causative variant(s):" section of `text` — checked against that
    section's own text only, since a gene demoted to a reasoning-prose
    footnote or the VUS section does not count as being listed as causative.
    """
    m2 = re.search(r"^[ \t#*]*2\)\s", text, re.MULTILINE)
    if not m2:
        return findings  # no section 2 at all — everything is "missing"
    m3 = re.search(r"^[ \t#*]*3\)\s", text, re.MULTILINE)
    section2 = text[m2.start(): m3.start() if m3 else len(text)]
    return [f for f in findings if not re.search(rf"\b{re.escape(f['gene'])}\b", section2)]


def _enforce_causative_completeness(
    text: str,
    layer_outputs: dict[str, list[str]],
) -> str:
    """
    Append a mechanically-assembled addendum, right before section 3, for any
    variant the CAUSATIVE THRESHOLD RULE requires in section 2 but that the
    LLM synthesis dropped. No-op (returns `text` unchanged) when nothing is
    missing.
    """
    findings = _qualifying_causative_findings(layer_outputs)
    missing = _missing_from_section_2(text, findings)
    if not missing:
        return text

    logger.warning(
        "[FinalConclusion] Completeness check: %d causative-threshold variant(s) "
        "missing from section 2 (%s) — appending mechanically.",
        len(missing), ", ".join(f["gene"] for f in missing),
    )
    addendum_lines = [
        "",
        "[COMPLETENESS CHECK — the following variant(s) independently meet the "
        "CAUSATIVE THRESHOLD RULE (own layer block total ACMG points >= 6, or "
        "part of a CAUSATIVE compound-het pair) but were not named above; "
        "listed here mechanically from each block's own already-verified "
        "total. Consult the MOI-layer blocks for full evidence, segregation, "
        "and citations.]",
    ]
    for f in missing:
        addendum_lines.append(
            f"- {f['gene']} ({f['detail']}) — [{f['layer']} layer] "
            f"{_fmt_points(f['points'])} pts total → {f['label']}"
        )
    addendum = "\n".join(addendum_lines) + "\n"

    m3 = re.search(r"^[ \t#*]*3\)\s", text, re.MULTILINE)
    if m3:
        return text[: m3.start()] + addendum + "\n" + text[m3.start():]
    return text + "\n" + addendum


# ── SLM reconciliation of the completeness check's own findings ─────────────
#
# _enforce_causative_completeness (above) is a purely mechanical safety net:
# it appends an FYI addendum, it never rewrites section 2 itself. The revise
# pass (final_conclusion_revise.txt) DOES carry a CAUSATIVE-THRESHOLD OMISSION
# CHECK instruction telling the model to self-scan for exactly this — but that
# is one bullet among ~15 mandatory checks competing for attention in a single
# call, and in practice it still misses cases the deterministic scan catches
# (e.g. a gene where a low-scoring variant in one layer and a high-scoring
# variant in a different layer for the SAME gene get conflated, and the low
# score is what survives into section 2 — the CAUSATIVE-THRESHOLD OMISSION
# CHECK's own instructions don't name this specific same-gene-different-
# variant failure mode). _reconcile_missing_causative hands the SLM the exact
# already-verified missing finding(s) plus their own source block text
# directly, as the ONLY task for that call, rather than asking it to re-derive
# the omission itself alongside 14 other checks.

MAX_NEW_TOKENS_RECONCILE = 3000


def _build_missing_findings_text(missing: list[dict]) -> str:
    """Render each missing finding's full source block, deduplicated — a
    compound-het PAIR block contributes the same block text for both of its
    variants (see _process's joint_m branch above), so it must be shown once,
    not twice."""
    seen: list[str] = []
    for f in missing:
        if f["block"] not in seen:
            seen.append(f["block"])
    return "\n\n---\n\n".join(seen)


def _reconcile_missing_causative(
    text: str,
    missing: list[dict],
    llm: "LLMClient",
    patient_phenotype: str,
    actionable_text: str,
) -> str | None:
    """
    Focused third SLM pass, fired ONLY when the deterministic completeness
    check finds a variant the CAUSATIVE THRESHOLD RULE requires in section 2
    that both the draft and revise passes missed. Single job: incorporate the
    named missing finding(s), given their own source block text directly.

    Returns the reconciled text, or None on any failure (network/parse error,
    or a malformed/truncated response) — callers fall back to
    _enforce_causative_completeness's mechanical addendum in that case, so a
    failure here never regresses below the pre-existing behavior.
    """
    try:
        template = _load_prompt(_RECONCILE_PROMPT_PATH)
        prompt = (template
            .replace("{patient_phenotype}", patient_phenotype)
            .replace("{actionable_variants}", actionable_text)
            .replace("{current_conclusion}", text)
            .replace("{missing_findings_blocks}", _build_missing_findings_text(missing)))

        reconciled = llm.generate(
            system=(
                "You are an expert clinical geneticist, adding a confirmed-missing "
                "finding to an existing clinical conclusion."
            ),
            user=prompt,
            max_tokens=MAX_NEW_TOKENS_RECONCILE,
        )
    except Exception as exc:
        logger.warning(
            "[FinalConclusion] Reconcile pass failed (%s) — falling back to mechanical addendum.",
            exc,
        )
        return None

    if not _is_well_formed(reconciled):
        logger.warning(
            "[FinalConclusion] Reconcile pass produced malformed/truncated output — "
            "falling back to mechanical addendum."
        )
        return None

    return reconciled


# ── Deterministic last-resort fallback ──────────────────────────────────────
#
# Used only when the LLM fails to produce a well-formed "# Clinical
# Conclusion" even after one retry (see run() below) — e.g. the borderline-variant/
# arithmetic-rambling failure this module's docstring-level comments already
# describe, where the model burns its entire token budget re-litigating a
# single variant and never reaches the actual conclusion. Rather than ship a
# report with zero synthesis (previously: "whichever malformed pass happened
# to be longer"), assemble a minimal but guaranteed-well-formed conclusion
# mechanically from each block's own header + already-verified total line —
# every MOI/base stage (moi_denovo.py, moi_dominant.py, moi_recessive.py,
# moi_xlinked.py, conclusion.py) already ran recompute_and_fix_totals on its
# output before this stage ever saw it, so these numbers need no re-checking.

_HEADER_LINE_RE = re.compile(
    r"^#\s*[^\n]*?—\s*(?P<gene>[^(\n]+?)\s*\((?P<detail>[^\n]*)\)\s*$", re.MULTILINE
)
_ANY_TOTAL_RE = re.compile(
    r"\*\*(?:Total )?ACMG points:\*\*\s*([-+]?\d+(?:\.\d+)?)\s*→\s*([A-Za-z /()]+)"
)


def _fmt_points(n: float) -> str:
    return str(int(n)) if n == int(n) else str(n)


# ── ACMG criteria and totals are copied by code, never written by the model ──
#
# The model does not score at this stage. A real observed failure: section 2
# listed a variant with one more criterion (a PP4) and a higher total than
# that variant's own layer block — invented at this stage, then made
# "authoritative" by re-summing the model's own list. Every finding line
# in sections 2 and 4 now gets its criteria and total copied verbatim from its
# own layer block, and any criterion/points/classification line the model
# wrote there is dropped.

_CRITERION_CODE = r"(?:PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])"

# A model-written ACMG line in the conclusion: a criterion bullet in any shape
# ("- PM2 [Moderate, +2]: ...", "*   **PS2** (Strong, +4 pts): ...") or an
# "ACMG criteria"/"ACMG classification"/"ACMG points" line.
_MODEL_ACMG_LINE_RE = re.compile(
    rf"^[ \t]*(?:[-*•][ \t]*)?\**\[?{_CRITERION_CODE}\b\]?\**[ \t]*[\(\[]"
    r"|^[ \t]*(?:[-*•][ \t]*)?\**[ \t]*(?:ACMG criteria|ACMG classification|(?:Total |Base )?ACMG points)\b",
    re.IGNORECASE,
)

# Lines copied out of a layer block: criterion bullets, a layer's
# "... criteria applied:" line when it states the criterion inline, and its
# "... delta:" line.
_BLOCK_CRITERION_RE = re.compile(rf"^[ \t]*[-*•][ \t]*\**\[?{_CRITERION_CODE}\b")
_BLOCK_APPLIED_RE   = re.compile(r"^\*\*[^*\n]*criteria applied:\*\*[ \t]*\S")
_BLOCK_DELTA_RE     = re.compile(r"^\*\*[^*\n]*delta:\*\*")

_CONCLUSION_SECTION_RE = re.compile(r"^[ \t#*]*([1-9])\)")
# A point value the model put on a finding line itself, e.g. "— 5 pts," or
# "with a total ACMG score of 5 points,". "points" must match too: the old
# "pts?" form let a model-invented total through next to the code-copied one.
# A threshold ("≥6 points") or a number glued to a code ("PM3 points") is kept.
_INLINE_POINTS_RE = re.compile(
    r"(?:[ \t]+(?:with|has|of)?[ \t]*(?:a[ \t]+)?(?:total[ \t]+)?(?:ACMG[ \t]+)?(?:score|total)[ \t]+of)?"
    r"[ \t]*[—-]?[ \t]*\[?(?<![>≥<≤=][ \t])(?<![>≥<≤=\w.])[-+]?\d+(?:\.\d+)?\]?[ \t]*(?:points?|pts?)\b"
    r"(?:[ \t]+total)?[ \t]*,?",
    re.IGNORECASE,
)
_CDNA_RE = re.compile(r"c\.[^\s:;|(),]+")


def _block_acmg_entries(
    layer_outputs: dict[str, list[str]]
) -> list[dict]:
    """One entry per variant per layer block: gene, cDNA notations, the
    block's own criteria lines, and its final total/label. A compound-het
    PAIR block yields one entry per "## Variant A/B" segment."""
    entries: list[dict] = []

    def _add(block: str, layer_name: str) -> None:
        joint_m = _JOINT_STATUS_RE.search(block)
        causative_pair = bool(joint_m and joint_m.group(1) == "CAUSATIVE")
        segments = [block]
        if joint_m:
            segments = [seg for seg in re.split(r"(?m)^(?=## )", block) if seg.startswith("## ")]
        for seg in segments:
            header_m = _HEADER_LINE_RE.search(seg)
            total_ms = list(_ANY_TOTAL_RE.finditer(seg))
            if not header_m or not total_ms:
                continue
            try:
                points = float(total_ms[-1].group(1))
            except ValueError:
                continue
            label = total_ms[-1].group(2).strip()
            lines = [
                line.strip() for line in seg.splitlines()
                if _BLOCK_CRITERION_RE.match(line)
                or _BLOCK_APPLIED_RE.match(line)
                or _BLOCK_DELTA_RE.match(line)
            ]
            # Recessive PM3 applied as a hypothesis (phase unconfirmed): the
            # classification is "Potential ..." — see clinical_conclusion.txt's
            # HYPOTHESIS-PM3 QUALIFIER rule.
            if (label in ("Pathogenic", "Likely Pathogenic")
                    and any(l.startswith("**") and "applied:" in l and "hypothesis" in l for l in lines)):
                label = f"Potential {label}"
            entries.append({
                "gene":   header_m.group("gene").split("—")[-1].strip(),
                "cdnas":  frozenset(_CDNA_RE.findall(header_m.group("detail"))),
                "points": points,
                "label":  label,
                "lines":  lines,
                "layer":  layer_name,
                "causative_pair": causative_pair,
            })

    for layer_name, blocks in layer_outputs.items():
        for block in blocks:
            _add(block, layer_name)
    return entries


def _match_block_entry(gene: str, finding_line: str, entries: list[dict]) -> dict | None:
    """The layer-block entry for one finding line, or None when it does not
    match exactly one variant. The finding line's own cDNA picks the variant
    when the gene has several; the same variant scored in several layers
    takes a CAUSATIVE compound-het pair block first, then the highest total."""
    cands = [e for e in entries if e["gene"] == gene]
    line_cdnas = set(_CDNA_RE.findall(finding_line))
    if line_cdnas:
        cands = [e for e in cands if e["cdnas"] & line_cdnas]
    if not cands or len({e["cdnas"] for e in cands}) > 1:
        return None
    return max(cands, key=lambda e: (e["causative_pair"], e["points"]))


def _render_block_acmg(gene: str, finding_line: str, entries: list[dict]) -> list[str]:
    """The criteria + classification lines for one finding, copied from the
    matching layer block (see _match_block_entry)."""
    best = _match_block_entry(gene, finding_line, entries)
    if best is None:
        logger.warning(
            "[FinalConclusion] No unique layer block for finding line %r — ACMG criteria not inserted.",
            finding_line.strip(),
        )
        return [
            "**ACMG criteria:** not inserted — this finding could not be matched "
            "to exactly one variant's layer block; see the MOI-layer blocks above."
        ]
    return (
        [f"**ACMG criteria ({best['layer']} layer block):**"]
        + best["lines"]
        + [f"**ACMG classification:** {best['label']} ({_fmt_points(best['points'])} pts total)"]
    )


def _insert_block_acmg(
    text: str,
    layer_outputs: dict[str, list[str]],
) -> str:
    """
    Sections 2 and 4 of the Clinical Conclusion: drop every ACMG criterion,
    points, or classification line the model wrote, and insert under each
    finding (after its prose) the criteria and total copied from that
    variant's own layer block. Lines already assembled by code ("... pts
    total → ...", from the completeness check or the deterministic fallback)
    are left as they are.
    """
    entries = _block_acmg_entries(layer_outputs)
    genes = sorted({e["gene"] for e in entries if e["gene"]}, key=len, reverse=True)
    finding_re = None
    if genes:
        gene_alt = "|".join(re.escape(g) for g in genes)
        finding_re = re.compile(
            rf"^[ \t#]*(?:[-*•][ \t]+)?(?:\d+\.[ \t]+)?\**[ \t]*\*?({gene_alt})\b\**\*?[ \t]*\("
        )

    out: list[str] = []
    in_conclusion = False
    section: str | None = None
    pending: tuple[str, str] | None = None
    # Section 4 is VUS with a block total in [4, 6) only. The model picks the
    # entries from its own reading of the totals and has listed a 3-point
    # variant as "5.5 points"; an entry whose block total is out of range is
    # dropped with its prose, by the block's number, not the model's.
    skipping = False
    kept_vus = dropped_vus = 0

    def _flush() -> None:
        nonlocal pending
        if pending is None:
            return
        trailing_blank = []
        while out and not out[-1].strip():
            trailing_blank.append(out.pop())
        out.extend(_render_block_acmg(pending[0], pending[1], entries))
        out.extend(trailing_blank)
        pending = None

    for line in text.split("\n"):
        if line.strip().startswith("# Clinical Conclusion"):
            in_conclusion = True
        sm = _CONCLUSION_SECTION_RE.match(line) if in_conclusion else None
        if sm:
            _flush()
            skipping = False
            if section == "4" and dropped_vus and not kept_vus:
                out.extend(["None identified.", ""])
            section = sm.group(1)
            out.append(line)
            continue
        if section in ("2", "4"):
            if _MODEL_ACMG_LINE_RE.match(line):
                continue
            fm = finding_re.match(line) if finding_re else None
            if fm and "pts total →" not in line:
                _flush()
                skipping = False
                if section == "4":
                    best = _match_block_entry(fm.group(1), line, entries)
                    if best is not None and not (4 <= best["points"] < 6):
                        logger.warning(
                            "[FinalConclusion] Section 4 entry %r dropped: block total %s is outside [4, 6).",
                            line.strip()[:120], _fmt_points(best["points"]),
                        )
                        skipping = True
                        dropped_vus += 1
                        continue
                    kept_vus += 1
                line = re.sub(r"(?<=\S)[ \t]{2,}", " ", _INLINE_POINTS_RE.sub(" ", line)).rstrip()
                pending = (fm.group(1), line)
            elif skipping and line.strip():
                continue
        out.append(line)
    _flush()
    if section == "4" and dropped_vus and not kept_vus:
        out.append("None identified.")
    return "\n".join(out)


def _extract_variant_finding(block: str) -> dict | None:
    """
    Best-effort deterministic extraction of (gene, detail, points, label)
    from one layer/base block's own header line and its own final total
    line (the LAST "ACMG points"/"Total ACMG points" match — a MOI-layer
    block states a "Base ACMG points" line first and the delta-adjusted
    "Total ACMG points" line after it; the last match is always the correct,
    fully-adjusted one). Returns None if either piece can't be found — a
    missing finding here is preferable to a fabricated one.
    """
    header_m = _HEADER_LINE_RE.search(block)
    total_ms = list(_ANY_TOTAL_RE.finditer(block))
    if not header_m or not total_ms:
        return None
    points_str, label = total_ms[-1].groups()
    try:
        points = float(points_str)
    except ValueError:
        return None
    return {
        "gene":   header_m.group("gene").strip(),
        "detail": header_m.group("detail").strip(),
        "points": points,
        "label":  label.strip(),
    }


def _deterministic_fallback(
    layer_outputs: dict[str, list[str]],
    patient_phenotype: str,
    actionable_text: str,
) -> str:
    """
    Last-resort synthesis, guaranteed to pass _is_well_formed(), assembled
    with zero LLM involvement so it cannot itself ramble or truncate. Trades
    the LLM's prose/citations/segregation narrative for a guarantee that the
    report never ships with no answer at all.
    """
    findings = []
    for layer_name, blocks in layer_outputs.items():
        for block in blocks:
            f = _extract_variant_finding(block)
            if f:
                f["layer"] = layer_name
                findings.append(f)

    causative = [f for f in findings if f["points"] >= 6]
    vus       = [f for f in findings if 4 <= f["points"] < 6]

    lines = [
        "# Clinical Conclusion",
        "",
        "[AUTOMATED FALLBACK — the clinical-conclusion synthesis stage did not "
        "produce a well-formed response even after a retry; this section was "
        "assembled mechanically from each finding's own already-verified ACMG "
        "total rather than narrative synthesis. Consult the MOI-layer blocks "
        "above for full evidence, segregation, and citations.]",
        "",
        f"1) Summary of clinical phenotype: {patient_phenotype}",
        "",
        "2) Causative variant(s):",
    ]
    if causative:
        for f in causative:
            lines.append(
                f"- {f['gene']} ({f['detail']}) — [{f['layer']} layer] "
                f"{_fmt_points(f['points'])} pts total → {f['label']}"
            )
    else:
        lines.append("None identified.")
    lines += [
        "",
        "3) Actionable findings (ACMG SF):",
        actionable_text,
        "",
        "4) Notable VUS (ACMG >= 4 and < 6 points):",
    ]
    if vus:
        for f in vus:
            lines.append(f"- {f['gene']} ({f['detail']}) — {_fmt_points(f['points'])} pts")
    else:
        lines.append("None identified.")
    lines += [
        "",
        "5) Summary: See causative variant(s) in section 2 above — full "
        "narrative synthesis was unavailable for this run; refer to each "
        "variant's own MOI-layer block for detailed evidence.",
    ]
    return "\n".join(lines)


def run(
    layer_outputs: dict[str, list[str]],
    patient_phenotype: str,
    llm: "LLMClient",
    actionable_variants: list[dict] | None = None,
) -> str:
    """
    Cross-MOI clinical conclusion synthesis (Layer 8).

    Args:
        layer_outputs:     {layer_name: [block, ...]} — one entry per MOI layer
                           that produced at least one finding (e.g. "De Novo",
                           "Dominant-Inherited", "Recessive", "X-Linked"). Each
                           block already carries its own "Total ACMG points:
                           N → Classification" line (base + that layer's delta).
        patient_phenotype: Free-text patient phenotype string from the request.
        llm:               Shared LLMClient instance.
        actionable_variants: ACMG SF actionable-gene findings (gene, hgvs,
                           condition, zygosity, classification dicts) — ground
                           truth for the Clinical Conclusion's own "Actionable
                           findings" section, computed independently upstream
                           (pipeline.py's acmg_sf.build_actionable_set).

    Returns:
        The "# Clinical Conclusion" section, now naming which MOI layer(s)
        jointly explain the phenotype rather than picking one variant from a
        flat list.
    """
    logger.info("[FinalConclusion] Synthesising cross-MOI clinical conclusion...")

    # Stage A — MAP: each layer synthesized independently/in parallel (see
    # _synthesize_layers_parallel's docstring). Replaces the previous direct
    # _build_layers_text(layer_outputs) call, which
    # concatenated every raw block from every layer into one prompt.
    layer_synth = _synthesize_layers_parallel(
        layer_outputs, patient_phenotype, llm,
    )
    conclusions_text = _build_layers_text_from_synth(layer_synth)

    if len(conclusions_text) > _MAX_CONCLUSIONS_CHARS:
        # Last-resort safety net only now — the map stage already dropped
        # everything below the VUS floor, so hitting this budget after that
        # filtering means a genuinely enormous number of >=4pt findings, not
        # the common case this used to guard against.
        logger.warning(
            "[FinalConclusion] Combined layer outputs (%d chars) exceeds budget — truncating.",
            len(conclusions_text),
        )
        conclusions_text = conclusions_text[:_MAX_CONCLUSIONS_CHARS] + "\n\n[... truncated ...]"

    actionable_text = _build_actionable_text(actionable_variants)

    template    = _load_prompt()
    user_prompt = (template
        .replace("{patient_phenotype}", patient_phenotype)
        .replace("{conclusions}", conclusions_text)
        .replace("{actionable_variants}", actionable_text))

    draft = llm.generate(
        system="You are an expert clinical geneticist. Limit your response to 1000 words maximum.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_FINAL_CONCLUSION,
    )

    # Second pass: hand the draft back to the model as a fresh call and ask it to
    # check for contradictions against the per-variant conclusions. A self-check
    # instruction folded into the same generation as the draft doesn't reliably
    # catch its own mistakes (single-pass, buried instruction); a separate call
    # with the draft as input text to critique works — same pattern as the
    # /chat follow-up endpoint, which reliably corrects these when asked directly.
    revise_template = _load_prompt(_REVISE_PROMPT_PATH)
    revise_prompt = (revise_template
        .replace("{patient_phenotype}", patient_phenotype)
        .replace("{conclusions}", conclusions_text)
        .replace("{actionable_variants}", actionable_text)
        .replace("{draft}", draft))

    def _finalize(candidate: str) -> str:
        fixed = candidate

        # Deterministic scan first (cheap, no SLM call): only fire the
        # reconcile pass when something is actually missing.
        findings = _qualifying_causative_findings(layer_outputs)
        missing = _missing_from_section_2(fixed, findings)
        if missing:
            reconciled = _reconcile_missing_causative(
                fixed, missing, llm, patient_phenotype, actionable_text,
            )
            if reconciled is not None:
                fixed = reconciled

        # Safety net either way: a no-op if the reconcile pass (or the
        # absence of any missing finding) already left nothing missing;
        # otherwise mechanically appends whatever still wasn't incorporated.
        fixed = _enforce_causative_completeness(fixed, layer_outputs)

        # ACMG criteria/totals in sections 2 and 4 come from the layer
        # blocks, never from the model (see _insert_block_acmg).
        return _insert_block_acmg(fixed, layer_outputs)

    def _recover_from_malformed_draft() -> str:
        """
        Both draft and revised failed _is_well_formed() — a real recurring
        failure (see clinical_conclusion.txt's borderline-variant/arithmetic-rambling
        notes): the model burns its whole token budget re-litigating a
        single variant and never reaches "# Clinical Conclusion" at all.
        Retry the draft generation once more (fresh call, same prompt —
        Qwen3.5 is non-deterministic enough at temperature>0 that a repeat
        call frequently avoids the same rabbit hole) before giving up. If
        the retry also fails, fall back to a deterministic, zero-LLM
        synthesis rather than shipping "whichever malformed pass happened
        to be longer" — a raw rambling transcript with no actual answer.
        """
        logger.warning(
            "[FinalConclusion] Both draft and revise passes are malformed/truncated "
            "(missing a numbered section) — retrying draft generation once more."
        )
        retry_draft = llm.generate(
            system=(
                "You are an expert clinical geneticist. Limit your response to "
                "1000 words maximum. Decide each variant's status ONCE and move "
                "on immediately — do not write multiple rounds of "
                "reconsideration or re-derive any stated ACMG total."
            ),
            user=user_prompt,
            max_tokens=MAX_NEW_TOKENS_FINAL_CONCLUSION,
        )
        if _is_well_formed(retry_draft):
            logger.info("[FinalConclusion] Retry succeeded — using retried draft.")
            return _finalize(retry_draft)

        logger.warning(
            "[FinalConclusion] Retry also malformed — falling back to a "
            "deterministic conclusion assembled from each layer's own "
            "already-verified totals."
        )
        return _enforce_causative_completeness(
            _deterministic_fallback(
                layer_outputs, patient_phenotype, actionable_text,
            ),
            layer_outputs,
        )

    try:
        revised = llm.generate(
            system="You are an expert clinical geneticist, fact-checking a draft report against source data.",
            user=revise_prompt,
            max_tokens=MAX_NEW_TOKENS_FINAL_CONCLUSION,
        )
    except Exception as exc:
        logger.warning("[FinalConclusion] Revise pass failed (%s) — using unrevised draft.", exc)
        if _is_well_formed(draft):
            return _finalize(draft)
        return _recover_from_malformed_draft()

    if not _is_well_formed(revised):
        if _is_well_formed(draft):
            logger.warning(
                "[FinalConclusion] Revise pass produced malformed/truncated output "
                "(missing a numbered section) — using unrevised draft."
            )
            return _finalize(draft)
        return _recover_from_malformed_draft()

    # Re-check causative-list completeness against the full layer-output set,
    # then replace whatever ACMG content the model wrote in sections 2 and 4
    # with each finding's own layer-block criteria and total, before handing
    # the report to the user. See _enforce_causative_completeness and
    # _insert_block_acmg above.
    return _finalize(revised)

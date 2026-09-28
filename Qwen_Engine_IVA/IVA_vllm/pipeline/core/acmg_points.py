"""
pipeline/core/acmg_points.py — shared "**ACMG points:** N → Classification"
line parsing/rewriting, used by every mechanical ACMG-criterion validator
(acmg_pp3.py, acmg_pp2_bp1.py, ...) that strips a criterion the SLM applied
without adequate grounding and needs to keep the stated total consistent.
"""

import logging
import re

logger = logging.getLogger(__name__)

_POINTS_LINE_RE = re.compile(
    r"(\*\*(?:Total )?ACMG points:\*\*\s*)([-+]?\d+(?:\.\d+)?)(\s*→\s*)([A-Za-z /()]+)"
)

# moi_*.py's "**Base ACMG points:** N" line (the copied-verbatim base total,
# before that layer's own delta) — a third total-line format alongside
# _POINTS_LINE_RE ("ACMG points:"/"Total ACMG points:") and
# _CLASSIFICATION_LINE_RE below ("ACMG classification:"). Kept separate from
# _POINTS_LINE_RE (rather than making "Total" one of several optional
# prefixes) so adjust_points_line()/relabel_all_points_lines() — used
# elsewhere against the "Total" total specifically — don't start matching
# the "Base" line too.
#
# The trailing "→ Label" is OPTIONAL here, unlike _POINTS_LINE_RE, because
# every moi_*.py prompt template (moi_dominant.txt, moi_recessive.txt,
# moi_denovo.txt, moi_xlinked.txt, moi_recessive_homozygous.txt) explicitly
# instructs this line as a bare copied number with no label — "**Base ACMG
# points:** [copy the numeric value ... verbatim]" — never "N → Label". A
# real, significant observed failure: this regex previously REQUIRED the
# arrow+label suffix, so it silently never matched the Base line's actual
# real-world shape in ANY MOI layer block, meaning recompute_and_fix_totals's
# Base-line correction (first pass, below) and its Total-line reconciliation
# (second pass, which depends on successfully matching the Base line to read
# its value) both silently no-op'd for every single moi_*.py block ever
# generated — the mechanical arithmetic safety net this function exists to
# provide was never actually running for MOI-layer totals at all.
_BASE_POINTS_LINE_RE = re.compile(
    r"(\*\*Base ACMG points:\*\*\s*)([-+]?\d+(?:\.\d+)?)(\s*→\s*[A-Za-z /()]+)?"
)

# final_conclusion.py's per-variant format: "**ACMG classification:** Label
# (N pts total)" — distinct from the "**ACMG points:** N -> Label" format
# above (conclusion.py / moi_*.py), so recompute_and_fix_totals() needs to
# recognize and rewrite both.
_CLASSIFICATION_LINE_RE = re.compile(
    r"(\*\*ACMG classification:\*\*\s*)([A-Za-z /()]+?)(\s*\(\s*)([-+]?\d+(?:\.\d+)?)(\s*pts total\s*\))"
)

# A criterion bullet's own point tag, e.g. "[VeryStrong, +8]" or
# "[Moderate, +2 pts]" — every criterion line carries exactly one of these
# per the fixed format both conclusion.txt and clinical_conclusion.txt
# require.
_CRITERION_TAG_RE = re.compile(r"\[[A-Za-z\s]+,\s*([+-]?\d+(?:\.\d+)?)\s*(?:pts?)?\]")

# Every standard ACMG/AMP criterion code.
_CRITERION_CODE_RE = r"(?:PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])"

# ── ACMG criterion recognizer ───────────────────────────────────────────────
#
# One recognizer for "an applied ACMG criterion", shared by gather_criteria()
# (totals re-sum) and extract_base_acmg() (base block copy). It replaced two
# independent regexes that each broke on SLM layout drift: a whole-block regex
# that needed "-" bullets with no blank line between heading, bullets and
# total, and a mention regex that needed the [ or ( tag DIRECTLY after the
# code, so it silently dropped the prompt's own "PVS1 VeryStrong (+8 pts)"
# shape and every "PVS1_Moderate" suffix form.
#
# Instead of one pattern for the whole line, it anchors on the criterion
# code, then reads the tag after it as a sequence of small tokens — a
# strength word, a [..]/(..) group, a signed number or "N pts" — in any
# order and any markup, and stops at the first thing that is not a tag
# token, so the justification text after the tag is never read as one.
#
# A code mention counts as an applied criterion when its tag carries a point
# value. Inside a known criteria list (in_list=True: the "ACMG criteria"
# section extract_base_acmg() reads), also when the code opens a list item
# (bullet, numbered item, start of a "criteria:" line or a ";" segment) and
# either its tag carries only a strength word, or it has no tag and is
# followed directly by ":" ("- PM2: Absent from controls") — the code's
# default strength then applies. Outside such a list those two shapes are
# prose as often as not ("**PS3 (Strong):** the evidence does not support
# it"), so they don't count. Never counted: a negated mention — "not
# applied"/"N/A"/... in its tag, right after it, or opening its
# justification ("- PVS1 [VeryStrong, +8]: Not applicable; ..." is a real,
# frequent SLM shape) — and prose ("supported by prior evidence (PS1)",
# "**PS1 Application:** ...").
#
# Points: the stated number, when present; otherwise the strength's value.
# A stated number that disagrees with every strength word in the same tag
# gives the lower of the two (logged) — a conflict never adds points neither
# side supports. Benign codes always count negative; BA1 is always
# stand-alone (-8). A pathogenic code with a negative number is the SLM
# arguing against it ("PM2 [Absent, -2]: common polymorphism"), not applying
# it — not counted.

_STRENGTH_WORD = r"very[ \t_-]?strong|stand[ \t_-]?alone|strong|moderate|supporting"
_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(" + _CRITERION_CODE_RE[3:-1] + r")"
    r"(?:[_-](" + "(?i:" + _STRENGTH_WORD + ")" + r"))?(?![A-Za-z0-9])"
)
_STRENGTH_RE = re.compile(r"(?i)(?<![A-Za-z])(" + _STRENGTH_WORD + r")(?![A-Za-z])")
_STRENGTH_POINTS = {"verystrong": 8.0, "standalone": 8.0, "strong": 4.0, "moderate": 2.0, "supporting": 1.0}
_STRENGTH_LABEL = {"verystrong": "VeryStrong", "standalone": "Standalone", "strong": "Strong",
                   "moderate": "Moderate", "supporting": "Supporting"}
_POINTS_STRENGTH = {8.0: "verystrong", 4.0: "strong", 2.0: "moderate", 1.0: "supporting"}
_DEFAULT_STRENGTH = {"PVS": "verystrong", "PS": "strong", "PM": "moderate", "PP": "supporting",
                     "BA": "standalone", "BS": "strong", "BP": "supporting"}

_MINUS = "−–-"  # "-" last: inside [+...] it must not form a range
# Number inside a [..]/(..) group: signed, or followed by pt/pts/points, or
# (only when the group also names a strength) a lone bare 0-8.
_GROUP_NUM_RE = re.compile(
    r"(?<![\w.:/])([+" + _MINUS + r"]?)\s*(\d+(?:\.\d+)?)(?![\w.])(\s*(?:pts?|points?)\b)?", re.IGNORECASE
)
_BARE_STRENGTH_RE = re.compile(r"(?i)(" + _STRENGTH_WORD + r")(?:[ \t_-]+(?:benign|pathogenic))?(?![A-Za-z])")
# Bare number outside a group: sign glued to the digit ("+4", not "- 3
# probands"), or "N pts".
_BARE_NUM_RE = re.compile(
    r"(?i)(?:([+" + _MINUS + r"])(\d+(?:\.\d+)?)(?:\s*(?:pts?|points?)\b)?|(\d+(?:\.\d+)?)\s*(?:pts?|points?)\b)"
)
_GROUP_RE = re.compile(r"[\(\[]([^\(\)\[\]\n]{0,100})[\)\]]")
_TAG_LEAD_RE = re.compile(r"[ \t*_\]\)]*")
_TAG_COLON_RE = re.compile(r":[ \t*_]*")
_TAG_SEP_RE = re.compile(r"[ \t*_,=/|]*")
# A clause that withdraws a criterion: the opening of its justification
# ("- PVS1 [VeryStrong, +8]: Not applicable; ...") or one part of its tag
# ("(not applied)"). Anchored at the clause start and kept to explicit
# phrases, since the same words mid-clause don't withdraw anything
# ("consanguinity not excluded", "does not appear in gnomAD" opens a valid PM2).
_NEGATION_RE = re.compile(
    r"(?i)(?:this\s+)?(?:criterion\s+)?(?:is\s+|was\s+)?(?:therefore\s+)?"
    r"(?:not\s+(?:applicable|applied|met|satisfied|fulfilled|warranted|used|assigned|invoked)\b"
    r"|n/?a\b|inapplicable\b|(?:does|do)\s+not\s+(?:apply|meet|qualify)\b"
    r"|(?:cannot|can't|should\s+not|must\s+not)\s+be\s+applied\b"
    r"|(?:withheld|excluded|withdrawn|removed|stripped)\b)"
)
# Phrases that withdraw a criterion wherever they sit in the justification's
# first clause ("PM2 [Moderate, -2]: Absent from controls is not applicable").
_NEGATION_ANYWHERE_RE = re.compile(r"(?i)\b(?:not\s+(?:applicable|applied|met)|(?:does|do)\s+not\s+apply|n/a)\b")
# What may precede a code on its line for it to "open a list item": markup,
# bullet/number markers, a "... criteria ...:" label, or a ";" segment break.
_ITEM_PREFIX_RE = re.compile(r"^[ \t>#]*(?:(?:[-*•·+]|\d+[.)])[ \t]*)?[*_\[ \t]*$")
_CRITERIA_LABEL_RE = re.compile(r"(?i)criteria[^:\n]{0,60}:[*_ \t]*")


class Criterion:
    """One applied ACMG criterion found in text. `start` is the code's offset
    in the scanned text, `tag_end` where its tag ends (justification follows)."""
    __slots__ = ("code", "points", "strength", "start", "tag_end")

    def __init__(self, code, points, strength, start, tag_end):
        self.code, self.points, self.strength = code, points, strength
        self.start, self.tag_end = start, tag_end

    def tag(self) -> str:
        """Canonical "[Strength, +N]" tag ("[+N]" when no strength fits N)."""
        pts = ("+" if self.points >= 0 else "") + _sum_str(self.points)
        return f"[{_STRENGTH_LABEL[self.strength]}, {pts}]" if self.strength else f"[{pts}]"


def _to_float(sign: str, digits: str) -> float:
    return -float(digits) if sign and sign in _MINUS else float(digits)


def _read_group(content: str) -> tuple[list[str], list[float], bool] | None:
    """Strength words and point values in one [..]/(..) group, plus whether it
    is negated. None when the group holds neither — then it is not a tag."""
    strengths = [re.sub(r"[^a-z]", "", s.lower()) for s in _STRENGTH_RE.findall(content)]
    nums = []
    bare = []
    for sign, digits, pts in _GROUP_NUM_RE.findall(content):
        if sign or pts:
            nums.append(_to_float(sign, digits))
        else:
            bare.append(float(digits))
    if not nums and strengths and len(bare) == 1:
        nums = bare
    nums = [n for n in nums if abs(n) <= 8]
    negated = any(_NEGATION_RE.match(part.strip()) for part in re.split(r"[,;]", content))
    if not strengths and not nums and not negated:
        return None
    return strengths, nums, negated


def _read_tag(line: str, pos: int) -> tuple[list[str], list[float], bool, int]:
    """Reads the tag tokens after a code ending at `pos`. Returns (strength
    keys, numbers, negated, end offset of the last tag token or `pos`)."""
    strengths, nums, negated = [], [], False
    end = pos
    i = _TAG_LEAD_RE.match(line, pos).end()
    colon = _TAG_COLON_RE.match(line, i)
    if colon:
        i = colon.end()
    for _ in range(4):
        m = _GROUP_RE.match(line, i)
        if m:
            got = _read_group(m.group(1))
            if got is None:
                break
            strengths += got[0]
            nums += got[1]
            negated = negated or got[2]
        elif (m := _BARE_STRENGTH_RE.match(line, i)):
            strengths.append(re.sub(r"[^a-z]", "", m.group(1).lower()))
        elif (m := _BARE_NUM_RE.match(line, i)):
            n = _to_float(m.group(1), m.group(2)) if m.group(2) else float(m.group(3))
            if abs(n) > 8:
                break
            nums.append(n)
        else:
            break
        end = m.end()
        i = _TAG_SEP_RE.match(line, end).end()
    return strengths, nums, negated, end


def _opens_item(prefix: str) -> bool:
    """True when `prefix` (the line text before a code) leaves the code at the
    start of a list item."""
    prefix = prefix.rsplit(";", 1)[-1]
    labels = list(_CRITERIA_LABEL_RE.finditer(prefix))
    if labels:
        prefix = prefix[labels[-1].end():]
    return bool(_ITEM_PREFIX_RE.match(prefix))


def _justification_negated(line: str, tag_end: int) -> bool:
    """True when the justification after the tag opens by withdrawing it."""
    rest = re.sub(r"^[ \t*_:—–\-]*", "", line[tag_end:])
    clause = re.split(r"[;.]", rest, maxsplit=1)[0][:80]
    return bool(_NEGATION_RE.match(clause) or _NEGATION_ANYWHERE_RE.search(clause))


def find_criteria(text: str, in_list: bool = False) -> list[Criterion]:
    """Every applied ACMG criterion in `text`, in order (not deduplicated).
    in_list=True when `text` is a criteria list (see the rules above)."""
    found = []
    offset = 0
    for line in text.splitlines(keepends=True):
        for m in _CODE_RE.finditer(line):
            code = m.group(1)
            if line[max(0, m.start() - 2):m.start()] == "~~":
                continue
            strengths, nums, negated, tag_end = _read_tag(line, m.end())
            if m.group(2):
                strengths.insert(0, re.sub(r"[^a-z]", "", m.group(2).lower()))
            if negated or _justification_negated(line, tag_end):
                continue
            if not nums:
                if not (in_list and _opens_item(line[:m.start()])):
                    continue
                if not strengths and not re.match(r"[ \t*_\]\)]*:", line[m.end():]):
                    continue
            sign = -1.0 if code[0] == "B" else 1.0
            if code == "BA1":  # stand-alone by definition, whatever word the SLM put on it
                strengths, nums = ["standalone"], []
            strength = strengths[-1] if strengths else _DEFAULT_STRENGTH[code[:2] if code[:3] != "PVS" else "PVS"]
            if nums:
                if code[0] == "P" and nums[-1] < 0:
                    continue
                points = abs(nums[-1])
                if strengths and not any(_STRENGTH_POINTS[s] == points for s in strengths):
                    lower = min(points, _STRENGTH_POINTS[strength])
                    logger.warning(
                        "acmg_points.find_criteria: %s states %s pts but strength %s — using the lower, %s",
                        code, _sum_str(nums[-1]), _STRENGTH_LABEL[strength], _sum_str(lower),
                    )
                    points = lower
                if _STRENGTH_POINTS.get(strength) != points:
                    strength = _POINTS_STRENGTH.get(points)
            else:
                points = _STRENGTH_POINTS[strength]
            found.append(Criterion(code, sign * points, strength, offset + m.start(), offset + tag_end))
        offset += len(line)
    return found


def gather_criteria(text: str) -> list[tuple[str, float]]:
    """
    (code, points) for every applied ACMG criterion in `text` (see
    find_criteria), in first-seen order, one per code. A code restated with
    the SAME points is a redundant echo and is dropped; restated with
    DIFFERENT points is a genuine inconsistency in the SLM's output — the
    first occurrence wins and the disagreement is logged.
    """
    seen: dict[str, float] = {}
    order: list[str] = []
    for c in find_criteria(text):
        if c.code in seen:
            if seen[c.code] != c.points:
                logger.warning(
                    "acmg_points.gather_criteria: %s mentioned twice with "
                    "different point values (%s vs %s) in the same block — "
                    "keeping the first, dropping the duplicate",
                    c.code, seen[c.code], c.points,
                )
            continue
        seen[c.code] = c.points
        order.append(c.code)
    return [(code, seen[code]) for code in order]


def sum_criteria(text: str) -> float:
    """Sum of gather_criteria(text)'s deduplicated per-code point values."""
    return sum(points for _, points in gather_criteria(text))


# moi_*.py's own "**Base ACMG criteria (...):**" header — the boundary a
# Base-line's criteria are scoped to start from (excludes that layer's own
# earlier "<Layer> criteria applied" bullet, e.g. PS2, which must count
# toward the Total line but not the Base line). Worded slightly differently
# across templates ("BASE CONCLUSION's" vs "the BASE CONCLUSION's") — match
# on the stable "Base ACMG criteria" prefix only.
_BASE_HEADER_RE = re.compile(r"\*\*Base ACMG criteria\b[^\n]*\*\*")

# The header that opens a self-contained "one variant's worth of ACMG
# criteria" sub-block, in either of the two shapes the pipeline renders:
# moi_*.py's "**<Layer> criteria applied:**" (e.g. "De novo criteria
# applied") or conclusion.py's/final_conclusion.py's own "**ACMG
# criteria:**". Used to bound how far back recompute_and_fix_totals()
# looks for criteria belonging to a given total line, so a concatenated
# multi-variant text (final_conclusion.py's combined output, or a
# compound-het pair's two variant blocks in one moi_recessive.py result)
# doesn't pull an earlier variant's criteria into a later variant's total.
_BLOCK_START_RE = re.compile(
    r"\*\*(?:[A-Za-z][A-Za-z ]* criteria applied|ACMG criteria):\*\*"
)

# (inclusive lower bound, label) — highest first; matches the thresholds
# block at the bottom of prompts/conclusion.txt.
_THRESHOLDS = [
    (10.0, "Pathogenic"),
    (6.0,  "Likely Pathogenic"),
    (0.0,  "Uncertain Significance (VUS)"),
    (-6.0, "Likely Benign"),
    (float("-inf"), "Benign"),
]


def classify(points: float) -> str:
    for lo, label in _THRESHOLDS:
        if points >= lo:
            return label
    return "Benign"


def adjust_points_line(text: str, delta: float) -> str:
    """
    Adds `delta` to the stated "**ACMG points:** N → Label" total and
    rewrites the classification label to match. No-op if the line isn't
    found. Safe to call repeatedly (e.g. once per stripped criterion) since
    it re-reads the current value from `text` each time.
    """
    def _adjust(m: re.Match) -> str:
        prefix, points_str, arrow, _old_label = m.groups()
        try:
            new_points = float(points_str) + delta
        except ValueError:
            return m.group(0)
        new_points_str = str(int(new_points)) if new_points == int(new_points) else str(new_points)
        return f"{prefix}{new_points_str}{arrow}{classify(new_points)}"

    return _POINTS_LINE_RE.sub(_adjust, text, count=1)


def relabel_all_points_lines(text: str) -> str:
    """
    Re-derives the classification label on EVERY "**[Total ]ACMG points:** N →
    Label" line in `text` from N via classify(), leaving N itself unchanged.
    Fixes the SLM occasionally writing an internally-inconsistent label for
    its own stated total (e.g. "4.5 → Likely Pathogenic" when 4.5 is in the
    0-5 VUS band, not the 6-9 Likely Pathogenic band) — a real observed
    failure where a homozygous ClinVar Likely-Pathogenic variant scored 4.5
    points, got mislabeled "Likely Pathogenic" instead of "Uncertain
    Significance (VUS)", and as a result fell through every section of the
    Clinical Conclusion (not causative since 4.5 < 6, not Notable VUS since
    its label wasn't "VUS", so the SCOPE RULE omitted it from the report
    entirely). Unlike adjust_points_line(), fixes ALL matches in the text
    (a compound-het pair block has two such lines), not just the first.
    """

    def _relabel(m: re.Match) -> str:
        prefix, points_str, arrow, _old_label = m.groups()
        try:
            points = float(points_str)
        except ValueError:
            return m.group(0)
        return f"{prefix}{points_str}{arrow}{classify(points)}"

    return _POINTS_LINE_RE.sub(_relabel, text)


def _sum_str(n: float) -> str:
    return str(int(n)) if n == int(n) else str(n)


def recompute_and_fix_totals(text: str) -> str:
    """
    Deterministically re-sums the ACMG criteria that belong to each stated
    total line via gather_criteria() — a position-independent scan for
    every criterion mention in the relevant scope, deduplicated by code —
    and overwrites that total, and its classification label, whenever it
    disagrees with the SLM's own stated number.

    A real, RECURRING observed failure — the exact scenario conclusion.txt's
    own "real past failure" warning already describes, which happened again
    anyway: a criteria list totalling PS3(+4)+PM2(+2)+PP4(+1) = 7 stated as
    "ACMG points: 11". The stale 11 then carried forward unchanged through a
    MOI layer's "+2" delta into a final reported total of 13, with every
    downstream stage trusting the upstream total rather than the actual
    listed criteria. Prompt instructions to self-check this arithmetic
    exist and are followed only sometimes — this makes the check
    unconditional.

    This replaces an earlier, position-based version (walk back over the
    contiguous run of non-blank lines immediately above the total) that
    broke silently the instant the SLM's own rendering deviated even
    slightly from the prompt template's exact spacing — e.g. a single blank
    line the SLM inserted before "**Base ACMG points:**" stopped the
    backward scan before it ever reached the criteria bullets, leaving a
    stale, wrong total (and the classification derived from it) shipped
    unchanged, with no error or warning anywhere. gather_criteria() instead
    finds every criterion mention within a bounded scope regardless of
    exactly where it sits relative to blank lines or restated delta lines.

    Handles all three rendered total-line formats used across the pipeline,
    each with its own scope:
    - "**ACMG points:**" (conclusion.py) / "**ACMG classification:** Label
      (N pts total)" (final_conclusion.py) — scoped from the nearest
      preceding "**ACMG criteria:**" header to the total line, so a
      concatenated multi-variant text (final_conclusion.py's combined
      output) doesn't pull an earlier variant's criteria into this one's
      total.
    - "**Base ACMG points:** N" (each moi_*.py layer's copied-verbatim base
      total) — scoped from the nearest preceding "**Base ACMG criteria**"
      header, which deliberately excludes that layer's own earlier
      "<Layer> criteria applied" delta bullet (e.g. PS2) — the Base line
      must reflect the base score alone.
    - "**Total ACMG points:** N" (moi_*.py's base+delta total) — scoped
      from the nearest preceding "**<Layer> criteria applied:**" header,
      which — unlike the Base line's scope — DOES include that layer's own
      delta bullet, giving base + delta in one pass without needing a
      separate reconciliation step.
    Falls back to scoping from the start of `text` if no bounding header is
    found, matching the old function's behavior for any block shape that
    predates this convention.
    """

    out = text

    # (regex to match the total line, header marking the start of its
    #  criteria scope, index of the group holding the currently-stated
    #  numeric value, rewrite function given (match, corrected_total))
    passes = (
        (
            _BASE_POINTS_LINE_RE,
            _BASE_HEADER_RE,
            2,
            lambda m, actual: f"{m.group(1)}{_sum_str(actual)} → {classify(actual)}",
        ),
        (
            _POINTS_LINE_RE,
            _BLOCK_START_RE,
            2,
            lambda m, actual: f"{m.group(1)}{_sum_str(actual)}{m.group(3)}{classify(actual)}",
        ),
        (
            _CLASSIFICATION_LINE_RE,
            _BLOCK_START_RE,
            4,
            lambda m, actual: (
                f"{m.group(1)}{classify(actual)}{m.group(3)}"
                f"{_sum_str(actual)}{m.group(5)}"
            ),
        ),
    )

    for regex, header_re, value_group, fmt in passes:
        pos = 0
        pieces = []
        for m in regex.finditer(out):
            pieces.append(out[pos : m.start()])
            start = 0
            for hm in header_re.finditer(out, 0, m.start()):
                start = hm.end()
            crit = gather_criteria(out[start : m.start()])
            if crit:
                actual = sum(v for _, v in crit)
                if actual != float(m.group(value_group)):
                    pieces.append(fmt(m, actual))
                    pos = m.end()
                    continue
            pieces.append(m.group(0))
            pos = m.end()
        pieces.append(out[pos:])
        out = "".join(pieces)

    return out


# Stage-4 conclusion's criteria heading ("**ACMG criteria:**", "### ACMG
# Criteria", "ACMG criteria applied:"), its points line, and the next field
# header ("**Comment:**", "# Variant 2 ...") that ends the criteria section.
# Only a real list heading: "ACMG criteria", optionally "applied" and/or a
# "(...)" label, an optional colon — then the line ends or the list starts
# on it (a code, "None", a bullet). "- ACMG Criteria definitions: ..." or
# prose mentioning ACMG criteria is not a heading.
_CRITERIA_HEADING_RE = re.compile(
    r"(?im)^[ \t>#*_-]*ACMG[ \t]+criteria(?:[ \t]+applied)?[ \t]*(?:\([^)\n]{0,80}\))?[ \t]*:?[*_ \t]*"
    r"(?=$|(?:PVS|PS|PM|PP|BA|BS|BP)\d|none\b|[-*•\d\[])"
)
_POINTS_HEADING_RE = re.compile(r"(?im)^[ \t>#*_-]*(?:total[ \t]+)?ACMG[ \t]+points\b")
_STATED_POINTS_RE = re.compile(r"(?i)ACMG[ \t]+points\b[^\d\n+\-−–]{0,12}([+\-−–]?\d+(?:\.\d+)?)")
_FIELD_HEADER_RE = re.compile(r"(?m)^[ \t]*(?:\*\*[A-Za-z][^*\n]{0,60}:\*\*|#)")


def criteria_section_bounds(text: str) -> tuple[int, int, bool, bool]:
    """(start, end, has_heading, ends_at_points_line) of the "ACMG criteria"
    section. The heading is the LAST one before the first points line — the
    list that points line totals, the same scope recompute_and_fix_totals
    re-sums — else the first one. The section ends at that points line or at
    the first field header after the heading line ("**ACMG classification:**",
    "**Comment:**", "# ...") that isn't itself a criterion line. Without a
    heading: from the start of the text to the points line (or the end)."""
    first_points = _POINTS_HEADING_RE.search(text)
    before = [h for h in _CRITERIA_HEADING_RE.finditer(text)
              if not first_points or h.start() < first_points.start()]
    h = before[-1] if before else _CRITERIA_HEADING_RE.search(text)
    start = h.end() if h else 0
    p = _POINTS_HEADING_RE.search(text, start)
    end = p.start() if p else len(text)
    at_points = p is not None
    if h:
        body_start = text.find("\n", start)
        for f in _FIELD_HEADER_RE.finditer(text, body_start if body_start != -1 else end, end):
            line_end = text.find("\n", f.start())
            if not _CODE_RE.search(text, f.start(), line_end if line_end != -1 else end):
                end, at_points = f.start(), False
                break
    return start, end, h is not None, at_points


def _criterion_insert_at(text: str) -> int | None:
    """Offset where a new criterion line goes (start of a line), or None: the
    end of the "ACMG criteria" section (criteria_section_bounds) — its points
    line or the next field header. A section that simply runs to the end of
    the text: after its last criterion (or its heading line). No heading:
    before a points line, else after the last criterion."""
    start, end, has_heading, at_points = criteria_section_bounds(text)
    if has_heading:
        if at_points or end < len(text):
            return end
        after = _after_last_criterion(text, start, True)
        if after is not None:
            return after
        body = text.find("\n", start)
        return len(text) if body == -1 else body + 1
    if at_points:
        return end
    return _after_last_criterion(text, 0, False)


def _after_last_criterion(text: str, start: int, in_list: bool) -> int | None:
    found = find_criteria(text[start:], in_list=in_list)
    if not found:
        return None
    end = text.find("\n", start + found[-1].tag_end)
    return len(text) if end == -1 else end + 1


def insert_criterion_line(text: str, line: str) -> str | None:
    """`text` with the criterion `line` ("- CODE [Strength, +N]: ...\\n") added at
    the end of its "ACMG criteria" section (see _criterion_insert_at), or None
    when the text has no criteria section or points line to place it by. The
    caller re-sums totals afterwards (recompute_and_fix_totals)."""
    at = _criterion_insert_at(text)
    if at is None:
        return None
    if at > 0 and text[at - 1] != "\n":
        line = "\n" + line
    return text[:at] + line + text[at:]


def _criteria_section(base_conclusion: str) -> tuple[str, bool]:
    """The conclusion's "ACMG criteria" section text and whether a heading
    was found (see criteria_section_bounds)."""
    start, end, has_heading, _ = criteria_section_bounds(base_conclusion)
    return base_conclusion[start:end], has_heading


def extract_base_acmg(base_conclusion: str) -> tuple[str, float] | None:
    """
    Pulls the applied ACMG criteria out of a variant's own Stage-4
    conclusion.py output (conclusions[i]) and computes their base score —
    the ONE canonical base score for that variant. Every MOI layer (de novo,
    dominant, recessive, X-linked) must splice in this exact block
    mechanically rather than asking the SLM to re-transcribe or re-derive it
    from context on every layer call.

    A real observed failure this replaces: the same variant's "Base
    ACMG points" came out as three different numbers (8, 4, 2) across its
    three MOI-layer blocks in one run, none of which even matched their own
    listed criteria in that same block — despite every layer prompt
    instructing "copy verbatim, do not invent" from the identical frozen
    base_conclusion text. The model was re-deriving the number under each
    layer's framing instead of copying it, and no mechanical check caught
    the disagreement because each layer's "Base ACMG points" line was
    validated only against criteria the model ALSO transcribed itself in
    that same call — a check that can never catch a model that transcribes
    a wrong number and a matching-but-wrong criteria list together.

    Criteria are read by find_criteria() from the "ACMG criteria" section,
    whatever its layout (bullet style, blank lines, one criterion per line
    or ";"-joined, tag shape), and re-rendered one per line as
    "- CODE [Strength, +N]: justification". The points are their sum,
    computed here — never the conclusion's own stated total, which is only
    compared and logged when it disagrees. This replaced a whole-block regex
    that needed "-" bullets with nothing between heading, bullets and total:
    a single blank line or "*" bullet made it return None, no base block was
    spliced, and the final conclusion's LLM made up its own total instead.

    Returns (block_text, points), where block_text is ready to splice in
    as-is — a "**Base ACMG criteria:**" bullet list followed by a "**Base
    ACMG points:** N → Label" line. A section with no applied criterion
    gives a "- None applied" block worth 0. Returns None (logging the raw
    conclusion) only when no criterion is found but the conclusion states a
    non-zero total — a shape nothing here can read, so the caller falls back
    to what the LLM produced rather than splicing a wrong 0.
    """
    section, has_heading = _criteria_section(base_conclusion)
    found = find_criteria(section, in_list=has_heading)
    stated_m = _STATED_POINTS_RE.search(base_conclusion)
    stated = float(re.sub("[−–]", "-", stated_m.group(1))) if stated_m else None

    if not found and stated:
        logger.warning(
            "acmg_points.extract_base_acmg: no criterion recognized but conclusion states %s pts — "
            "base block not built. Raw conclusion:\n%s", _sum_str(stated), base_conclusion[:4000],
        )
        return None

    bullets = []
    kept: dict[str, float] = {}
    for n, c in enumerate(found):
        if c.code in kept:
            if kept[c.code] != c.points:
                logger.warning(
                    "acmg_points.extract_base_acmg: %s listed twice (%s vs %s pts) — keeping the first",
                    c.code, _sum_str(kept[c.code]), _sum_str(c.points),
                )
            continue
        kept[c.code] = c.points
        line_end = section.find("\n", c.tag_end)
        stop = line_end if line_end != -1 else len(section)
        if n + 1 < len(found) and found[n + 1].start < stop:
            stop = found[n + 1].start
        why = re.sub(r"^[ \t*_:—–\-]+", "", section[c.tag_end:stop])
        why = re.sub(r"[ \t;,*_]+$", "", why)
        bullets.append(f"- {c.code} {c.tag()}: {why}" if why else f"- {c.code} {c.tag()}")

    points = sum(kept.values())
    if not has_heading:
        logger.warning("acmg_points.extract_base_acmg: no 'ACMG criteria' heading — read criteria with point tags from the whole conclusion.")
    if stated is not None and stated != points:
        logger.warning(
            "acmg_points.extract_base_acmg: conclusion states %s pts, its criteria sum to %s — using %s. Raw conclusion:\n%s",
            _sum_str(stated), _sum_str(points), _sum_str(points), base_conclusion[:4000],
        )
    block = (
        "**Base ACMG criteria (ground truth — mechanically copied from this "
        "variant's own Stage-4 conclusion; not re-derived at this layer):**\n"
        + ("\n".join(bullets) if bullets else "- None applied") + "\n"
        f"**Base ACMG points:** {_sum_str(points)} → {classify(points)}"
    )
    return block, points


# ── MOI layers: base block spliced by code, Total computed by code ──────────
#
# Every MOI layer (de novo, dominant-inherited, recessive pair, recessive
# homozygous, X-linked) scores the same way: the model writes ONLY its own
# layer criterion and its "**<Layer> delta:** +N" line. Code then drops any
# base criteria / Base points / Total line the model wrote, splices in the
# variant's frozen Stage-4 base block (extract_base_acmg), and computes
# Total = sum of the base bullets + delta. A layer can never add, drop, or
# re-derive a base criterion (e.g. a PP4 the base conclusion withheld).

# "**De novo delta:** +4", "**Recessive delta:** +2 hypothesis", "**X-linked delta:** [+1]"
_MOI_DELTA_LINE_RE = re.compile(
    r"(?m)^\*\*[A-Za-z][A-Za-z -]* delta:\*\*\s*\[?\s*([-+]?\d+(?:\.\d+)?)\s*(confirmed|hypothesis)?[^\n]*$",
    re.IGNORECASE,
)
# Model-written base content: the "Base ACMG criteria" header with its bullets
# (and template placeholder lines), the Base points line, the Total line.
_MODEL_BASE_BLOCK_RE = re.compile(
    r"(?m)^\*\*Base ACMG criteria\b[^\n]*\n(?:[ \t]*(?:[-•]|\*[ \t]|\[repeat)[^\n]*\n?)*"
)
_MODEL_BASE_POINTS_RE = re.compile(r"(?m)^\*\*Base ACMG points:\*\*[^\n]*\n?")
_MODEL_TOTAL_RE = re.compile(r"(?m)^\*\*Total ACMG points:\*\*[^\n]*\n?")


def splice_base_and_total(section: str, base_conclusion: str) -> str:
    """
    For ONE variant's MOI-layer section: removes whatever base criteria, Base
    points, or Total line the model wrote, inserts the Stage-4 base block
    (extract_base_acmg) right after the layer's delta line, and computes the
    Total (see resync_moi_total). No-op, with a warning, if the delta line or
    the base conclusion's own ACMG block can't be found — degrades to what
    the model produced rather than guessing.
    """
    if not _MOI_DELTA_LINE_RE.search(section):
        logger.warning("acmg_points.splice_base_and_total: no '<Layer> delta:' line — base block not spliced.")
        return section
    extracted = extract_base_acmg(base_conclusion)
    if extracted is None:
        logger.warning("acmg_points.splice_base_and_total: no ACMG block in base conclusion — base block not spliced.")
        return section
    base_block, _ = extracted

    original = section
    section = _MODEL_BASE_BLOCK_RE.sub("", section)
    section = _MODEL_BASE_POINTS_RE.sub("", section)
    section = _MODEL_TOTAL_RE.sub("", section)
    dm = _MOI_DELTA_LINE_RE.search(section)
    if not dm:
        logger.warning("acmg_points.splice_base_and_total: delta line lost while removing model base lines — base block not spliced.")
        return original
    section = (section[: dm.end()] + f"\n{base_block}\n**Total ACMG points:** 0 → "
               + classify(0) + section[dm.end():])
    return resync_moi_total(section)


def resync_moi_total(section: str) -> str:
    """
    For ONE variant's MOI-layer section whose base block was spliced by
    splice_base_and_total: Base = sum of the bullets under "Base ACMG
    criteria" (so a bullet a mechanical validator added afterwards, e.g.
    BS2, is counted), Total = Base + the layer's delta line, labelled via
    classify() with a "Potential " prefix when the delta is a hypothesis
    (recessive PM3 with unconfirmed phase). Replaces recompute_and_fix_totals
    for MOI layers: that one re-sums the Total from criteria bullets only and
    so drops a delta stated inline without a point tag (e.g. "PM3 (hypothesis
    — ...)"), and relabel_all_points_lines drops the "Potential" prefix.
    """
    dm = _MOI_DELTA_LINE_RE.search(section)
    hm = _BASE_HEADER_RE.search(section)
    bm = _BASE_POINTS_LINE_RE.search(section, hm.end()) if hm else None
    if not dm or not hm or not bm:
        return section
    base = sum_criteria(section[hm.end(): bm.start()])
    delta = float(dm.group(1))
    total = base + delta
    label = classify(total)
    if (dm.group(2) or "").lower() == "hypothesis" and label in ("Pathogenic", "Likely Pathogenic"):
        label = f"Potential {label}"

    section = (section[: bm.start()] + f"**Base ACMG points:** {_sum_str(base)} → {classify(base)}"
               + section[bm.end():])
    return _MODEL_TOTAL_RE.sub(
        lambda _m: f"**Total ACMG points:** {_sum_str(total)} → {label}\n", section, count=1
    )

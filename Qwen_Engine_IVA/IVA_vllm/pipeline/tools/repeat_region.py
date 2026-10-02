"""
pipeline/tools/repeat_region.py — PM4 grounding: is this in-frame indel in a
repeat region?

PM4 (protein length change) applies to in-frame deletions/insertions in a
non-repeat region and to stop-loss variants. This tool decides the "non-repeat"
part from the UCSC Genome Browser API (api.genome.ucsc.edu, no key):
  - simpleRepeat (Tandem Repeats Finder): any overlapping record = repeat.
  - rmsk (RepeatMasker): only repClass Simple_repeat / Low_complexity counts.
    SINE/LINE/LTR elements are ignored: an exon lying in an old transposon
    copy is not a repetitive coding tract.
UCSC returns only the items overlapping the requested range, so a non-empty
list is an overlap.

Coordinates come from AutoPVS1's normalized VCF-style variant line
("Variant : 4-121828696-ATGT-A  (Inframe_deletion)"), which is also what marks
the variant as in-frame. UCSC ranges are 0-based, end-exclusive:
  - deletion (padded REF/ALT): the deleted bases, [pos, pos + len(REF) - 1)
  - insertion: the two bases flanking the junction, [pos - 1, pos + 1)
Stop-loss variants need no repeat check.

Output is a "PM4 REPEAT CHECK" block read by pipeline/core/acmg_pm4.py. A failed
UCSC request raises ToolFetchError: PM4 is then not applied.
"""

import re

from pipeline.tools.base import NetworkTool
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError

UCSC_TRACK_URL = "https://api.genome.ucsc.edu/getData/track"
RMSK_REPEAT_CLASSES = {"Simple_repeat", "Low_complexity"}

_AUTOPVS1_VARIANT_RE = re.compile(
    r"^\s*Variant\s*:\s*(\w+)-(\d+)-([ACGTN]+)-([ACGTN]+)\s+\((Inframe_deletion|Inframe_insertion)\)",
    re.MULTILINE | re.IGNORECASE,
)
_STOP_LOSS_TYPES = ("stoploss", "stop_loss", "stop_lost", "stop loss", "stop lost")


def ucsc_span(pos: int, ref: str, alt: str) -> tuple[int, int]:
    """0-based, end-exclusive UCSC range for a padded VCF-style indel."""
    if len(ref) > len(alt):
        return pos, pos + len(ref) - 1
    return pos - 1, pos + 1


class RepeatRegionTool(NetworkTool):
    name        = "repeat_region"
    description = "Checks whether an in-frame indel lies in a UCSC repeat region (PM4 grounding)."

    def gate(self, variant: dict, context: ToolContext) -> bool:
        return True

    def run(self, variant: dict, context: ToolContext) -> str | None:
        type_val = context.field("Type").lower()
        if any(t in type_val for t in _STOP_LOSS_TYPES):
            return "PM4 REPEAT CHECK: MET — stop-loss variant (protein length change; no repeat check needed)."

        m = _AUTOPVS1_VARIANT_RE.search(context.all_outputs.get("autopvs1") or "")
        if not m:
            return None
        chrom, pos, ref, alt, kind = m.groups()
        chrom = chrom if chrom.lower().startswith("chr") else f"chr{chrom}"
        start, end = ucsc_span(int(pos), ref.upper(), alt.upper())
        genome = "hg19" if context.genome_build in ("19", "37", "hg19", "hg37", "GRCh37") else "hg38"
        where = f"{chrom}:{start + 1}-{end} ({genome}), {kind.lower()} {pos}-{ref}-{alt}"

        hits = []
        for track in ("simpleRepeat", "rmsk"):
            items = self._track_items(genome, track, chrom, start, end)
            for it in items:
                if track == "simpleRepeat":
                    hits.append(f"simpleRepeat period={it.get('period')} copies={it.get('copyNum')} "
                                f"unit={it.get('sequence')}")
                elif it.get("repClass") in RMSK_REPEAT_CLASSES:
                    hits.append(f"RepeatMasker {it.get('repClass')} {it.get('repName')}")

        if hits:
            return (f"PM4 REPEAT CHECK: NOT MET — in-frame indel in a repeat region, {where}: "
                    + "; ".join(hits) + " (UCSC).")
        return (f"PM4 REPEAT CHECK: MET — in-frame indel outside any repeat region, {where} "
                "(no UCSC simpleRepeat or Simple_repeat/Low_complexity RepeatMasker overlap).")

    def _track_items(self, genome: str, track: str, chrom: str, start: int, end: int) -> list[dict]:
        url = f"{UCSC_TRACK_URL}?genome={genome};track={track};chrom={chrom};start={start};end={end}"
        try:
            data = self._get(url, timeout=self.timeout).json()
        except ValueError as e:
            raise ToolParseError(f"UCSC {track} response not JSON ({url}): {e}") from e
        except Exception as e:
            raise ToolFetchError(f"UCSC {track} request failed ({url}): {e}") from e
        items = data.get(track)
        if not isinstance(items, list):
            raise ToolParseError(f"UCSC {track} response has no item list ({url}): {str(data)[:200]}")
        return items

"""
pipeline/tools/clinvar_gene_stats.py — ClinVar gene-level P/LP consequence counts
and per-variant submission classification tally.

Two independent pieces of evidence, both from ClinVar:
  1. Gene-level: counts of Pathogenic/Likely pathogenic ClinVar variants per gene,
     split by molecular consequence (missense vs. nonsense/frameshift), to ground
     PP2/BP1 in concrete gene-level evidence instead of gene-name plausibility.
     Gene-scoped and cached at class level, same pattern as GnomadConstraintTool /
     LitVar2SummaryTool._cgd_table.
  2. Variant-level: for THIS specific variant, how many individual ClinVar
     submitters classified it Pathogenic / Likely pathogenic / VUS / Likely
     benign / Benign — the submission-level classification tally. Previously
     this tally was only ever fetched inside WebSearchAgentTool's forced check,
     and only when the CSV's own ClinVar_class field already read Pathogenic/
     Likely pathogenic — so a stale or wrong CSV label meant the real ClinVar
     submitter breakdown was never seen at all. Fetched unconditionally here
     instead, for every variant with a resolvable ClinVar record.

     Every individual Pathogenic/Likely pathogenic submission is listed with
     its cited PMIDs and Comment. A submission whose Comment reports
     functional/experimental work (not in-silico, not negated) AND carries a
     literature reference (a PMID in that sentence, elsewhere in the Comment,
     or in the submission's own Citation list) is tagged "[functional
     evidence stated in submitter comment, citing PMID:X]" — the PS3 trigger
     (pipeline/core/acmg_ps3.py). The cited paper's abstract is NOT re-checked:
     the submitter's own statement plus its reference is the evidence.

  3. The per-variant CLINVAR status (clinvar_status()), printed as
     "CLINVAR STATUS: <value>" — one of core/clinvar_reference.py's
     CLINVAR_STATUSES. It is carried to
     every downstream block by pipeline/core/clinvar_reference.py and shown
     for every finding in the final report's sections 2-4.
"""

import logging
import re
import threading
import xml.etree.ElementTree as ET

from pipeline.tools.base import NetworkTool
from pipeline.tools.autopvs1 import parse_variant_coords
from pipeline.tools.websearch import _ncbi_get, _clean_xml_text, DEFAULT_TIMEOUT
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError
from pipeline.core.clinvar_reference import CLINVAR_UNDEFINED, clinvar_status

logger = logging.getLogger(__name__)

# Normalizes whatever GermlineClassification text ClinVar submitters used into
# one of the 5 standard tally buckets. Same mapping as NCBIFetchTool's in
# ncbi.py (kept as an independent copy — same duplication pattern as
# _FUNCTIONAL_MARKERS below).
_CLASS_BUCKETS = {
    "pathogenic": "Pathogenic",
    "likely pathogenic": "Likely pathogenic",
    "uncertain significance": "VUS",
    "likely benign": "Likely benign",
    "benign": "Benign",
}
_TALLY_ORDER = ["Pathogenic", "Likely pathogenic", "VUS", "Likely benign", "Benign"]

# Words that mark a submitter Comment sentence as reporting functional/
# experimental work (PS3-relevant; see comment_functional_pmids). Same
# list as NCBIFetchTool._FUNCTIONAL_MARKERS in ncbi.py (kept as an
# independent copy — this tool runs unconditionally per variant; that one
# only runs on-demand inside the ReAct agent).
_FUNCTIONAL_MARKERS = (
    "functional stud", "experimental stud", "in vitro", "in vivo", "assay", "minigene",
    "splicing assay", "reporter assay", "enzymatic activity",
    "protein function", "functional assay", "functional analysis",
    "functional characterization", "experimentally", "patient-derived",
    "patient derived", "fibroblast",
)

# A submitter Comment grounds PS3 when it reports functional/experimental
# work AND the submission carries a literature reference for it. The PMID is
# taken, in order of preference, from the functional sentence itself, from
# anywhere else in the Comment, or from the submission's own Citation list.
# In-silico/prediction wording and negated statements ("no experimental
# evidence...") never count as functional work.
_COMMENT_PMID_RE = re.compile(r"PMID[:\s]*((?:\d{6,9}(?:\s*[,;]\s*|\s+and\s+)?)+)", re.IGNORECASE)
_COMMENT_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_COMMENT_EXCLUDE_MARKERS = (
    "in silico", "in-silico", "predict", "modeling", "modelling", "algorithm",
    "computational", "not been", "not performed", "have not", "has not",
    "no functional", "no experimental", "no studies", "unknown", "unclear",
)


def _pmids_in(text: str) -> list[str]:
    out: list[str] = []
    for grp in _COMMENT_PMID_RE.findall(text):
        for p in re.findall(r"\d{6,9}", grp):
            if p not in out:
                out.append(p)
    return out


def comment_functional_pmids(comment: str, cited_pmids: list[str] | tuple = ()) -> list[str]:
    """PMIDs grounding a submitter Comment's report of functional work (see
    the comment above); [] when the Comment reports no functional work or no
    reference exists. Shared with ncbi.py's NCBIFetchTool."""
    functional = [
        sent for sent in _COMMENT_SENTENCE_SPLIT_RE.split(comment)
        if any(m in sent.lower() for m in _FUNCTIONAL_MARKERS)
        and not any(x in sent.lower() for x in _COMMENT_EXCLUDE_MARKERS)
    ]
    if not functional:
        return []
    out: list[str] = []
    for sent in functional:
        out += [p for p in _pmids_in(sent) if p not in out]
    if not out:
        out = _pmids_in(comment)
    if not out:
        out = [p for p in dict.fromkeys(cited_pmids)]
    return out




# A consequence class is "predominant" only when the OTHER class makes up
# less than this fraction of all P/LP variants in the gene — i.e. the gene is
# near-exclusively one mechanism. Deliberately strict: a looser ratio-based
# rule (e.g. 2x) flagged genes as "predominant" from mixed evidence, which
# over-applied BP1/PP2 on borderline genes.
MINOR_CLASS_FRACTION_THRESHOLD = 0.05


def classify_consequence_counts(missense: int, nonsense: int) -> str:
    """Verdict string from gene-level P/LP missense vs. nonsense/frameshift
    counts. Shared by run() and by the pipeline's gene-evidence summary table
    so both stay on the same threshold.

    Deliberately does NOT say "(supports BP1)" / "(supports PP2)" — this is a
    GENE-level fact (same for every variant in the gene regardless of that
    variant's own Type), and phrasing it as "supports BP1" primed the SLM to
    apply BP1 even to nonsense/frameshift candidate variants (BP1 requires the
    CANDIDATE to be missense; a nonsense candidate in a truncating-predominant
    gene is a mechanism MATCH — PVS1 territory — not BP1 territory). Criterion
    applicability, including the variant-Type gate, is decided entirely in
    prompts/conclusion.txt, not asserted here.
    """
    total = missense + nonsense
    if total == 0:
        return "insufficient data — no P/LP missense or nonsense/frameshift variants found in ClinVar"
    if nonsense / total < MINOR_CLASS_FRACTION_THRESHOLD:
        return "missense-predominant"
    if missense / total < MINOR_CLASS_FRACTION_THRESHOLD:
        return "nonsense/frameshift-predominant"
    return "balanced — neither consequence class clearly predominates"


class ClinVarGeneStatsTool(NetworkTool):
    """
    Fetches gene-level counts of ClinVar Pathogenic/Likely pathogenic variants,
    split into missense vs. nonsense/frameshift, via NCBI esearch (count-only).

    gate():  runs when Gene is present.
    run():   two esearch calls per gene, cached at class level.
    """

    name        = "clinvar_gene_stats"
    description = (
        "Fetches gene-level ClinVar P/LP variant counts split by missense vs. "
        "nonsense/frameshift consequence — grounds PP2/BP1 in concrete evidence."
    )

    timeout: int = DEFAULT_TIMEOUT

    # Class-level cache: {"GENE": {"missense": int, "nonsense": int} | None}
    _stats_cache: dict[str, dict | None] = {}
    _cache_lock = threading.Lock()

    def gate(self, variant: dict, context: ToolContext) -> bool:
        gene = context.field("Gene")
        return gene != "NA" and gene.strip() != ""

    @staticmethod
    def _pathogenic_base(gene: str) -> str:
        # ClinVar's Properties field indexes clinical significance as a
        # "clinsig <value>" compound phrase, not the bare value — e.g.
        # "clinsig pathogenic"[Properties], not pathogenic[Properties]. The
        # latter returns esearch's "phrasesnotfound" and silently matches
        # zero records for every gene (verified live: one of the most
        # heavily ClinVar-curated genes there is — returned 0/0 under the
        # bare-value query). Similarly "nonsense variant"[molecular
        # consequence] isn't an indexed phrase; the correct token is bare
        # "nonsense".
        return (
            f'{gene}[gene] AND ("clinsig pathogenic"[Properties] '
            f'OR "clinsig likely pathogenic"[Properties])'
        )

    def _esearch_count(self, term: str) -> int:
        try:
            data = _ncbi_get(
                "esearch.fcgi",
                {"db": "clinvar", "term": term, "retmode": "json", "retmax": 0},
                self.timeout,
            ).json()
            return int(data["esearchresult"]["count"])
        except Exception as e:
            raise ToolFetchError(f"ClinVar esearch failed for term={term!r}: {e}") from e

    def _fetch_stats(self, gene: str) -> dict:
        base = self._pathogenic_base(gene)
        missense_term = f'{base} AND "missense variant"[molecular consequence]'
        nonsense_term = (
            f'{base} AND (nonsense[molecular consequence] '
            f'OR "frameshift variant"[molecular consequence])'
        )
        try:
            missense = self._esearch_count(missense_term)
            nonsense = self._esearch_count(nonsense_term)
        except ToolFetchError:
            raise
        except Exception as e:
            raise ToolParseError(f"Failed to parse ClinVar counts for {gene}: {e}") from e
        return {"missense": missense, "nonsense": nonsense}

    def _get_stats(self, gene: str) -> dict:
        cache_key = gene.upper()
        if cache_key not in self._stats_cache:
            with self._cache_lock:
                if cache_key not in self._stats_cache:   # double-checked locking
                    self._stats_cache[cache_key] = self._fetch_stats(gene)
        return self._stats_cache[cache_key]

    # ── variant-level submission classification tally ────────────────────────

    # Extracts the cDNA-change token out of a combined/compound HGVS string,
    # e.g. "GENE_X:NM_000000:exon4:c.100A>C:p.K34T" -> "c.100A>C". ClinVar's own
    # esearch [variant name] index only matches this clean token, not a
    # colon-glued compound annotation (same fix already applied in ncbi.py's
    # resolve_clinvar_id for the same reason). "(" excluded too — an HGVS
    # string of the form "c.200T>A(p.Val67Asp)" otherwise swallows the
    # trailing protein annotation into the token, producing an unmatchable
    # esearch term and a false "not resolvable" even when ClinVar has the
    # variant (verified live: a real ClinVar-listed variant —
    # found instantly by ClinGenAlleleTool's genomic-HGVS route, but this
    # tool's own cDNA-token esearch silently failed on the untruncated token).
    _CDNA_CHANGE_RE = re.compile(r"c\.[^\s:;()]+")

    # Matches ClinGenAlleleTool's own "ClinVar variation ID: 123456 (RCV: ...)"
    # line — see the note on _resolve_variation_id below for why this is tried
    # first, before this tool's own independent (weaker) HGVS-only esearch.
    _CLINGEN_CLINVAR_ID_RE = re.compile(r"ClinVar variation ID:\s*(\d+)")

    # ClinVar's official star rating, keyed by the record's own aggregate
    # "ReviewStatus" text (ClassifiedRecord/Classifications/
    # GermlineClassification/ReviewStatus — the record-level consensus
    # review status shown as stars on ClinVar's own website, distinct from
    # each individual ClinicalAssertion's own per-submitter ReviewStatus,
    # which this tool does not use). Source: ClinVar's published
    # review-status-to-star-rating table. Lookup is case-insensitive; an
    # unrecognized status string (schema drift) maps to None rather than a
    # guessed star count.
    _REVIEW_STATUS_STARS: dict[str, int] = {
        "practice guideline": 4,
        "reviewed by expert panel": 3,
        "criteria provided, multiple submitters, no conflicts": 2,
        "criteria provided, conflicting classifications": 1,
        "criteria provided, conflicting interpretations": 1,  # older wording, same rating
        "criteria provided, single submitter": 1,
        "no assertion criteria provided": 0,
        "no classification provided": 0,
        "no classification for the single variant": 0,
        "no classification for the individual variant": 0,
    }

    @classmethod
    def _stars_for_review_status(cls, review_status: str | None) -> int | None:
        """Maps a ClinVar record's own aggregate ReviewStatus text to its
        star rating (0-4), or None if the text doesn't match a known status
        string."""
        if not review_status:
            return None
        return cls._REVIEW_STATUS_STARS.get(review_status.strip().lower())

    # Max distance (bp) between the variant's own Position and a ClinVar
    # record's start, on either assembly. An indel's start differs by a few
    # bases between ANNOVAR, VCF and ClinVar's own normalization.
    _LOCUS_WINDOW = 12

    # AutoPVS1's own normalized cDNA line ("cHGVS : NM_x.y:c.123del").
    _AUTOPVS1_CHGVS_RE = re.compile(r"^\s*cHGVS\s*:\s*(\S+)", re.MULTILINE)

    # A transcript and its own cDNA in one annotation: "NM_x.y:c.N" (AutoPVS1)
    # or "GENE:NM_x:exonK:c.N" (ANNOVAR).
    _TX_CDNA_RE = re.compile(r"\b([NX][MR]_\d+(?:\.\d+)?)(?::[^:|\s]*)*?:(c\.[^\s:;()|]+)")

    def _esearch_ids(self, term: str) -> list[str]:
        try:
            data = _ncbi_get(
                "esearch.fcgi",
                {"db": "clinvar", "term": term, "retmode": "json", "retmax": 20},
                self.timeout,
            ).json()
        except Exception as e:
            raise ToolFetchError(f"ClinVar esearch failed for term={term!r}: {e}") from e
        return data.get("esearchresult", {}).get("idlist", [])

    # Unversioned transcript accession -> current "NM_x.y" (None if unknown).
    _tx_version_cache: dict[str, str | None] = {}

    def _transcript_version(self, accession: str) -> str | None:
        if accession not in self._tx_version_cache:
            try:
                res = _ncbi_get(
                    "esummary.fcgi",
                    {"db": "nuccore", "id": accession, "retmode": "json"},
                    self.timeout,
                ).json().get("result", {})
            except Exception as e:
                raise ToolFetchError(f"nuccore version lookup failed for {accession}: {e}") from e
            versions = [res[u].get("accessionversion") for u in res.get("uids", [])]
            self._tx_version_cache[accession] = next(
                (v for v in versions if v and v.split(".")[0] == accession), None
            )
        return self._tx_version_cache[accession]

    def _ids_at_locus(self, ids: list[str], locus: tuple[str, int, str, str]) -> list[str]:
        """Keeps the IDs whose ClinVar record lies at the variant's own
        position (either assembly, within _LOCUS_WINDOW). For an SNV, a
        record whose canonical SPDI states different alleles is dropped too
        (several alleles share one rsID)."""
        if not ids:
            return []
        try:
            res = _ncbi_get(
                "esummary.fcgi",
                {"db": "clinvar", "id": ",".join(ids), "retmode": "json"},
                self.timeout,
            ).json().get("result", {})
        except Exception as e:
            raise ToolFetchError(f"ClinVar esummary failed for ids={ids}: {e}") from e
        chrom, pos, ref, alt = locus
        snv = len(ref) == 1 and len(alt) == 1
        kept = []
        for vid in ids:
            sets = res.get(vid, {}).get("variation_set", [])
            if not any(
                str(loc.get("chr", "")).upper() == chrom
                and str(loc.get("start", "")).isdigit()
                and abs(int(loc["start"]) - pos) <= self._LOCUS_WINDOW
                for vs in sets for loc in vs.get("variation_loc", [])
            ):
                continue
            if snv:
                spdis = [vs.get("canonical_spdi") or "" for vs in sets]
                parts = [sp.split(":") for sp in spdis if sp.count(":") == 3]
                if parts and not any(d.upper() == ref and i.upper() == alt for _, _, d, i in parts):
                    continue
            kept.append(vid)
        return kept

    def _resolve_variation_id(self, variant: dict, context: ToolContext) -> str | None:
        """
        Returns this variant's ClinVar variation ID, or None when no record
        could be matched to it. Routes, in order:
          1. rsID in its own field: rsN[VRID].
          2. Transcript-specific cDNA as one quoted variant name,
             "NM_x.y:c.N"[varnam]: AutoPVS1's own normalized cHGVS first,
             then, for an SNV only, the input HGVS's "NM_x:c.N" pair on
             AutoPVS1's transcript (every input pair when AutoPVS1 gave
             none; an unversioned transcript's current version is looked
             up in nuccore, since the [varnam] index needs one). An indel's
             input HGVS is skipped: ANNOVAR's indel cDNA is not valid HGVS
             (0 of 91 indels resolved by it on the 327-case run).
          3. GENE[gene] AND c.N[variant name], same cDNA tokens.
        Unquoted, ClinVar splits "NM_x:c.N" at the colon into two separate
        words and drops the field tag; without a version the quoted name
        matches nothing.
        Every hit must lie at the variant's own position (_ids_at_locus),
        and a route counts only when exactly one record is left. Without
        usable coordinates nothing can be checked and nothing is returned.

        Why: the input's ANNOVAR HGVS for an indel is often a truncated,
        non-HGVS string ("c.100_101G"), which the [variant name] index
        either misses or matches to an unrelated record of the same gene
        (verified on a 327-case run: every one of 18 statuses resolved by
        the old name-only search belonged to another variant). Checked
        against the record's position, rsID search was always right, and
        the AutoPVS1 cDNA recovered most indels the input HGVS missed.

        A failed NCBI call raises ToolFetchError (not None): a transient
        failure must not read as "no ClinVar record" — the manifest's
        retry config then applies, and a final failure prints as
        "Fetch failed", not as a confident negative.
        """
        try:
            chrom, pos, ref, alt = parse_variant_coords(
                variant_str=variant.get("Variant", ""),
                chrom_field=variant.get("Chromosome", ""),
                pos_field=variant.get("Position", ""),
                ref_field=variant.get("Ref_seq", ""),
                alt_field=variant.get("Var_seq", ""),
            )
            locus = (chrom.strip().upper().removeprefix("CHR"), int(pos),
                     (ref or "").upper(), (alt or "").upper())
        except (ValueError, TypeError):
            return None

        gene = context.field("Gene")
        terms: list[str] = []
        rs_id = context.field("RS_ID").strip()
        if re.fullmatch(r"rs\d+", rs_id):
            terms.append(f"{rs_id}[VRID]")
        ap_m = self._AUTOPVS1_CHGVS_RE.search(context.all_outputs.get("autopvs1") or "")
        pairs: list[tuple[str, str]] = self._TX_CDNA_RE.findall(ap_m.group(1)) if ap_m else []
        ap_txs = {tx.split(".")[0] for tx, _ in pairs}
        if len(locus[2]) == 1 and len(locus[3]) == 1:
            seen = {(tx.split(".")[0], cdna) for tx, cdna in pairs}
            for src in context.field("HGVS").split("|"):
                for tx, cdna in self._TX_CDNA_RE.findall(src):
                    key = (tx.split(".")[0], cdna)
                    if (not ap_txs or key[0] in ap_txs) and key not in seen:
                        seen.add(key)
                        pairs.append((tx, cdna))
        for tx, cdna in pairs:
            tx = tx if "." in tx else self._transcript_version(tx)
            if tx:
                terms.append(f'"{tx}:{cdna}"[varnam]')
        cdnas = list(dict.fromkeys(cdna for _, cdna in pairs))
        if not cdnas:
            m = self._CDNA_CHANGE_RE.search(context.field("HGVS"))
            cdnas = [m.group(0)] if m else []
        if gene != "NA":
            terms += [f"{gene}[gene] AND {c}[variant name]" for c in cdnas]

        for term in terms:
            kept = self._ids_at_locus(self._esearch_ids(term), locus)
            if len(kept) == 1:
                logger.info("ClinVar variation %s resolved for %s via %r", kept[0], gene, term)
                return kept[0]
        return None

    def _fetch_classification_tally(self, variation_id: str) -> dict | None:
        """
        Returns {"counts": {bucket: n, ...}, "pl_evidence": [line, ...],
        "has_functional_ref": bool, "has_case_ref": bool, "review_status": str | None,
        "stars": int | None, "aggregate_classification": str | None} or None.

        pl_evidence has one entry per individual Pathogenic/Likely-pathogenic
        submission — submitter, SCV accession, cited PMIDs (both
        Classification-level and AttributeSet-level Citation elements, since
        submitters use either or both), and the "[functional evidence stated
        in submitter comment, citing PMID:X]" tag when its Comment reports
        functional work with a reference (comment_functional_pmids);
        has_functional_ref is True when any submission carries that tag.
        Always populated regardless of whether
        the tally turns out conflicting — run() decides what to surface.
        A P/LP submission citing a PMID (Citation list or Comment) without that
        functional tag is tagged "[case reference: P/LP submission citing
        PMID:X]" instead, and has_case_ref is True — the PM3/PS4_Supporting
        trigger (pipeline/core/acmg_case_ref.py).

        review_status/stars/aggregate_classification come from the record's
        OWN top-level consensus (ClassifiedRecord/Classifications/
        GermlineClassification), not from re-deriving anything ourselves —
        this is ClinVar's own official star rating and overall call, exactly
        what a human reviewer sees on the ClinVar website for this variant.
        """
        try:
            xml = _ncbi_get(
                "efetch.fcgi",
                {"db": "clinvar", "id": variation_id, "rettype": "vcv",
                 "is_variationid": "true", "retmode": "xml"},
                self.timeout,
            ).text
            root = ET.fromstring(xml)
        except Exception as e:
            # Fetch/parse failure — NOT the same as a genuine empty record
            # (see _resolve_variation_id's docstring for why this must not
            # collapse to a silent None here either).
            raise ToolFetchError(
                f"ClinVar submission fetch failed for variation {variation_id}: {e}"
            ) from e

        va = root.find(".//VariationArchive")
        cr = va.find("ClassifiedRecord") if va is not None else None
        cal = cr.find("ClinicalAssertionList") if cr is not None else None
        assertions = cal.findall("ClinicalAssertion") if cal is not None else []
        if not assertions:
            return None

        agg_gc = cr.find("Classifications/GermlineClassification") if cr is not None else None
        review_status = None
        aggregate_classification = None
        if agg_gc is not None:
            rs_el = agg_gc.find("ReviewStatus")
            review_status = (rs_el.text or "").strip() if rs_el is not None and rs_el.text else None
            desc_el = agg_gc.find("Description")
            aggregate_classification = (desc_el.text or "").strip() if desc_el is not None and desc_el.text else None
        stars = self._stars_for_review_status(review_status)

        counts: dict[str, int] = {}
        pl_evidence: list[str] = []
        has_functional_ref = False
        has_case_ref = False
        for ca in assertions:
            cl = ca.find("Classification")
            if cl is None:
                continue
            desc_el = cl.find("GermlineClassification")
            desc = (desc_el.text or "").strip() if desc_el is not None and desc_el.text else "Not provided"
            bucket = _CLASS_BUCKETS.get(desc.lower(), desc)
            counts[bucket] = counts.get(bucket, 0) + 1

            if desc.lower() not in ("pathogenic", "likely pathogenic"):
                continue

            acc = ca.find("ClinVarAccession")
            submitter = acc.get("SubmitterName", "Unknown submitter") if acc is not None else "Unknown submitter"
            scv = acc.get("Accession", "") if acc is not None else ""

            comment_el = cl.find("Comment")
            comment = _clean_xml_text(comment_el.text) if comment_el is not None and comment_el.text else ""

            # Classification-level Citation only. A ClinicalAssertion's own
            # top-level AttributeSet/Citation is paired with an
            # Attribute Type="AssertionMethod" (e.g. "ACMG Guidelines, 2015")
            # — it cites the classification METHODOLOGY paper (Richards et al.
            # 2015, PMID 25741868), not variant-specific evidence. Verified
            # live: that PMID appeared on every submission in a real record
            # regardless of actual content, confirming it is method boilerplate,
            # not evidence — do not pull citations from that level.
            pmids = [c.find("ID").text for c in cl.findall("Citation")
                     if c.find("ID") is not None and c.find("ID").get("Source") == "PubMed"]

            line = f"- [{desc}] source: {submitter} ({scv})"
            if pmids:
                line += f" — cites PMID: {', '.join(pmids[:8])}"

            # PS3 trigger: the submitter's Comment reports functional work
            # and the submission carries a reference for it (see
            # comment_functional_pmids). No reference, no tag.
            tag = ""
            ref_pmids = comment_functional_pmids(comment, pmids) if comment else []
            case_pmids = list(dict.fromkeys(pmids + _pmids_in(comment)))
            if ref_pmids:
                has_functional_ref = True
                tag = (" [functional evidence stated in submitter comment, citing "
                       + ", ".join(f"PMID:{p}" for p in ref_pmids[:5]) + "]")
            elif case_pmids:
                has_case_ref = True
                tag = (" [case reference: P/LP submission citing "
                       + ", ".join(f"PMID:{p}" for p in case_pmids[:5]) + "]")

            if tag:
                line += tag
            if comment:
                line += f"\n  Rationale: {comment[:600]}"
            pl_evidence.append(line)

        return {
            "counts": counts,
            "pl_evidence": pl_evidence,
            "has_functional_ref": has_functional_ref,
            "has_case_ref": has_case_ref,
            "review_status": review_status,
            "stars": stars,
            "aggregate_classification": aggregate_classification,
        }

    def run(self, variant: dict, context: ToolContext) -> str | None:
        gene = context.field("Gene")
        stats = self._get_stats(gene)
        missense = stats["missense"]
        nonsense = stats["nonsense"]

        verdict = classify_consequence_counts(missense, nonsense)

        gene_block = (
            f"CLINVAR GENE-LEVEL P/LP VARIANT COUNTS ({gene}):\n"
            f"P/LP missense variants   : {missense}\n"
            f"P/LP nonsense/frameshift : {nonsense}\n"
            f"-> {verdict}"
        )

        # The variant-level tally is fetched best-effort: a fetch failure
        # here (already retried internally by _ncbi_get — see its docstring)
        # must not discard the gene-level counts computed above, but it also
        # must not be reported the same way as a genuine "ClinVar has no
        # record for this variant" — those are different signals to the LLM,
        # and conflating them was the original bug (see _resolve_variation_id
        # docstring). fetch_failed distinguishes the two in the output text.
        #
        # PREFER THE ALREADY-RESOLVED CLINGEN ID (mandatory check before
        # falling back to this tool's own HGVS-only esearch): ClinGenAlleleTool
        # runs at the same manifest order and resolves this variant via BOTH
        # HGVS and genomic-coordinate fallback strategies (see clingen_allele.py),
        # so it succeeds in cases this tool's own HGVS-only _resolve_variation_id
        # cannot — most commonly when the canonical HGVS field is "NA" (the
        # normalizer had no HGVS column to map for this upload) but ClinGen
        # still resolved the variant from chrom/pos/ref/alt. A real past
        # failure: a variant had HGVS="NA" in the variant
        # dict; ClinGenAlleleTool's own raw output (available in
        # context.all_outputs) already showed "ClinVar variation ID: 123456"
        # resolved from genomic coordinates, but this tool's run() ignored
        # that and went straight to its own HGVS-only resolution, which
        # returned None immediately (no HGVS to build a query from — it
        # never even called NCBI) and printed "Not resolvable — no ClinVar
        # record found," even though ClinVar's record was sitting one tool
        # output away. Always check for the ClinGen ID first.
        clingen_raw = context.all_outputs.get("clingen_allele") or ""
        clingen_match = self._CLINGEN_CLINVAR_ID_RE.search(clingen_raw)
        preresolved_id = clingen_match.group(1) if clingen_match else None

        hgvs = context.field("HGVS")
        fetch_failed = False
        try:
            variation_id = preresolved_id or self._resolve_variation_id(variant, context)
            result = self._fetch_classification_tally(variation_id) if variation_id else None
        except ToolFetchError as e:
            logger.warning("ClinVar variant-level lookup failed for %s %s: %s", gene, hgvs, e)
            variation_id, result, fetch_failed = None, None, True

        if fetch_failed:
            variant_block = (
                f"CLINVAR STATUS: {CLINVAR_UNDEFINED}\n"
                "CLINVAR VARIANT-LEVEL SUBMISSION TALLY:\n"
                "Fetch failed (NCBI request error after retries) — this variant's "
                "ClinVar status is UNKNOWN, not confirmed absent. Do not treat this "
                "as evidence the variant lacks a ClinVar record."
            )
        elif result is None:
            variant_block = (
                f"CLINVAR STATUS: {CLINVAR_UNDEFINED}\n"
                "CLINVAR VARIANT-LEVEL SUBMISSION TALLY:\n"
                "Not resolvable — no ClinVar record found for this specific variant "
                "(or no individual submissions listed)."
            )
        else:
            tally = result["counts"]
            total = sum(tally.values())
            known_lines = "\n".join(
                f"{label:<18}: {tally.get(label, 0)}" for label in _TALLY_ORDER
            )
            other = {k: v for k, v in tally.items() if k not in _TALLY_ORDER}
            other_line = (
                f"\nOther/unrecognized: {', '.join(f'{k}={v}' for k, v in other.items())}"
                if other else ""
            )
            # Star rating / aggregate call come from the record's own
            # top-level consensus (ClinVar's official star rating and
            # overall classification, not re-derived from the per-submitter
            # tally below) — printed only when the XML actually carried
            # them, per the "errors are informative, not silent" rule but
            # also the flip side of it: don't assert a rating we don't have.
            review_status = result.get("review_status")
            stars = result.get("stars")
            aggregate_classification = result.get("aggregate_classification")
            review_line = ""
            if review_status:
                star_text = f"{stars} star{'s' if stars != 1 else ''}" if stars is not None else "star rating unknown for this status string"
                review_line = f"ClinVar review status: {review_status} ({star_text})\n"
            aggregate_line = (
                f"ClinVar aggregate classification (official consensus call): {aggregate_classification}\n"
                if aggregate_classification else ""
            )
            status = clinvar_status(tally, result["has_functional_ref"], result["has_case_ref"])
            variant_block = (
                f"CLINVAR STATUS: {status}\n"
                f"CLINVAR VARIANT-LEVEL SUBMISSION TALLY (variation ID {variation_id}, "
                f"{total} individual submissions):\n"
                f"{review_line}{aggregate_line}"
                f"{known_lines}{other_line}"
            )

            # A real past bug this replaces: pl_evidence (including PS3's
            # functional-evidence tags) was previously ONLY ever printed
            # inside the "distinct_buckets > 1" branch below — meaning any
            # variant with a UNANIMOUS ClinVar record (every submitter
            # agrees, e.g. 2/2 "Likely pathogenic") silently lost its entire
            # per-submission P/LP evidence block, tags and all, before it
            # ever reached the prompt. PS3 was then structurally unable to
            # fire for the (common) unanimous case regardless of what
            # evidence existed, and the model was left to improvise —
            # verified live: a homozygous variant with a clean 2-submitter
            # unanimous "Likely pathogenic" ClinVar record got no P/LP
            # evidence block at all, and the conclusion stage worked around
            # the gap by citing that variant's own ClinVar record as if it
            # were an independent PS1 precedent (the "same nucleotide as
            # this variant is this variant's own record, not a separate
            # precedent" case prompts/conclusion.txt's PS1 section explicitly
            # warns against). The block below is now unconditional on
            # agreement/disagreement; "conflicting" only adds an extra note.
            if result["pl_evidence"]:
                variant_block += (
                    "\n\nPathogenic/Likely pathogenic submissions — evidence & source "
                    "(used for PS3 functional-evidence grounding):\n"
                    + "\n".join(result["pl_evidence"][:10])
                )

            # "Conflicting" per ClinVar's own definition: more than one distinct
            # classification bucket represented among individual submitters.
            distinct_buckets = sum(1 for n in tally.values() if n > 0)
            if distinct_buckets > 1 and result["pl_evidence"]:
                variant_block += (
                    f"\n\nCLINVAR CONFLICTING: submitters disagree on classification "
                    f"({distinct_buckets} distinct classifications among {total} "
                    f"submissions) — the per-submission evidence above already breaks "
                    f"out each P/LP call's own basis rather than trusting the aggregate "
                    f"label; weigh conflicting submissions accordingly (see PS3 guidance)."
                )
            elif distinct_buckets > 1 and not result["pl_evidence"]:
                variant_block += (
                    "\n\nCLINVAR CONFLICTING: submitters disagree on classification, "
                    "but no individual submission was itself Pathogenic/Likely "
                    "pathogenic (the P/LP portion of the aggregate, if any, traces to "
                    "expert panel review rather than a single traceable submission)."
                )

        return f"{gene_block}\n\n{variant_block}"

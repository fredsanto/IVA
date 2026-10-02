"""
pipeline/core/normalizer.py — Accept any CSV/Excel variant file and normalize it to the
standard TARGET_COLUMNS schema expected by the pipeline.

Unknown columns are silently ignored.
Missing target columns are filled with "NA".

Column-to-field mapping is SLM-driven (_map_columns_llm) — the model inspects the
header row (+ one sample data row) and classifies each column into a TARGET_COLUMNS
field, producing a human-readable summary of what it understood. The previous
deterministic alias-dict mapper is kept as _map_columns_old() for reference but is
no longer called.

Public function:
    normalize_upload(raw: bytes, filename: str, llm) -> tuple[list[dict], list[dict], list[dict] | None, str]
        Accepts raw file bytes + original filename (used to detect .xlsx vs .csv) +
        an LLMClient for header interpretation.
        Returns (normalized_variant_dicts, raw_field_dicts, parental_ab, header_mapping_summary).
        parental_ab is a list of dicts with keys like "proband", "mother", "father"
        (depending on how many Allelic balance columns exist), or None if none found.
        header_mapping_summary is human-readable text showing what the SLM understood.
"""

import io
import re

import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_COLUMNS = [
    "Variant", "Chromosome", "Position", "RS_ID", "Ref_seq", "Var_seq",
    "Type", "Transcript", "HGVS", "Zygosity", "Gene", "OMIM_phenotype",
    "OMIM_inheritance", "Inheritance", "ClinVar_class",
    "Allelic_balance", "Allelic_balance_mother", "Allelic_balance_father",
    "Frequency", "CADD_score",
    "REVEL_score", "SIFT_score", "PolyPhen2_score", "AlphaMissense_score",
    "SpliceAI_score",
]

COLUMN_ALIASES = {
    # Variant
    "variant": "Variant", "variant_id": "Variant", "snp": "Variant",
    "mutation": "Variant", "variant_name": "Variant",
    # Chromosome
    "chr": "Chromosome", "chrom": "Chromosome", "chromosome": "Chromosome",
    "contig": "Chromosome",
    # Position
    "position": "Position", "pos": "Position", "start": "Position",
    "genomic_position": "Position", "coord": "Position",
    # RS_ID
    "rs_id": "RS_ID", "rsid": "RS_ID", "dbsnp": "RS_ID",
    "rs": "RS_ID", "snp_id": "RS_ID", "rs id": "RS_ID",
    "avsnp150": "RS_ID",
    # Ref_seq
    "ref_seq": "Ref_seq", "ref": "Ref_seq", "reference": "Ref_seq",
    "ref_allele": "Ref_seq", "reference_allele": "Ref_seq",
    # Var_seq
    "var_seq": "Var_seq", "alt": "Var_seq", "alt_seq": "Var_seq",
    "alternate": "Var_seq", "alt_allele": "Var_seq",
    "alternate_allele": "Var_seq", "obs": "Var_seq",
    # Type — ExonicFunc takes priority; Func is a lower-priority fallback
    "type": "Type", "variant_type": "Type", "mutation_type": "Type",
    "class": "Type", "var_type": "Type",
    "exonicfunc": "Type",
    "func": "Type",
    # Transcript
    "transcript": "Transcript", "refseq": "Transcript", "refseq_id": "Transcript",
    "nm_id": "Transcript", "accession": "Transcript", "transcript_id": "Transcript",
    # HGVS
    "hgvs": "HGVS", "cdna": "HGVS", "hgvs_c": "HGVS",
    "hgvs_p": "HGVS", "c_dot": "HGVS", "p_dot": "HGVS",
    "nucleotide_change": "HGVS",
    # Zygosity
    "zygosity": "Zygosity", "genotype": "Zygosity", "gt": "Zygosity",
    "zyg": "Zygosity",
    # Gene
    "gene": "Gene", "gene_name": "Gene", "gene_symbol": "Gene",
    "hugo": "Gene", "symbol": "Gene", "genes": "Gene",
    # OMIM_phenotype
    "omim_phenotype": "OMIM_phenotype", "omim": "OMIM_phenotype",
    "phenotype": "OMIM_phenotype", "disease": "OMIM_phenotype",
    "condition": "OMIM_phenotype", "disorder": "OMIM_phenotype",
    # OMIM_inheritance
    "omim_inheritance": "OMIM_inheritance",
    # Inheritance
    "inheritance": "Inheritance", "inheritance_pattern": "Inheritance",
    "mode_of_inheritance": "Inheritance", "moi": "Inheritance",
    # ClinVar_class
    "clinvar_class": "ClinVar_class", "clinvar": "ClinVar_class",
    "clinical_significance": "ClinVar_class", "classification": "ClinVar_class",
    "pathogenicity": "ClinVar_class", "clinsig": "ClinVar_class",
    "interp": "ClinVar_class",
    "clnsig": "ClinVar_class",
    # Allelic_balance
    "allelic_balance": "Allelic_balance", "ab": "Allelic_balance",
    "vaf": "Allelic_balance", "allele_fraction": "Allelic_balance",
    "allele_balance": "Allelic_balance",
    # Frequency
    "frequency": "Frequency", "gnomad": "Frequency", "maf": "Frequency",
    "af": "Frequency", "allele_frequency": "Frequency",
    "population_frequency": "Frequency", "gnomad_af": "Frequency",
    "exac_af": "Frequency",
    "gnomad30_af_popmax": "Frequency",
    "gnomad211_exome_af": "Frequency",
    # CADD_score
    "cadd_score": "CADD_score", "cadd": "CADD_score", "phred": "CADD_score",
    "cadd_phred": "CADD_score",
    "cadd_v17_phred": "CADD_score",
    # REVEL_score
    "revel_score": "REVEL_score", "revel": "REVEL_score",
    # SIFT_score
    "sift_score": "SIFT_score", "sift": "SIFT_score",
    "sift4g_score": "SIFT_score", "sift_pred": "SIFT_score",
    # PolyPhen2_score
    "polyphen2_score": "PolyPhen2_score", "polyphen2": "PolyPhen2_score",
    "polyphen": "PolyPhen2_score",
    "polyphen2_hdiv_score": "PolyPhen2_score",
    "polyphen2_hvar_score": "PolyPhen2_score",
    # AlphaMissense_score
    "alphamissense_score": "AlphaMissense_score", "alphamissense": "AlphaMissense_score",
    "am_pathogenicity": "AlphaMissense_score", "am_score": "AlphaMissense_score",
    # SpliceAI_score
    "spliceai_score": "SpliceAI_score", "spliceai": "SpliceAI_score",
    "spliceai_max": "SpliceAI_score", "spliceai_max_score": "SpliceAI_score",
    "ds_max": "SpliceAI_score",
}

# ═══════════════════════════════════════════════════════════════════════════════
# TYPE NORMALISATION MAP  (ANNOVAR ExonicFunc/Func → pipeline vocabulary)
# ═══════════════════════════════════════════════════════════════════════════════

_TYPE_MAP = {
    "nonsynonymous snv":          "SNV",
    "synonymous snv":             "synonymous",
    "stopgain":                   "nonsense",
    "stoploss":                   "stoploss",
    "frameshift substitution":    "frameshift",
    "frameshift insertion":       "frameshift",
    "frameshift deletion":        "frameshift",
    "nonframeshift substitution": "indel",
    "nonframeshift insertion":    "indel",
    "nonframeshift deletion":     "indel",
    "splicing":                   "splicing",
    "exonic":                     "SNV",
    "intronic":                   "intronic",
    "intergenic":                 "intergenic",
    "upstream":                   "upstream",
    "downstream":                 "downstream",
    "utr3":                       "utr3",
    "utr5":                       "utr5",
    "ncrna_exonic":               "synonymous",
    "ncrna_intronic":             "intronic",
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _clean(v) -> str:
    """Normalize a cell value to a clean string or 'NA'."""
    s = str(v).strip()
    return "NA" if s in ("", "nan", "NaN", "None", "none", "NULL", "null", ".") else s


_AD_RE = re.compile(r"^\s*\d+(?:\s*[|,/]\s*\d+)+\s*$")
_AD_COLUMN_NAMES = {"ad", "allelic_depth", "allele_depth", "allelic_depths"}


def _ab_from_ad(raw: str) -> str:
    """Allelic balance from a read-count string "ref|alt" (or "ref,alt",
    "ref|alt1|alt2"): alt reads / all reads. "NA" if not that shape or no reads."""
    if not raw or not _AD_RE.match(raw):
        return raw if raw else "NA"
    counts = [int(x) for x in re.split(r"[|,/]", raw)]
    total = sum(counts)
    return f"{(total - counts[0]) / total:.4f}" if total else "NA"


def _collapse_repeated_gene(raw: str) -> str:
    parts = [p.strip() for p in re.split(r"[;,|]", raw) if p.strip()]
    if len(parts) > 1 and len({p.upper() for p in parts}) == 1:
        return parts[0]
    return raw


def _normalize_type(raw: str) -> str:
    """Map ANNOVAR ExonicFunc/Func values to pipeline Type vocabulary."""
    return _TYPE_MAP.get(raw.strip().lower(), raw)


def _parse_spliceai_value(raw: str) -> str:
    """
    Collapse a compound splice-prediction annotation string down to a single
    max score. Triggered by value shape, never by the source column's name —
    works regardless of which annotation tool/plugin produced it or what its
    header was called.

    Two known fixed schemas are handled by position (same approach as
    _parse_aachange() for ANNOVAR's AAChange column):
      - SpliceAI plugin, 10 pipe-delimited fields:
        ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL
        Only indices 2-5 are 0-1 delta scores; 6-9 are integer genomic
        offsets (can coincide with 0/1 and would otherwise look like a
        plausible delta) — so a blind "any float in 0-1" scan is unsafe here.
      - dbscSNV ada_score;rf_score pair — both already plain 0-1 scores.

    Any other shape falls back to treating every ';'/'|'/','-delimited token
    that parses as a 0-1 float as a candidate score. Leaves the value
    untouched if it already looks like a plain scalar.
    """
    if not raw or raw in ("NA", ".", "", "nan"):
        return raw

    pipe_tokens = raw.split("|")
    if len(pipe_tokens) == 10:
        deltas = []
        for t in pipe_tokens[2:6]:
            try:
                deltas.append(float(t.strip()))
            except ValueError:
                pass
        if deltas:
            return f"{max(deltas):.4f}"

    tokens = re.split(r"[|,;]", raw)
    if len(tokens) < 2:
        return raw  # already a scalar

    deltas = []
    for t in tokens:
        t = t.strip()
        try:
            f = float(t)
        except ValueError:
            continue
        if 0.0 <= f <= 1.0:
            deltas.append(f)

    if len(deltas) < 2:
        return raw  # not enough plausible scores — leave as-is

    return f"{max(deltas):.4f}"


# PolyPhen-2 categorical prediction severity, worst-first tiebreak when no
# numeric score is present in a compound annotation (ANNOVAR-style D/P/B calls).
_POLYPHEN_CATEGORICAL_SEVERITY = {
    "D": 3, "PROBABLY_DAMAGING": 3, "PROBABLYDAMAGING": 3,
    "P": 2, "POSSIBLY_DAMAGING": 2, "POSSIBLYDAMAGING": 2,
    "B": 1, "BENIGN": 1, "TOLERATED": 1,
}


def _parse_multivalue_score(raw: str, direction: str) -> str:
    """
    Collapse a compound multi-transcript score annotation (comma/semicolon/
    pipe-delimited, possibly with 'None'/'.' placeholders from transcripts
    the tool didn't score) down to the single most PATHOGENIC-leaning value
    — same treatment as _parse_spliceai_value() for splice scores,
    generalized to CADD/REVEL/SIFT/AlphaMissense. Different transcripts of
    the same variant can score differently; taking the most damaging value
    across them (rather than dropping the field as unparseable, or reading
    only the first token) is the conservative choice for a pathogenicity
    criterion — it never UNDER-calls deleteriousness that some transcript's
    annotation actually supports.

    direction: "high" if a HIGHER value is more pathogenic (CADD, REVEL,
               AlphaMissense); "low" if a LOWER value is more pathogenic (SIFT).
    """
    if not raw or raw in ("NA", ".", "", "nan"):
        return raw

    tokens = re.split(r"[|,;]", raw)
    if len(tokens) < 2:
        return raw  # already a scalar

    values = []
    for t in tokens:
        t = t.strip()
        try:
            values.append(float(t))
        except ValueError:
            continue  # 'None', blank, or non-numeric — skip, not a failure

    if not values:
        return raw  # nothing numeric found — leave as-is (caller may retry as categorical)

    chosen = max(values) if direction == "high" else min(values)
    return f"{chosen:.4g}"


def _parse_polyphen2_value(raw: str) -> str:
    """
    PolyPhen-2 needs its own wrapper: numeric compound strings reduce like
    any other high-is-worse score, but ANNOVAR-style categorical (D/P/B)
    compound strings ("D,D,P,B") have no numeric form to parse at all —
    reduce those to the single worst category present instead.
    """
    numeric = _parse_multivalue_score(raw, direction="high")
    if numeric != raw:
        return numeric  # successfully reduced numerically

    if not raw or raw in ("NA", ".", "", "nan"):
        return raw
    tokens = [t.strip().upper() for t in re.split(r"[|,;]", raw) if t.strip()]
    if len(tokens) < 2:
        return raw
    scored = [(t, _POLYPHEN_CATEGORICAL_SEVERITY.get(t)) for t in tokens]
    scored = [(t, s) for t, s in scored if s is not None]
    if not scored:
        return raw
    worst_token, _ = max(scored, key=lambda pair: pair[1])
    return worst_token


# Allelic-balance thresholds for deriving zygosity when no Zygosity value is
# available for a row. AB >= 0.85 reads as homozygous (the reference allele is
# essentially absent from reads); AB < 0.20 is too low to call het confidently
# (mosaic/subclonal/technical-noise territory) so it's flagged rather than
# stated plainly. These are conservative, clinically-conventional cutoffs, not
# an ACMG rule — the report always labels the value "derived", never "stated".
_ZYG_HOM_AB_THRESHOLD = 0.85
_ZYG_HET_AB_MIN = 0.20


def _derive_zygosity_from_ab(ab_raw: str) -> str:
    """Best-effort zygosity call from Allelic_balance (VAF) for a single row,
    used only when that row's own Zygosity value is "NA" — a stated Zygosity
    value (from a real Zygosity/Genotype/Zyg column in the upload) is always
    trusted as-is and never overridden by this. Returns "NA" if ab_raw isn't a
    parseable float, so the row is no worse off than before.
    """
    try:
        ab = float(ab_raw)
    except (TypeError, ValueError):
        return "NA"
    if ab >= _ZYG_HOM_AB_THRESHOLD:
        return f"Homozygous (derived from AB={ab:.2f}; input had no Zygosity column)"
    if ab >= _ZYG_HET_AB_MIN:
        return f"Heterozygous (derived from AB={ab:.2f}; input had no Zygosity column)"
    return f"Heterozygous (derived from AB={ab:.2f}, low — verify; input had no Zygosity column)"


def _parse_aachange(raw: str) -> str:
    """Extract a display HGVS string from ANNOVAR AAChange annotation."""
    if not raw or raw in ("NA", ".", "", "nan"):
        return "NA"
    first = raw.split("|")[0]
    parts = first.split(":")
    # expected: GENE, NM_xxx, exonN, c.xxx, p.xxx
    if len(parts) >= 5:
        return f"{parts[1]}:{parts[3]} {parts[4]}"
    elif len(parts) >= 4:
        return f"{parts[1]}:{parts[3]}"
    return "NA"


def _extract_raw_fields(df: pd.DataFrame) -> list[dict]:
    """Return full original rows as dicts, one per variant."""
    return [
        {col: _clean(row[col]) for col in df.columns}
        for _, row in df.iterrows()
    ]


def _map_columns_old(df: pd.DataFrame) -> pd.DataFrame:
    """
    DEPRECATED — superseded by _map_columns_llm(). Kept for reference/fallback
    only; not called from normalize_upload() anymore.

    Rename columns to TARGET_COLUMNS names using COLUMN_ALIASES.
    Columns that don't match any alias are kept as-is (they'll be ignored later).
    When two columns alias to the same target, ExonicFunc takes priority over Func
    for the Type field; otherwise first-encountered wins.

    Special handling: columns matching 'Allelic balance - *' (case-insensitive)
    are mapped to 'Allelic_balance' (first match only). These are sample-specific
    allelic balance columns.
    """
    df.columns = [str(c) for c in df.columns]
    claimed: set[str] = set()
    mapping: dict[str, str] = {}

    # Sort so ExonicFunc is processed before Func, enforcing Type field priority
    cols = sorted(
        df.columns,
        key=lambda c: (0 if c.strip().lower() == "exonicfunc" else 1),
    )

    # Regex pattern for sample-specific allelic balance columns
    _ab_pattern = re.compile(r"^allelic\s+balance\s*-", re.IGNORECASE)
    ab_found = False

    for col in cols:
        col_lower = col.strip().lower()

        # Special handling: "Allelic balance - SAMPLE_ID" → Allelic_balance (first only)
        if _ab_pattern.match(col):
            with open("/tmp/ab_debug.txt", "a") as _dbg:
                _dbg.write(f"_map_columns: regex MATCH [{col}] → Allelic_balance={not ab_found}\n")
            if not ab_found and "Allelic_balance" not in claimed:
                mapping[col] = "Allelic_balance"
                claimed.add("Allelic_balance")
                ab_found = True
            # Additional AB columns are skipped here — handled by extract_parental_ab()
            continue

        canonical = COLUMN_ALIASES.get(col_lower)
        if canonical and canonical not in claimed:
            mapping[col] = canonical
            claimed.add(canonical)

    if mapping:
        df = df.rename(columns=mapping)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SLM-DRIVEN HEADER INTERPRETATION
# ═══════════════════════════════════════════════════════════════════════════════

_FIELD_DESCRIPTIONS = {
    "Variant":             "free-text variant identifier/name, e.g. \"chr1:1000000 C>T\"",
    "Chromosome":          "chromosome number/name, e.g. \"16\", \"chrX\"",
    "Position":            "genomic position (integer coordinate)",
    "RS_ID":               "dbSNP rsID, e.g. \"rs000000\"",
    "Ref_seq":             "reference allele",
    "Var_seq":             "alternate/variant allele",
    "Type":                "variant type/consequence, e.g. missense, nonsense, frameshift, synonymous, splicing",
    "Transcript":          "RefSeq transcript/accession ID, e.g. \"NM_000000.1\", ONLY when given as its own column separate from the HGVS notation",
    "HGVS":                "cDNA and/or protein change notation, e.g. \"c.100A>C p.(Lys34Thr)\" (may include a transcript prefix). "
                           "Match this by the VALUE'S SHAPE, not the column name — a column with a generic/unrelated-"
                           "sounding name (e.g. \"variant_info\", \"annotation\", \"details\") still counts as HGVS if its "
                           "sample value contains a \"c.\" and/or \"p.\" change token, even bundled inside a longer "
                           "compound annotation across multiple transcripts, e.g. "
                           "\"GENE:MANE_Select-NM_000000.1:exon13:c.100A>C:p.Lys34Thr,GENE:NM_000001.1:exon13:"
                           "c.100A>C:p.Lys34Thr\" is HGVS (map the whole cell — downstream parsers pull the c./p. "
                           "tokens back out of it). A real past failure: a column literally named \"variant_info\" "
                           "with exactly this shape was left unmapped in every run because its column name gave no "
                           "hint and the description above only showed a already-clean example — AutoPVS1 and other "
                           "tools then skipped that variant entirely for \"hgvs=NA\" even though the c./p. change was "
                           "sitting right there in an unmapped column.",
    "Zygosity":            "heterozygous/homozygous/hemizygous genotype call",
    "Gene":                "gene symbol",
    "OMIM_phenotype":      "disease/phenotype name associated with the gene/variant",
    "OMIM_inheritance":    "OMIM-reported inheritance mode text",
    "Inheritance":         "inheritance mode, e.g. AR/AD/XLR/XLD",
    "ClinVar_class":       "ClinVar clinical significance/classification",
    "Allelic_balance":     "the PROBAND's own allelic balance / variant allele fraction (a single float, typically 0-1). "
                           "When a file has multiple allelic-balance-style columns (trio data), this is whichever one "
                           "represents the PROBAND/index case specifically — by explicit naming (\"proband\", \"index\", "
                           "\"affected\") if present, else the FIRST allelic-balance column in the file's own column order.",
    "Allelic_balance_mother": "the MOTHER's own allelic balance / variant allele fraction for this same variant, "
                           "when the file provides trio (proband+parents) data as separate columns. Column naming "
                           "varies widely and is NOT limited to any fixed pattern — match ANY column that is clearly "
                           "a second/third allelic-balance-style column belonging to a parent, regardless of exact "
                           "wording: explicit naming (\"mother\", \"mom\", \"maternal\", \"AB_mom\"), a family/sample-ID "
                           "suffix that differs from the proband's own ID, or a purely positional/numbered suffix with "
                           "no explicit parent wording at all (e.g. \"Allelic_balance_1\", \"AF_sample2\", \"VAF_2\"). "
                           "When there is no explicit naming to tell mother from father apart (e.g. bare numbered "
                           "suffixes), use the file's own column order as the tiebreaker: proband first, then mother, "
                           "then father — i.e. the first non-proband allelic-balance column is the mother's. Only "
                           "applies when there are 2+ non-proband allelic-balance columns; leave null for a single-"
                           "sample file.",
    "Allelic_balance_father": "the FATHER's own allelic balance / variant allele fraction — same matching rules as "
                           "Allelic_balance_mother above (explicit naming, sample-ID suffix, or positional fallback), "
                           "but for the father: when naming gives no explicit parent identity, this is the LAST "
                           "allelic-balance column in file order (after proband and mother). Only applies when there "
                           "are 3 total allelic-balance columns (proband + both parents); leave null otherwise.",
    "Frequency":           "population allele frequency, e.g. gnomAD/ExAC/1000G",
    "CADD_score":          "CADD Phred-scaled deleteriousness score. Column naming varies widely across "
                           "uploads — match ANY column whose name is clearly a CADD variant regardless of "
                           "separator/case/wording: \"CADD score\", \"CADD-Score\", \"CADD_Score\", \"cadd\", "
                           "\"CADD_phred\", \"CADD PHRED\", \"CADD_v17_PHRED\". Do NOT match \"CADD_raw\" or "
                           "\"CADD_raw_rankscore\" (or similarly-named \"raw\"/\"rankscore\" variants) — those "
                           "are different internal sub-metrics, not the interpretable Phred-scaled score this "
                           "field means. If both a plain/phred CADD column and a raw/rankscore CADD column are "
                           "present, map only the plain/phred one here.",
    "REVEL_score":         "REVEL pathogenicity score. Match any naming variant regardless of separator/case: "
                           "\"REVEL score\", \"REVEL-Score\", \"revel\", \"REVEL_score\".",
    "SIFT_score":          "SIFT score or prediction (damaging/tolerated). Match any naming variant regardless "
                           "of separator/case: \"Sift score\", \"SIFT-Score\", \"sift\", \"SIFT_pred\", "
                           "\"SIFT_converted_rankscore\". If both a SIFT score and a separate SIFT prediction/"
                           "pred column exist, prefer the score column.",
    "PolyPhen2_score":     "PolyPhen-2 score or prediction. Match any naming variant regardless of separator/"
                           "case: \"PolyPhen2 score\", \"Polyphen-2\", \"polyphen2_hdiv_score\", "
                           "\"Polyphen2_HVAR_pred\".",
    "AlphaMissense_score": "AlphaMissense pathogenicity score. Match any naming variant regardless of "
                           "separator/case: \"AlphaMissense score\", \"alphamissense\", \"am_pathogenicity\".",
    "SpliceAI_score":      "precomputed splicing-impact prediction for this variant, from SpliceAI, dbscSNV "
                           "(ada_score/rf_score ensemble prediction), or any similar splice-effect tool. May "
                           "appear as a single score (e.g. \"0.87\"), a semicolon-pair (e.g. dbscSNV's "
                           "\"0.999;0.685\"), OR a compound annotation string bundling several pipe/comma-"
                           "delimited values together (allele, gene symbol, the four acceptor/donor gain/loss "
                           "delta scores, plus positions, e.g. \"T|GENE|0.01|0.00|0.85|0.02|-2|33|1|-38\") — "
                           "any of these compound forms is still a match for this field, whatever the column "
                           "is named.",
}

_HEADER_INTERPRETATION_SYSTEM = (
    "You are inspecting the header row (and one sample data row) of an uploaded "
    "clinical variant table. For each canonical field below, decide which ONE "
    "original column (if any) corresponds to it. Use the sample row's values to "
    "disambiguate when the header name alone is ambiguous (e.g. a generically-named "
    "\"Score\" column can often be identified by its value range). Do not force a "
    "match — use null for a field if none of the columns genuinely fit it. Each "
    "original column may be used for AT MOST one field (pick the best match if "
    "several fields look similar).\n\n"
    "IMPORTANT — column names vary a great deal across uploads and NEVER require an "
    "exact string match to a canonical field name. The same field routinely shows up "
    "with different separators (space/underscore/hyphen), casing, or abbreviation — "
    "e.g. \"CADD score\", \"CADD-Score\", \"CADD_Score\", and \"cadd\" all mean the same "
    "thing as canonical field CADD_score. Match on MEANING, not spelling: recognize the "
    "underlying tool/metric a column is named after even when its exact wording differs "
    "from the canonical field name below. The per-field notes call out specific naming "
    "variants and any sub-metric qualifiers (e.g. \"_raw\", \"_rankscore\") that should "
    "NOT be matched even though they share the same tool name — read those carefully, "
    "since a raw/internal sub-metric is a different value from the interpretable score "
    "the canonical field means.\n\n"
    "Canonical fields:\n"
    + "\n".join(f"  - {name}: {desc}" for name, desc in _FIELD_DESCRIPTIONS.items())
    + "\n\nOutput ONLY a single JSON object whose KEYS are EXACTLY the canonical "
    "field names listed above (every one of them, spelled exactly as given) and "
    "whose VALUES are each either the matching ORIGINAL column name (copied "
    "exactly as given) or null. No explanation, no markdown, no extra text — "
    "JSON only."
)


def _map_columns_llm(df: pd.DataFrame, llm) -> tuple[pd.DataFrame, str]:
    """
    SLM-driven replacement for _map_columns_old(): asks the model to classify
    each header into a TARGET_COLUMNS field, using one sample data row for
    disambiguation. This INCLUDES allelic-balance columns (proband, mother,
    father) — previously a hardcoded regex (`^allelic\\s+balance\\s*-`) claimed
    these deterministically before the model ever saw them, and, being
    space-strict, silently failed on any other naming convention (verified:
    a real upload used "Allelic_balance_1"/"Allelic_balance_2" and every one
    of those columns came back "(unmapped)", silently discarding trio
    parental data for every variant in that file). The model now classifies
    ALL header columns itself, including these — see the
    Allelic_balance/Allelic_balance_mother/Allelic_balance_father entries in
    _FIELD_DESCRIPTIONS for the matching rules (explicit naming, sample-ID
    suffix, or positional fallback when naming gives no parent identity).

    Returns (renamed_df, human_readable_summary) — the summary is meant to be
    shown to the user (SSE + final report) so they can see what the model
    understood before the rest of the pipeline runs on it.
    """
    import json as _json

    df.columns = [str(c) for c in df.columns]
    claimed: set[str] = set()
    mapping: dict[str, str] = {}

    # ── Deterministic pass: raw headers that already literally match a
    # canonical field name (e.g. a "Frequency" or "Chromosome" column already
    # named exactly that). Auto-claimed and excluded from the SLM pass —
    # otherwise the SLM can independently map a *different* raw column onto
    # the same canonical name (e.g. mapping "Frequency" <- "Alternate Allele
    # Coverage" while the literal "Frequency" column sits untouched), leaving
    # two columns both named "Frequency" after rename. pandas then raises
    # "cannot reindex on an axis with duplicate labels" the first time
    # anything downstream selects that column by name.
    #
    # Deliberately EXACT-only, not fuzzy/normalized — column naming in the
    # wild varies too much (CADD score / CADD-Score / CADD_Score / cadd /
    # CADD_Phred / ...) for a hardcoded separator-normalization rule to
    # generalize; that recognition job belongs to the SLM pass below, which
    # is given explicit naming-variant examples for exactly this reason. Note
    # this exact-match fast path is intentionally NOT applied to
    # "Allelic_balance" itself — a file can have several allelic-balance-style
    # columns (trio), and which one is the proband/mother/father needs the
    # same model judgment as any other trio-naming variant, not a first-match
    # shortcut.
    _target_by_lower = {
        f.lower(): f for f in TARGET_COLUMNS
        if f not in ("Allelic_balance", "Allelic_balance_mother", "Allelic_balance_father")
    }

    llm_cols: list[str] = []
    exact_matches: list[tuple[str, str]] = []
    for col in df.columns:
        exact_field = _target_by_lower.get(col.strip().lower())
        if exact_field and exact_field not in claimed:
            if col != exact_field:
                mapping[col] = exact_field
            claimed.add(exact_field)
            exact_matches.append((col, exact_field))
            continue
        llm_cols.append(col)

    # ── SLM pass: everything else ─────────────────────────────────────────────
    summary_lines = ["Column header interpretation (SLM-driven):"]

    # Exact-header-name matches never reach the SLM pass below (claimed here,
    # deterministically) — log them too, or a column named e.g. "Frequency"
    # verbatim is silently correct but looks IDENTICAL in this log to a
    # genuinely-missing column, since neither appears in the SLM-mapped list
    # that follows. Distinguish it explicitly instead of leaving it invisible.
    for col, exact_field in exact_matches:
        summary_lines.append(f"  {col!r:<30} -> {exact_field} (exact header match)")

    if llm_cols:
        sample_row = {}
        if len(df) > 0:
            first = df.iloc[0]
            sample_row = {c: _clean(first[c]) for c in llm_cols}

        user_prompt = (
            "Columns to classify:\n" + "\n".join(f"  - {c!r}" for c in llm_cols)
            + "\n\nSample row values (for disambiguation):\n"
            + "\n".join(f"  {c!r}: {sample_row.get(c, 'NA')!r}" for c in llm_cols)
        )

        raw = llm.generate(
            system=_HEADER_INTERPRETATION_SYSTEM, user=user_prompt, max_tokens=800
        ).strip()

        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed: dict = _json.loads(match.group()) if match else {}
        except (_json.JSONDecodeError, AttributeError) as e:
            print(f"[csv_normalizer] Header interpretation JSON parse failed ({e}) — no columns mapped via SLM")
            parsed = {}

        # Resolve each returned column value against llm_cols. Exact match first;
        # falls back to a whitespace/case-normalized match since the model isn't
        # always byte-exact when echoing back column names (seen in practice with
        # real uploaded headers containing BOM/whitespace artifacts) — this keeps
        # the lookup robust without requiring literal reproduction.
        def _norm(s) -> str:
            return re.sub(r"\s+", " ", str(s).strip()).lower()

        norm_to_col = {_norm(c): c for c in llm_cols}
        assigned_cols: set[str] = set()

        # Iterate fields in a stable order (skip Allelic_balance if the deterministic
        # pass already claimed it) so the summary reads in a sensible sequence.
        for field in TARGET_COLUMNS:
            if field in claimed:
                continue
            value = parsed.get(field)
            if not value or not isinstance(value, str):
                continue
            resolved = value if value in norm_to_col.values() else norm_to_col.get(_norm(value))
            if resolved and resolved not in assigned_cols:
                mapping[resolved] = field
                claimed.add(field)
                assigned_cols.add(resolved)

        for col in llm_cols:
            target = mapping.get(col)
            if target:
                note = f"  {col!r:<30} -> {target}"
            elif "allelic balance" in col.lower():
                # Not a red flag: this column isn't unused, it's just not the
                # canonical single-value Allelic_balance field. Parental AB
                # columns (proband/mother/father, however many/whatever named)
                # are extracted separately by extract_parental_ab() and appear
                # in SEGREGATION ANALYSIS, not here — a bare "(unmapped)" for
                # a second/third allelic-balance column reads as dropped data
                # when it's actually just handled by a different code path.
                note = f"  {col!r:<30} -> (handled separately — parental allelic balance, see SEGREGATION ANALYSIS)"
            else:
                note = f"  {col!r:<30} -> (unmapped)"
            summary_lines.append(note)

    summary = "\n".join(summary_lines)
    print(f"[csv_normalizer] {summary}")

    if mapping:
        df = df.rename(columns=mapping)

    # Defense-in-depth: if a canonical target name still ended up duplicated
    # (unforeseen collision beyond the exact-literal-match case handled
    # above), keep the first occurrence and drop the rest rather than
    # letting a bare pandas "cannot reindex on an axis with duplicate
    # labels" surface downstream the first time that column is selected.
    dupe_targets = {f for f in TARGET_COLUMNS if (df.columns == f).sum() > 1}
    if dupe_targets:
        print(f"[csv_normalizer] WARNING: duplicate columns after mapping, keeping first occurrence: {sorted(dupe_targets)}")
        keep = ~df.columns.duplicated()
        df = df.loc[:, keep]

    return df, summary


def _build_normalized_df(df: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
    """Project df onto TARGET_COLUMNS, filling missing ones with 'NA'.

    Post-processing:
      - If HGVS is all NA but AAChange exists in the original input, fill from AAChange.
      - Normalise Type values through the ANNOVAR vocabulary map.
    """
    # index=df.index (not a bare pd.DataFrame()) so a scalar "NA" assigned to
    # an unmapped column (e.g. "Variant", first in TARGET_COLUMNS and usually
    # unmapped) broadcasts across the real row index immediately. Without it,
    # the scalar assignment on a still-empty frame produces no rows, and once
    # a later column assigns a real Series and establishes the index, pandas
    # backfills that earlier column with NaN instead of "NA" — a real observed
    # failure where a bare float NaN (not the string "NA") reached a downstream
    # `.strip()` call and crashed with AttributeError.
    out = pd.DataFrame(index=df.index)
    for col in TARGET_COLUMNS:
        if col in df.columns:
            out[col] = df[col].apply(_clean)
        else:
            out[col] = "NA"

    # ClinVar_class is never taken from the upload (annotation snapshots go
    # stale); ClinVarGeneStatsTool fills it from the live ClinVar record.
    out["ClinVar_class"] = "NA"

    # Fill HGVS from AAChange if needed
    if (out["HGVS"] == "NA").all():
        aa_col = next(
            (c for c in df_original.columns if c.strip().lower() == "aachange"),
            None,
        )
        if aa_col is not None:
            out["HGVS"] = df_original[aa_col].apply(
                lambda v: _parse_aachange(_clean(v))
            ).values

    # Collapse a repeated gene symbol ("GENE;GENE", one entry per transcript
    # in multi-transcript annotation exports) to the single symbol. Left as-is,
    # every gene-level lookup queried "GENE;GENE[gene]", matched nothing, and
    # silently reported the gene as empty (0 ClinVar P/LP variants, PM1 window
    # empty). Only identical repeats are collapsed; distinct symbols are kept.
    out["Gene"] = out["Gene"].apply(_collapse_repeated_gene)

    # Normalise Type vocabulary
    out["Type"] = out["Type"].apply(
        lambda v: _normalize_type(v) if v != "NA" else v
    )

    # Allelic balance from raw read counts. Annotation exports (e.g. ANNOVAR)
    # often give only an "AD" column ("ref|alt" read counts) and no allelic-
    # balance column: left unmapped, every variant reached the model as
    # Allelic_balance=NA, which it then read as "zygosity unconfirmed" even
    # for clear homozygous calls. Also converts an AB column that was mapped
    # but holds raw "ref|alt" counts instead of a fraction.
    out["Allelic_balance"] = out["Allelic_balance"].apply(_ab_from_ad)
    if (out["Allelic_balance"] == "NA").all():
        ad_col = next(
            (c for c in df_original.columns if c.strip().lower() in _AD_COLUMN_NAMES), None,
        )
        if ad_col is not None:
            out["Allelic_balance"] = df_original[ad_col].apply(
                lambda v: _ab_from_ad(_clean(v))
            ).values

    # Derive Zygosity from Allelic_balance for any row that has no stated
    # Zygosity value — whether because no source column mapped to Zygosity at
    # all, or that specific row's cell was blank/NA while others had a real
    # value. A real stated value is always left untouched.
    if "Allelic_balance" in out.columns:
        out["Zygosity"] = [
            _derive_zygosity_from_ab(ab) if zyg == "NA" else zyg
            for zyg, ab in zip(out["Zygosity"], out["Allelic_balance"])
        ]

    # Collapse compound SpliceAI annotation strings (whatever the source
    # column was named) down to a single max delta score
    out["SpliceAI_score"] = out["SpliceAI_score"].apply(_parse_spliceai_value)

    # Same treatment for the other in-silico predictors: a multi-transcript
    # compound annotation (e.g. "0.007,0.001,0.003,0.005,0.004,0.005" or
    # "None,None,0.766,0.766,0.766,None") collapses to the single most
    # pathogenic-leaning value across transcripts, rather than being left
    # unparseable (which silently drops real evidence from PP3 downstream).
    out["CADD_score"]          = out["CADD_score"].apply(lambda v: _parse_multivalue_score(v, "high"))
    out["REVEL_score"]         = out["REVEL_score"].apply(lambda v: _parse_multivalue_score(v, "high"))
    out["AlphaMissense_score"] = out["AlphaMissense_score"].apply(lambda v: _parse_multivalue_score(v, "high"))
    out["SIFT_score"]          = out["SIFT_score"].apply(lambda v: _parse_multivalue_score(v, "low"))
    out["PolyPhen2_score"]     = out["PolyPhen2_score"].apply(_parse_polyphen2_value)

    return out


def _df_to_variant_dicts(df: pd.DataFrame) -> list[dict]:
    """Convert each row of a normalized DataFrame to a plain dict."""
    return [
        {col: row[col] for col in TARGET_COLUMNS}
        for _, row in df.iterrows()
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# PARENTAL ALLELIC BALANCE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

# Labels when 3 columns are present: proband, mother, father (fixed order)
_PARENTAL_LABELS_3 = ["proband", "mother", "father"]

_MOTHER_NAME_RE = re.compile(r"\b(mother|mom|maternal)\b", re.IGNORECASE)
_FATHER_NAME_RE = re.compile(r"\b(father|dad|paternal)\b", re.IGNORECASE)


def _parent_label_from_name(col: str) -> str | None:
    """If a column name explicitly names the parent (e.g. "Allelic balance -
    mother", "Allelic balance_mother", "Allelic balance (Father)"), return
    "mother" or "father" — else None to fall back to positional convention.
    Underscores/hyphens are normalised to spaces first — "_" is a \\w
    character, so a \\b boundary never forms at "balance_mother"'s "e_m"
    junction and a bare \\bmother\\b search silently fails to match there."""
    normalized = re.sub(r"[_\-]+", " ", col)
    if _MOTHER_NAME_RE.search(normalized):
        return "mother"
    if _FATHER_NAME_RE.search(normalized):
        return "father"
    return None


def _build_parental_ab_from_mapped(df: pd.DataFrame) -> list[dict] | None:
    """
    Build parental_ab directly from the LLM's own header classification —
    Allelic_balance / Allelic_balance_mother / Allelic_balance_father are now
    canonical TARGET_COLUMNS fields the model maps like any other (see their
    entries in _FIELD_DESCRIPTIONS for the matching rules: explicit naming,
    sample-ID suffix, or positional fallback). This is the PRIMARY path —
    unlike a fixed regex, the model isn't tied to one naming convention.

    Returns None if the proband's own Allelic_balance column wasn't found at
    all (nothing to build from here — caller falls back to the regex-based
    extract_parental_ab() as a safety net).
    """
    if "Allelic_balance" not in df.columns or (df["Allelic_balance"] == "NA").all():
        return None

    has_mother = "Allelic_balance_mother" in df.columns and not (df["Allelic_balance_mother"] == "NA").all()
    has_father = "Allelic_balance_father" in df.columns and not (df["Allelic_balance_father"] == "NA").all()

    result = []
    for _, row in df.iterrows():
        entry = {"proband": row["Allelic_balance"]}
        if has_mother:
            entry["mother"] = row["Allelic_balance_mother"]
        if has_father:
            entry["father"] = row["Allelic_balance_father"]
        result.append(entry)

    print(f"[csv_normalizer] Parental AB from SLM header mapping: "
          f"proband{'+mother' if has_mother else ''}{'+father' if has_father else ''}")
    return result


def extract_parental_ab(df: pd.DataFrame) -> list[dict] | None:
    """
    FALLBACK path only — used when _build_parental_ab_from_mapped() finds no
    proband Allelic_balance column in the model's own mapping. Detects
    columns containing "Allelic balance" in the header directly via regex
    and extracts per-variant allelic balance values.

    Column naming convention: "Allelic balance - SAMPLE_ID"
    Order is always: proband first, then parents (if present).

    Returns:
        list of dicts (one per row) with keys depending on column count:
          - 1 column:  {"proband": "0.48"}
          - 2 columns: {"proband": "0.48", "extra1": "0.51"}
          - 3 columns: {"proband": "0.48", "mother": "0.51", "father": "0.00"}
        Or None if no matching columns found.
    """
    # Find all columns with "allelic balance" (case-insensitive, separator-
    # insensitive substring — a real input file used "Allelic_balance_1"/
    # "Allelic_balance_proband" (underscore, not space) and the literal-space
    # check below silently matched zero columns, discarding trio parental
    # data for EVERY variant in that run, not just one. Normalize whitespace/
    # underscore/hyphen runs to a single space first, same fix already
    # applied in _parent_label_from_name() below for the same reason.
    ab_cols = [
        str(c) for c in df.columns
        if "allelic balance" in re.sub(r"[_\-\s]+", " ", str(c).lower())
    ]

    if not ab_cols:
        return None

    # Preserve original column order (df.columns is already in CSV order)
    ab_cols_ordered = [c for c in df.columns if c in ab_cols]

    # Assign labels based on count
    n = len(ab_cols_ordered)
    if n == 1:
        labels = ["proband"]
    elif n >= 3:
        labels = _PARENTAL_LABELS_3  # proband, mother, father
    else:
        # 2 columns: proband + second. If the second column's own name names
        # its parent explicitly (e.g. "Allelic balance_mother"), use that
        # label directly instead of the generic "extra1" — otherwise a
        # correctly-present mother/father AB value is extracted but under a
        # key (segregation.classify_segregation() etc. look for "mother"/
        # "father" specifically) nothing downstream ever reads, which reads
        # identically to that parent's data being absent.
        second_label = _parent_label_from_name(ab_cols_ordered[1]) or "extra1"
        labels = ["proband", second_label]

    result = []
    for _, row in df.iterrows():
        entry = {}
        for col, label in zip(ab_cols_ordered, labels):
            val = _clean(row[col])
            # Keep as string — LLM will interpret the float value
            entry[label] = val
        result.append(entry)

    print(f"[csv_normalizer] Found {n} Allelic balance column(s): {labels}")
    print(f"[csv_normalizer] AB column names: {ab_cols_ordered}")
    print(f"[csv_normalizer] First variant AB entry: {result[0] if result else 'none'}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_upload(
    raw: bytes,
    filename: str,
    llm,
) -> tuple[list[dict], list[dict], list[dict] | None, str]:
    """
    Accept raw file bytes from any CSV or Excel upload and return a tuple of
    (normalized_variant_dicts, raw_field_dicts, parental_ab, header_mapping_summary).

    - normalized_variant_dicts: one dict per row, keyed by TARGET_COLUMNS
    - raw_field_dicts:          one dict per row, full original columns before normalization
    - parental_ab:              list of dicts with allelic balance values per variant,
                                or None if no "Allelic balance" columns found.
                                Keys depend on column count: "proband", "mother", "father" (3 cols),
                                "proband" + "extra1" (2 cols), or just "proband" (1 col).
    - header_mapping_summary:   human-readable text showing which canonical field the SLM
                                mapped each original header to — meant to be surfaced to the
                                user (SSE + final report) so they can sanity-check it.

    Column mapping is SLM-driven (_map_columns_llm) — `llm` must be an LLMClient
    (see pipeline.llm.base.LLMClient), used once per upload to classify headers.

    Supports:
      - .csv  (any delimiter — auto-detected)
      - .xlsx / .xls / .xlsm

    Raises:
      ValueError  if the file cannot be parsed or contains no rows.
    """
    fname = filename.lower()

    # ── Load into DataFrame ───────────────────────────────────────────────────
    try:
        if fname.endswith((".xlsx", ".xls", ".xlsm")):
            df = pd.read_excel(io.BytesIO(raw), dtype=str)
        else:
            # Try comma first, then tab, then semicolon
            text = raw.decode("utf-8", errors="replace")
            for sep in (",", "\t", ";"):
                try:
                    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
                    if len(df.columns) > 1:
                        break
                except Exception:
                    continue
            else:
                raise ValueError("Could not parse CSV with common delimiters (,  \\t  ;)")
    except Exception as e:
        raise ValueError(f"File parsing failed: {e}") from e

    if df.empty:
        raise ValueError("File contains no data rows.")

    # ── Normalize ─────────────────────────────────────────────────────────────
    # InterVar columns are never used: InterVar is an automated ACMG call, and
    # the header mapper read it as ClinVar_class in ~half of the uploads.
    df = df.drop(columns=[c for c in df.columns if "intervar" in str(c).lower()])

    df_original = df.copy()          # preserve original before column mapping

    df, header_mapping_summary = _map_columns_llm(df, llm)
    df = _build_normalized_df(df, df_original)

    if df.empty:
        raise ValueError("No variant rows found after normalization.")

    # Parental allelic balance: primarily from the SLM's own header mapping
    # (Allelic_balance_mother/_father — handles whatever naming convention the
    # file uses), falling back to the regex-based extract_parental_ab() only
    # if the SLM found no proband AB column at all.
    parental_ab = _build_parental_ab_from_mapped(df)
    if parental_ab is None:
        parental_ab = extract_parental_ab(df_original)

    variants  = _df_to_variant_dicts(df)
    raw_rows  = _extract_raw_fields(df_original)

    print(f"[csv_normalizer] {len(variants)} variants loaded from '{filename}'")

    # Log which target columns were actually populated
    populated = [c for c in TARGET_COLUMNS if any(df[c] != "NA")]
    missing   = [c for c in TARGET_COLUMNS if c not in populated]
    print(f"[csv_normalizer] Populated columns : {populated}")
    if missing:
        print(f"[csv_normalizer] Missing (→ NA)   : {missing}")

    return variants, raw_rows, parental_ab, header_mapping_summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI convenience  (python -m pipeline.core.normalizer input.csv)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.core.normalizer <input_file> [output_file]")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "normalized_output.csv"

    with open(input_path, "rb") as f:
        raw = f.read()

    import os as _os
    from pipeline.llm.vllm_client import VLLMClient
    _llm = VLLMClient(base_url=_os.environ.get("VLLM_BASE_URL", "http://localhost:8001"))

    variants, _, parental_ab, header_summary = normalize_upload(raw, filename=input_path, llm=_llm)
    print(f"\n{header_summary}\n")

    if parental_ab:
        print(f"\nParental AB data detected for {len(parental_ab)} variants:")
        for i, pa in enumerate(parental_ab[:3]):
            print(f"  Variant {i+1}: {pa}")
        if len(parental_ab) > 3:
            print(f"  ... ({len(parental_ab)} total)")

    # Write as CSV for inspection
    import csv as _csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(TARGET_COLUMNS)
        for v in variants:
            writer.writerow([v.get(c, "NA") for c in TARGET_COLUMNS])

    print(f"Saved normalized CSV to {output_path}")

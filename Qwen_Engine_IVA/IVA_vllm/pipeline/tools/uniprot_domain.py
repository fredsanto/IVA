"""
pipeline/tools/uniprot_domain.py — deterministic PM1 "critical functional domain"
check: the domain half of PM1, next to clinvar_hotspot.py's hotspot half.

For a missense variant, finds the UniProt domain containing its residue (feature
types Domain, DNA binding, Zinc finger — boundaries are UniProt's) and decides,
from data only, whether that domain is a critical region free of benign
variation:

  - pathogenic : ClinVar P/LP missense in the domain              >= MIN_PATHOGENIC
  - benign     : ClinVar B/LB missense in the domain, plus gnomAD v4 missense in the
                 domain with AF > COMMON_AF or >= MIN_HOMOZYGOTES homozygotes (not
                 already counted from ClinVar)                    <= MAX_BENIGN_FRACTION
                                                                     of pathogenic + benign
  - support    : >= 1 UniProt disease variant, mutagenesis entry, or active/binding/
                 metal/site feature in the domain (independent of ClinVar)

The variant's own change is excluded from every count and from the support, so it
never supports itself. Residue numbering is checked against the UniProt sequence:
a variant, ClinVar record or gnomAD variant whose reference residue doesn't match
UniProt at that position is on a differently numbered transcript and is left out
(the variant itself -> NOT EVALUABLE). Verdict line: "PM1 DOMAIN: MET | NOT MET |
NOT EVALUABLE" — read by core/acmg_pm1.py. The SLM never decides it.

UniProt, ClinVar (via ClinVarHotspotTool's class cache) and gnomAD are each
fetched once per gene and cached at class level.
"""

import logging
import re
import threading
import time

from pipeline.tools.base import NetworkTool
from pipeline.tools.clinvar_hotspot import ClinVarHotspotTool
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError
from pipeline.core.protein_change import _to_aa3

logger = logging.getLogger(__name__)

MIN_PATHOGENIC = 4
MAX_BENIGN_FRACTION = 0.10
COMMON_AF = 0.01
MIN_HOMOZYGOTES = 10

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
GNOMAD_API_URL = "https://gnomad.broadinstitute.org/api"
_REGION_TYPES = ("Domain", "DNA binding", "Zinc finger")
_SITE_TYPES = ("Active site", "Binding site", "Site")
_UNIPROT_FIELDS = "accession,sequence,ft_domain,ft_dna_bind,ft_zn_fing,ft_variant,ft_mutagen,ft_act_site,ft_binding,ft_site"
_GNOMAD_QUERY = """
query($gene: String!) {
  gene(gene_symbol: $gene, reference_genome: GRCh38) {
    variants(dataset: gnomad_r4) {
      variant_id consequence hgvsp
      exome { af homozygote_count }
      genome { af homozygote_count }
    }
  }
}"""
_HGVSP_RE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")
# UniProt natural-variant descriptions: "in CORD2; ..." marks a disease variant,
# "in dbSNP:rs..." a plain polymorphism.
_DISEASE_VARIANT_RE = re.compile(r"^in (?!dbSNP)[A-Za-z0-9]")
_NOT_DISEASE_RE = re.compile(r"(?i)uncertain significance|likely benign|\bbenign\b")
_AA1 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
        "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
        "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}


def _aa1(aa: str) -> str:
    return _AA1.get(_to_aa3(aa).capitalize(), aa.upper()[:1])


class UniProtDomainTool(NetworkTool):
    """
    PM1 domain check for a missense variant (see module docstring).

    gate():  same as clinvar_hotspot — at least one missense protein change.
    run():   one UniProt search + one gnomAD query per gene (ClinVar records
             shared with ClinVarHotspotTool), all cached at class level.
    """

    name = "uniprot_domain"
    description = (
        "Deterministic PM1 domain check: UniProt domain around the variant, ClinVar "
        "P/LP vs benign (ClinVar B/LB + common/homozygous gnomAD) missense in it, "
        "and independent UniProt support."
    )

    timeout: int = 60

    _uniprot_cache: dict[str, dict | None] = {}
    _gnomad_cache: dict[str, list[dict]] = {}
    _cache_lock = threading.Lock()

    def gate(self, variant: dict, context: ToolContext) -> bool:
        gene, hgvs = context.field("Gene"), context.field("HGVS")
        if gene == "NA" or hgvs == "NA":
            return False
        return bool(ClinVarHotspotTool._missense_seeds(hgvs))

    # ── fetches ──────────────────────────────────────────────────────────────
    def _retrying(self, what: str, fn):
        last = None
        for attempt in range(4):
            try:
                return fn()
            except Exception as e:  # rate limits (gnomAD 429) and transient errors
                last = e
                time.sleep(2 * 2 ** attempt)
        raise ToolFetchError(f"{what} failed after retries: {last}") from last

    def _fetch_uniprot(self, gene: str) -> dict | None:
        params = {"query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
                  "fields": _UNIPROT_FIELDS, "format": "json", "size": 1}
        resp = self._retrying(f"UniProt search for {gene}",
                              lambda: self._get(UNIPROT_URL, timeout=self.timeout, params=params))
        try:
            results = resp.json().get("results", [])
        except Exception as e:
            raise ToolParseError(f"UniProt response for {gene} not JSON: {e}") from e
        return results[0] if results else None

    def _fetch_gnomad(self, gene: str) -> list[dict]:
        def call():
            r = self._post(GNOMAD_API_URL, timeout=self.timeout,
                           json={"query": _GNOMAD_QUERY, "variables": {"gene": gene}},
                           headers={"Content-Type": "application/json"})
            data = r.json()
            if data.get("errors"):
                raise ToolFetchError(data["errors"][0].get("message", "gnomAD error"))
            return data
        data = self._retrying(f"gnomAD v4 variants for {gene}", call)
        return ((data.get("data") or {}).get("gene") or {}).get("variants") or []

    def _cached(self, cache: dict, gene: str, fetch):
        key = gene.upper()
        if key not in cache:
            with self._cache_lock:
                if key not in cache:
                    cache[key] = fetch(gene)
        return cache[key]

    # ── evaluation ───────────────────────────────────────────────────────────
    def run(self, variant: dict, context: ToolContext) -> str | None:
        gene, hgvs = context.field("Gene"), context.field("HGVS")
        seeds = ClinVarHotspotTool._missense_seeds(hgvs)
        if not seeds:
            return None
        header = f"UNIPROT DOMAIN EVIDENCE ({gene}):"

        entry = self._cached(self._uniprot_cache, gene, self._fetch_uniprot)
        if not entry:
            return f"{header}\nNo reviewed UniProt entry for {gene}.\nPM1 DOMAIN: NOT EVALUABLE"
        seq = (entry.get("sequence") or {}).get("value", "")
        feats = entry.get("features") or []

        seed = next(((r, p, a) for r, p, a in seeds if 0 < p <= len(seq) and seq[p - 1] == _aa1(r)), None)
        if seed is None:
            listed = ", ".join(f"p.{r}{p}{a}" for r, p, a in seeds)
            return (f"{header}\n{listed}: reference residue does not match UniProt {entry.get('primaryAccession')} "
                    "at that position (transcript numbering differs).\nPM1 DOMAIN: NOT EVALUABLE")
        ref, pos, alt = seed
        own = f"p.{ref}{pos}{alt}"

        regions = [f for f in feats if f.get("type") in _REGION_TYPES
                   and f["location"]["start"]["value"] <= pos <= f["location"]["end"]["value"]]
        if not regions:
            return (f"{header}\nResidue {own} is not inside a UniProt domain / DNA-binding / "
                    f"zinc-finger region ({entry.get('primaryAccession')}).\nPM1 DOMAIN: NOT MET")
        region = min(regions, key=lambda f: f["location"]["end"]["value"] - f["location"]["start"]["value"])
        start, end = region["location"]["start"]["value"], region["location"]["end"]["value"]
        name = f"{region['type']} \"{region.get('description') or region['type']}\" {start}-{end}"
        inside = lambda p: start <= p <= end

        def matches_seq(p: int, aa: str) -> bool:
            return 0 < p <= len(seq) and seq[p - 1] == _aa1(aa)

        records = ClinVarHotspotTool()._get_records(gene)
        if records is None:
            return (f"{header}\nResidue {own} in {name}: too many ClinVar missense records "
                    "to count.\nPM1 DOMAIN: NOT EVALUABLE")
        own_cdnas = set(re.findall(r"c\.[^\s:;|]+", hgvs))
        cv = [r for r in records if inside(r["pos"]) and matches_seq(r["pos"], r["ref"])
              and r["nt"] not in own_cdnas and not (r["pos"] == pos and _aa1(r["alt"]) == _aa1(alt))]
        plp = [r for r in cv if r["pathogenic"]]
        blb = [r for r in cv if not r["pathogenic"]]
        clinvar_changes = {(r["pos"], _aa1(r["alt"])) for r in cv}

        common = []
        for v in self._cached(self._gnomad_cache, gene, self._fetch_gnomad):
            m = _HGVSP_RE.match(v.get("hgvsp") or "")
            if v.get("consequence") != "missense_variant" or not m:
                continue
            vref, vpos, valt = m.group(1), int(m.group(2)), m.group(3)
            if not inside(vpos) or not matches_seq(vpos, vref) or (vpos, _aa1(valt)) in clinvar_changes:
                continue
            if vpos == pos and _aa1(valt) == _aa1(alt):
                continue
            ex, ge = v.get("exome") or {}, v.get("genome") or {}
            af = max(ex.get("af") or 0, ge.get("af") or 0)
            hom = (ex.get("homozygote_count") or 0) + (ge.get("homozygote_count") or 0)
            if af > COMMON_AF or hom >= MIN_HOMOZYGOTES:
                common.append((v["hgvsp"], af, hom, v.get("variant_id", "")))

        support = []
        for f in feats:
            s, e = f["location"]["start"]["value"], f["location"]["end"]["value"]
            if not (start <= s and e <= end):
                continue
            desc = f.get("description") or ""
            alts = (f.get("alternativeSequence") or {}).get("alternativeSequences") or []
            same_change = s == pos and _aa1(alt) in alts
            if f["type"] == "Natural variant" and _DISEASE_VARIANT_RE.match(desc) \
                    and not _NOT_DISEASE_RE.search(desc) and not same_change:
                support.append(f"disease variant {s} ({desc.split(';')[0]})")
            elif f["type"] == "Mutagenesis" and not same_change:
                support.append(f"mutagenesis {s}")
            elif f["type"] in _SITE_TYPES:
                support.append(f"{f['type'].lower()} {s}")

        n_benign = len(blb) + len(common)
        total = len(plp) + n_benign
        frac = n_benign / total if total else 0.0
        met = len(plp) >= MIN_PATHOGENIC and frac <= MAX_BENIGN_FRACTION and bool(support)
        lines = [
            header,
            f"Residue {own} lies in UniProt {entry.get('primaryAccession')} {name}.",
            f"ClinVar missense in this region: {len(plp)} P/LP, {len(blb)} B/LB (this variant excluded).",
            f"gnomAD v4 missense in this region counted as benign (AF > {COMMON_AF:g} or >= "
            f"{MIN_HOMOZYGOTES} homozygotes, not in ClinVar): {len(common)}"
            + "".join(f"\n  - {h} AF={af:.3g} homozygotes={hm} ({vid})" for h, af, hm, vid in common[:10]),
            f"Benign fraction: {frac:.0%} of {total} classified (limit {MAX_BENIGN_FRACTION:.0%}).",
            f"Independent UniProt support in region: {len(support)}"
            + (f" — {'; '.join(support[:6])}" if support else " — none"),
            f"Rule: >= {MIN_PATHOGENIC} ClinVar P/LP missense, benign <= {MAX_BENIGN_FRACTION:.0%}, "
            ">= 1 UniProt disease variant / mutagenesis / functional site in the region.",
            f"PM1 DOMAIN: {'MET' if met else 'NOT MET'}",
        ]
        return "\n".join(lines)

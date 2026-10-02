"""
pipeline/config.py — Global pipeline configuration constants.
"""

EXECUTION_MODE = "async"   # "async" | "sequential"

DEFAULT_GENOME_BUILD = "hg38"  # fallback when build detection finds no SNV to vote with (e.g. indel-only upload)

TRIAGE_ENABLED_THRESHOLD  = 0    # skip first_triage if n_variants <= this (0 = always run)

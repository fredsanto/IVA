"""
benchmark_runner.py
────────────────────────────────────────────────────────────────────────────────
Runs a MCQ benchmark (150 questions × 24 permutations) across a list of models.

All 4! = 24 answer orderings are tested exhaustively and deterministically —
permutation_id 0-23 maps to a fixed ordering of [A,B,C,D] answers, identical
across all models and all runs, so results are directly comparable.

Each model produces one output file:
    results/<safe_model_id>__<8-char-config-hash>_results.csv

The config hash encodes (model_id + revision + load kwargs) so two runs with
different weights/revisions never silently overwrite each other.

Output columns (per row = one question × one permutation):
    question_id | permutation_id | labels | is_correct
    logit_A | logit_B | logit_C | logit_D

    is_correct    : argmax(slot logits) == correct slot
    logit_A/B/C/D : logits anchored to ORIGINAL answer identity (not slot).
                    i.e. logit_A always = model's belief in the text that was
                    answer A in the source CSV, regardless of which slot it
                    occupied in this permutation.
                    → comparable and aggregable across all 24 permutations.
────────────────────────────────────────────────────────────────────────────────
"""

import gc
import hashlib
import itertools
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit this block only
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_CSV   = "large_mcq_150_labelled.csv"
RESULTS_DIR = Path("results")

# All 4! = 24 orderings of answer slots, fixed and shared across all models.
# permutation_id i always refers to the same ordering → results are comparable.
ALL_PERMUTATIONS: list[tuple[str, ...]] = list(itertools.permutations(["A", "B", "C", "D"]))
# ALL_PERMUTATIONS[0]  = ('A', 'B', 'C', 'D')  ← original order
# ALL_PERMUTATIONS[23] = ('D', 'C', 'B', 'A')  ← fully reversed

# Each entry: (model_id, optional revision, optional extra kwargs for from_pretrained)
# Examples:
#   ("google/gemma-4-E4B-it", None, {})
#   ("Qwen/Qwen2.5-7B-Instruct",           None, {})
#    ("Qwen/Qwen3.5-9B",           None, {}),
#    ("Qwen/Qwen3.5-4B",           None, {}),
#    ("Qwen/Qwen3.5-2B",           None, {}),
#    ("Qwen/Qwen3.5-0.8B",           None, {}),
#    ("Qwen/Qwen2.5-3B",           None, {}),
#   ("mistralai/Mistral-7B-Instruct-v0.3", None, {}),
#   ("meta-llama/Llama-3.1-8B-Instruct", None, {}),
#   ("microsoft/Phi-3-mini-128k-instruct", None, {}),
#   ("microsoft/MediPhi-Instruct", None, {}),
#    ("microsoft/MediPhi-Clinical", None, {}),
#    ("microsoft/MediPhi-Guidelines", None, {}),


MODELS: list[tuple[str, str | None, dict]] = [
    
    ("OpenMeditron/Meditron3-8B", None, {}),

]

# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE / DTYPE
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

LETTERS = ["A", "B", "C", "D"]


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT NAMING
# ═══════════════════════════════════════════════════════════════════════════════

def make_output_path(model_id: str, revision: str | None, extra_kwargs: dict) -> Path:
    """
    Unique filename = safe_slug + 8-char hash of (model_id, revision, extra_kwargs).
    Guarantees that same weights loaded differently → different files.
    """
    config_str = json.dumps(
        {"model_id": model_id, "revision": revision or "main", "kwargs": extra_kwargs},
        sort_keys=True,
    )
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    safe_slug   = model_id.replace("/", "_").replace("-", "_").replace(".", "_")
    filename    = f"{safe_slug}__{config_hash}_results.csv"
    return RESULTS_DIR / filename


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING / UNLOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_id: str, revision: str | None, extra_kwargs: dict):
    print(f"\n  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        **extra_kwargs,
    )
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer):
    """Free GPU memory before loading the next model."""
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════════════════════════
# DETECT MODEL FAMILY  →  drives prompt formatting quirks
# ═══════════════════════════════════════════════════════════════════════════════

def detect_family(model_id: str) -> str:
    mid = model_id.lower()
    if "mistral" in mid or "mixtral" in mid:
        return "mistral"
    if "qwen" in mid:
        return "qwen"
    if "gemma" in mid:
        return "gemma"
    return "default"   # llama, phi, falcon, ...


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDING  (per-family quirks)
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_MSG = (
    "You are a multiple-choice assistant. "
    "Respond with exactly one letter: A, B, C, or D. No explanation."
)


def build_messages(row: pd.Series, family: str) -> list[dict]:
    user_content = (
        f"{row['question']}\n"
        f"A: {row['A']}\n"
        f"B: {row['B']}\n"
        f"C: {row['C']}\n"
        f"D: {row['D']}\n"
        "Answer with only one letter: A, B, C, or D."
    )

    if family == "mistral":
        # Mistral chat templates do not support a system role
        return [{"role": "user", "content": f"{SYSTEM_MSG}\n\n{user_content}"}]

    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": user_content},
    ]


def apply_template(tokenizer, messages: list[dict], family: str) -> str:
    kwargs: dict = dict(tokenize=False, add_generation_prompt=True)
    if family == "qwen":
        kwargs["enable_thinking"] = False   # suppress <think> preamble
    return tokenizer.apply_chat_template(messages, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN IDS FOR A B C D
# ═══════════════════════════════════════════════════════════════════════════════

def get_token_ids(tokenizer) -> dict[str, int]:
    """
    Resolve the single vocab token that represents each answer letter.
    Works across BPE variants: 'A', 'A', 'GA', etc.
    Uses the last sub-token so prefix bytes don't matter.
    """
    ids = {}
    for letter in LETTERS:
        tokens = tokenizer.encode(letter, add_special_tokens=False)
        ids[letter] = tokens[-1]
    return ids


# ═══════════════════════════════════════════════════════════════════════════════
# LOGIT EXTRACTION  (single forward pass, no generation)
# ═══════════════════════════════════════════════════════════════════════════════

def get_slot_logits(
    model,
    tokenizer,
    token_ids: dict[str, int],
    messages: list[dict],
    family: str,
) -> dict[str, float]:
    """
    Returns raw pre-softmax logits keyed by SLOT (A/B/C/D as presented
    in this permutation). Remapping to original answer identity happens
    in run_model() using slot_order.
    """
    text   = apply_template(tokenizer, messages, family)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)

    last_logits = out.logits[0, -1, :]   # shape: (vocab_size,)
    return {l: last_logits[token_ids[l]].item() for l in LETTERS}


def remap_to_original(
    slot_logits: dict[str, float],
    slot_order: tuple[str, ...],
) -> dict[str, float]:
    """
    Remap slot logits → original answer identity logits.

    slot_order = ('C', 'A', 'D', 'B') means:
        slot A was filled with original answer C
        slot B was filled with original answer A
        slot C was filled with original answer D
        slot D was filled with original answer B

    So:  original_logits[C] = slot_logits[A]
         original_logits[A] = slot_logits[B]
         ...

    After remapping, logit_A always = model's belief in the text that
    was answer A in the source CSV, regardless of which slot it occupied.
    """
    original_logits = {}
    for new_slot, original_slot in zip(LETTERS, slot_order):
        original_logits[original_slot] = slot_logits[new_slot]
    return original_logits


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS  (live monitoring only — not stored)
# ═══════════════════════════════════════════════════════════════════════════════

def logits_to_probs(logits: dict[str, float]) -> dict[str, float]:
    t = torch.tensor([logits[l] for l in LETTERS], dtype=torch.float32)
    p = torch.softmax(t, dim=0).tolist()
    return dict(zip(LETTERS, p))


# ═══════════════════════════════════════════════════════════════════════════════
# PERMUTATION  (deterministic — no randomness)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_permutation(row: pd.Series, slot_order: tuple[str, ...]) -> pd.Series:
    """
    Rearrange answer texts according to slot_order.
    slot_order e.g. ('C','A','D','B') means slot A gets original answer C, etc.
    The correct answer label is updated accordingly.
    """
    correct_text = row[row["answer"].strip().upper()]

    new_row    = row.copy()
    new_answer = None
    for new_slot, original_slot in zip(LETTERS, slot_order):
        text = row[original_slot]
        new_row[new_slot] = text
        if text == correct_text:
            new_answer = new_slot

    new_row["answer"] = new_answer
    return new_row


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ONE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def run_model(
    model_id: str,
    revision: str | None,
    extra_kwargs: dict,
    df: pd.DataFrame,
) -> None:
    output_path = make_output_path(model_id, revision, extra_kwargs)
    family      = detect_family(model_id)
    n_perms     = len(ALL_PERMUTATIONS)   # always 24

    print(f"\n{'='*70}")
    print(f"  MODEL   : {model_id}")
    print(f"  REVISION: {revision or 'default'}")
    print(f"  FAMILY  : {family}")
    print(f"  PERMS   : {n_perms} (exhaustive 4!)")
    print(f"  OUTPUT  : {output_path}")
    print(f"{'='*70}")

    # resume support: skip already-completed (question_id, permutation_id) pairs
    done: set[tuple[int, int]] = set()
    if output_path.exists():
        prev = pd.read_csv(output_path, sep=";")
        done = set(zip(prev["question_id"], prev["permutation_id"]))
        print(f"  Resuming - {len(done)} rows already done.")

    model, tokenizer = load_model(model_id, revision, extra_kwargs)
    token_ids = get_token_ids(tokenizer)
    print(f"  Token ids - " + " ".join(f"{l}:{token_ids[l]}" for l in LETTERS))

    results: list[dict] = []
    t0 = time.time()

    for idx, original_row in df.iterrows():
        for p, slot_order in enumerate(ALL_PERMUTATIONS):

            if (idx, p) in done:
                continue

            row      = apply_permutation(original_row, slot_order)
            correct  = row["answer"].strip().upper()        # correct SLOT in this permutation
            messages = build_messages(row, family)

            try:
                # logits keyed by slot (A/B/C/D as shown to the model)
                slot_logits = get_slot_logits(model, tokenizer, token_ids, messages, family)

                # remap to original answer identity — logit_A always = belief in original answer A
                orig_logits = remap_to_original(slot_logits, slot_order)

                # correctness: argmax of slot logits == correct slot (equivalent to argmax of orig logits == original correct answer)
                chosen     = max(slot_logits, key=slot_logits.get)
                is_correct = int(chosen == correct)

                # for live print only
                probs     = logits_to_probs(slot_logits)
                p_correct = probs[correct]

                print(
                    f"  [{idx+1:3d}/{len(df)} | perm {p:2d}/23 {slot_order}] "
                    f"chosen={chosen} correct={correct} "
                    f"p_correct={p_correct:.3f} "
                    f"{'OK' if is_correct else 'X'}"
                )

            except Exception as exc:
                print(f"  [ERROR] row={idx+1} perm={p}: {exc}")
                orig_logits = {l: 0.0 for l in LETTERS}
                is_correct  = 0

            results.append({
                "question_id"   : idx,
                "permutation_id": p,
                "labels"        : original_row.get("labels", ""),
                "is_correct"    : is_correct,
                # logits anchored to original answer identity, comparable across permutations
                "logit_A"       : round(orig_logits["A"], 6),
                "logit_B"       : round(orig_logits["B"], 6),
                "logit_C"       : round(orig_logits["C"], 6),
                "logit_D"       : round(orig_logits["D"], 6),
            })

        # crash-safe: write after every question (all 24 permutations)
        all_results = results
        if done:   # merge with previously saved rows if resuming
            prev_df     = pd.read_csv(output_path, sep=";")
            all_results = prev_df.to_dict("records") + results
        pd.DataFrame(all_results).to_csv(output_path, index=False, sep=";")

    elapsed   = time.time() - t0
    total     = len(results)
    n_correct = sum(r["is_correct"] for r in results)

    print(f"\n  -- {model_id} --")
    print(f"  New rows  : {total}")
    print(f"  Accuracy  : {n_correct}/{total} = {n_correct/total:.2%}" if total else "  No new rows.")
    print(f"  Time      : {elapsed:.1f}s")
    print(f"  Saved  ->  {output_path}")

    unload_model(model, tokenizer)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, sep=";")

    print(f"Benchmark    : {INPUT_CSV}  ({len(df)} questions)")
    print(f"Permutations : {len(ALL_PERMUTATIONS)} (exhaustive 4!)")
    print(f"Total rows   : {len(df) * len(ALL_PERMUTATIONS)} per model")
    print(f"Models       : {len(MODELS)}")
    print(f"Device       : {DEVICE}")

    for model_id, revision, extra_kwargs in MODELS:
        run_model(model_id, revision, extra_kwargs, df)

    print(f"\n{'='*70}")
    print("All models done. Results in:", RESULTS_DIR.resolve())
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
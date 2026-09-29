# Pushing the pipeline subtree to fredsanto/IVA

Run these yourself (`!`-prefixed in chat, or directly in a terminal) — do not
paste your PAT into any command I run, only into git's own interactive
username/password prompt.

**First, rotate the PAT you pasted earlier in this session** — it's been
exposed in the transcript and must be treated as compromised. Generate a new
one at GitHub → Settings → Developer settings → Personal access tokens, then
use the new one below.

```bash
cd $PROJECT_ROOT

# 1. Split out just the pipeline subtree (not the whole monorepo) as its own
#    branch with linear history rewritten to that subdirectory's root.
git subtree split --prefix=ServerQwen/Qwen_Engine_IVA/IVA_vllm -b iva-export

# 2. Push that branch to IVA's main. Git will prompt for a username
#    and password — enter "fredsanto" and your NEW PAT as the password.
git push iva iva-export:main

# 3. Clean up the local export branch (optional).
git branch -D iva-export
```

## If step 2 is rejected (non-fast-forward)

That means `fredsanto/IVA` already has commits that don't share history
with this subtree split (e.g. it was seeded independently). Two options:

- **Merge instead of overwrite** (safer, keeps remote history):
  ```bash
  git fetch iva main
  git checkout iva-export
  git merge iva/main --allow-unrelated-histories
  # resolve any conflicts, then:
  git push iva iva-export:main
  ```
- **Force-overwrite remote main** (only if you're sure nothing on the remote
  needs keeping — this discards whatever is there now):
  ```bash
  git push iva iva-export:main --force-with-lease
  ```

## Context

- `iva` remote → `https://github.com/fredsanto/IVA.git` (private,
  owned by fredsanto).
- Only `ServerQwen/Qwen_Engine_IVA/IVA_vllm` is being pushed
  — not the rest of the GenMasterAI monorepo (unrelated LoRA scripts, other
  server dirs, etc. stay out of it).
- The pipeline's proprietary `LICENSE` file (copyright Eric Ducret) has been
  removed from this copy — confirmed authorized before doing so.

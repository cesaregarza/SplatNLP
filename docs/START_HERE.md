# SplatNLP — Start Here

This is a short “reviewer guide” to the repo: what it is, where the important
code lives, and what you can run quickly.

## What This Repo Demonstrates

- **A set-structured prediction problem** with hard constraints (Splatoon gear
  builds) framed as **multi-label set completion**.
- A **SetTransformer-style model** (`SetCompletionModel` / “SplatGPT”) with
  weapon-conditioned embeddings and permutation-invariant aggregation.
- **Constraint-aware reconstruction** (greedy closure + beam search + exact
  allocator) that turns token-space probabilities into legal build-space gear.
- **Mechanistic interpretability tooling**: a Sparse Autoencoder (SAE) trained
  on a 512D pooled activation, plus hooks to read/edit features during inference.

## Canonical Writeup (Blog)

The canonical, publishable blog posts live on `cegarza.com`:

- Blog home: https://cegarza.com/
- Series entry: https://cegarza.com/splatgpt-part-1/

Note: the `docs/splatgpt-blog-part-*.txt` / `docs/splatgpt-blog-part-*-draft.md`
files in this repo are **LLM-friendly extracts/drafts for tooling**, not the
canonical published posts.

## 5-Minute Tour (No Big Artifacts Required)

1. Run unit tests (fast, no dataset required):
   - `poetry install --with dev`
   - `poetry run pytest -q`
2. Skim the core model code:
   - `src/splatnlp/model/models.py` (SetTransformer layers + SetCompletionModel)
3. Skim reconstruction (how logits become a legal build):
   - `src/splatnlp/utils/reconstruct/`
4. Skim SAE + hooks:
   - `src/splatnlp/monosemantic_sae/`
   - `src/splatnlp/monosemantic_sae/hooks.py`
5. Optional: run the Colab demo:
   - `notebooks/colab_demo.ipynb`
   - https://colab.research.google.com/github/cesaregarza/SplatNLP/blob/main/notebooks/colab_demo.ipynb

## 30-Minute Tour (With Local Artifacts)

If you have local checkpoints/vocabs available, you can reproduce the “wow”
path on CPU:

1. Download artifacts (or point at your local copies):
   - `poetry run python -m splatnlp.utils.download_artifacts --dataset-dir dataset_v2`
   - (Optional) include Ultra + Ultra SAE:
     `poetry run python -m splatnlp.utils.download_artifacts --dataset-dir dataset_v2 --include-ultra-sae`
2. Run the local inference demo in `README.md` (loads from
   `saved_models/dataset_v2`).
3. Run the beam-search + SAE trace recipe from `AGENTS.md` (writes JSON output
   to `tmp_results/`).

## MechInterp Workflow (Deeper Dive)

- Workflow guide: `docs/mechinterp_workflow.md`

## Local-Only Artifacts (Ignore)

Depending on the workspace, you may see additional documents under `docs/`
generated as part of an LLM-assisted resume-building workflow (e.g.
`REPO_HIGHLIGHTS*.md`, `L4_MLE_REPORT.md`, `EVIDENCE_PACK.md`). These are not
canonical project documentation and may not be present in all clones.

## Repo Map (Where To Look)

- Model + training: `src/splatnlp/model/`
- Reconstruction + constraints: `src/splatnlp/utils/reconstruct/`
- SAE + hooks + training: `src/splatnlp/monosemantic_sae/`
- MechInterp experiments + activation server: `src/splatnlp/mechinterp/`
- Serving: `src/splatnlp/serve/`
- Tests: `tests/`

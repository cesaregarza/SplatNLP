# SplatNLP — Project One‑Pager (Research + Systems Summary)

## One‑line summary

**SplatNLP** is an end‑to‑end ML + mechanistic interpretability project that
frames Splatoon 3 gear building as **multi‑label set completion** conditioned on
weapon, then decodes predictions into **legal builds under hard constraints**.

## What this demonstrates (at a glance)

- A **set‑structured prediction problem** with real constraints and a clean
  mapping between “token space” and “build space”.
- A **SetTransformer‑style model** adapted for permutation invariance +
  weapon‑conditioned context.
- **Constraint‑aware decoding** (greedy closure + beam search + exact allocator)
  that turns probabilities into a legal discrete structure.
- A practical **mech‑interp stack**: SAE training, probe/edit hooks, activation
  storage, and tooling for repeated investigations.

## Key technical pieces

**Problem framing**
- Tokenization encodes gear builds as cumulative AP “capstone” tokens plus
  weapon ID, enabling multi‑label prediction with hard legality constraints.

**Model**
- `SetCompletionModel` (“SplatGPT”): SetTransformer‑style stack with
  weapon‑conditioned embeddings and masked mean pooling.
- Two trained variants referenced in `README.md`: **Full** and **Ultra**
  (both ~**83M params**).

**Decoding / constraints**
- **Greedy closure + beam search over capstones**, then an **exact allocator**
  that materializes a legal 3‑main/9‑sub build under AP and slot rules.

**Interpretability**
- **Sparse Autoencoder (SAE)** on the pooled 512‑D representation.
- Hooking supports **probe/no‑change** tracing vs **reconstruction/edit**
  interventions.
- Dashboard + efficient activation storage + local activation server for fast,
  repeatable feature investigations.

**Serving / systems**
- FastAPI inference service with artifact loading via env‑configured URLs.
- Docker/Make targets for packaging + repeatable runs.

## Evaluation (what’s here, what’s optional)

- Trained‑model compute notes live in `README.md` (Full vs Ultra).
- A reconstruction‑from‑partial‑info evaluation (Sendou) lives under
  `src/splatnlp/eval/` (entrypoint: `src/splatnlp/eval/sendou_compare.py`), with
  baselines and paired comparisons.
- The most “end‑to‑end wow path” depends on large local artifacts
  (checkpoints / activation DBs); the Colab demo + activation server exist to
  keep the review experience lightweight.

## Evidence map (best places to look in the repo)

**Fast reviewer path**
- Reviewer guide: `docs/START_HERE.md`
- MechInterp workflow: `docs/mechinterp_workflow.md`
- Colab demo: `notebooks/colab_demo.ipynb`
- Tests / CI: `tests/`, `.github/workflows/tests.yml`

**Core implementation**
- Model: `src/splatnlp/model/models.py`
- Training loop + DDP path: `src/splatnlp/model/training_loop.py`,
  `src/splatnlp/model/cli.py`
- Constraint‑aware decoding: `src/splatnlp/utils/reconstruct/beam_search.py`,
  `src/splatnlp/utils/reconstruct/allocator.py`
- SAE + hook (probe vs reconstruction/edit): `src/splatnlp/monosemantic_sae/`,
  `src/splatnlp/monosemantic_sae/hooks.py`
- Activation server (avoid repeated cold loads): `src/splatnlp/mechinterp/server/activation_server.py`
- FastAPI service: `src/splatnlp/serve/app.py`, `src/splatnlp/serve/load_model.py`
- Deploy packaging: `dockerfile`, `Makefile`

## How to verify quickly (no big artifacts)

- Install + run tests: `poetry install --with dev && poetry run pytest -q`
- If you only do one thing: open `notebooks/colab_demo.ipynb` (linked in
  `docs/START_HERE.md`) and “Run all”.
- Optional local eval (requires `test_data/` + `saved_models/` present):
  `poetry run python -m splatnlp.eval.sendou_compare --masks 1 2 3 --limit 50`

# SplatNLP - Project One-Pager (Research + Systems Summary)

## One-line summary

SplatNLP is an end-to-end ML + mechanistic interpretability project that frames
Splatoon 3 gear building as multi-label set completion conditioned on weapon,
then decodes predictions into legal builds under hard constraints.

Key numbers: ~83M params; pooled 512-D rep; Ultra SAE has 24,576 features.

## Concrete contributions (3)

- Constraint-aware set completion: decode learned token probabilities into a
  legal build under hard rules (3 mains / 9 subs, 57 AP budget, slot rules).
- MechInterp workflow + tooling: Sparse Autoencoder (SAE) on a pooled 512-D
  representation with probe hooks, anti-flanderization guidance (core vs
  tail), beam-trace attribution, and an activation server to avoid repeated
  cold loads.
- Eval + baselines: reconstruction-from-partial-info harness with baselines and
  case-level outputs for analysis (Sendou; entrypoint below).

## Quick reviewer paths (time budget)

- 5 minutes: open `notebooks/colab_demo.ipynb` and "Run all" (tokenization,
  decoding/legality, and an interpretability readout).
- 30 minutes: read `docs/START_HERE.md`, skim the core implementation files
  below, and (optionally) run the Sendou eval if you have local artifacts.
- For a curated “wow” path: start with `docs/SHOWCASE.md`.

## Evidence map (best places to look)

- Start here: `docs/START_HERE.md`
- MechInterp workflow notes: `docs/mechinterp_workflow.md`
- Model: `src/splatnlp/model/models.py`
- Training loop + DDP path: `src/splatnlp/model/training_loop.py`,
  `src/splatnlp/model/cli.py`
- Constraint-aware decoding: `src/splatnlp/utils/reconstruct/beam_search.py`,
  `src/splatnlp/utils/reconstruct/allocator.py`
- SAE + hook (probe vs edit): `src/splatnlp/monosemantic_sae/hooks.py`,
  `src/splatnlp/monosemantic_sae/models.py`
- Decoder-output experiment: `src/splatnlp/mechinterp/experiments/decoder_output.py`
- Activation server: `src/splatnlp/mechinterp/server/activation_server.py`
- Serving: `src/splatnlp/serve/app.py`, `src/splatnlp/serve/load_model.py`
- Tests / CI: `tests/`, `.github/workflows/tests.yml`

## How to verify quickly (no big artifacts)

- Install + run tests: `poetry install --with dev && poetry run pytest -q`
- Optional local eval (requires `test_data/` + `saved_models/` present):
  `poetry run python -m splatnlp.eval.sendou_compare --masks 1 2 3 --limit 50`

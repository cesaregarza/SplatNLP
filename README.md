# SplatNLP: constraint-aware set completion + SAE interpretability (Splatoon 3)

[![CI](https://github.com/cesaregarza/SplatNLP/actions/workflows/tests.yml/badge.svg)](https://github.com/cesaregarza/SplatNLP/actions/workflows/tests.yml) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cesaregarza/SplatNLP/blob/main/notebooks/colab_demo.ipynb)

SplatNLP frames Splatoon 3 gear building as a public, structured proxy for a
common ML task: **set completion under hard constraints from partial
observation**.

**TL;DR (2-minute skim)**
- Problem: **set completion under constraints from partial observation** (many
  valid solutions, strict legality rules).
- Model: **SplatGPT** — an ~83M parameter Transformer-style
  **`SetCompletionModel`** conditioned on weapon/context.
- Decoding: greedy closure + beam search + exact allocator → **legal**
  build-space outputs.
- Interpretability: **Sparse Autoencoder (SAE)** on a pooled 512‑D activation →
  concept readout + feature→token influence.
- Headline result: **0.673** `completion_slot_acc` on Sendou Tier‑1 **Mask‑6**
  (Top‑1, multi-reference). Full report:
  [`docs/sendou_tier1_eval_report.md`](docs/sendou_tier1_eval_report.md).
- Try it in ~2 minutes: [`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb)
  (Run all).

**Quick links**
- Canonical blog post: [SplatGPT: Set-Based Deep Learning for Splatoon 3 Gear Completion](https://cegarza.com/splatgpt-part-1/)
- Colab demo (recommended): [`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb)
- Eval report: [`docs/sendou_tier1_eval_report.md`](docs/sendou_tier1_eval_report.md)
- Docs index: [`docs/README.md`](docs/README.md)
- Reviewer guide: [`docs/START_HERE.md`](docs/START_HERE.md)
- Showcase path: [`docs/SHOWCASE.md`](docs/SHOWCASE.md)
- Mechinterp workflow: [`docs/mechinterp_workflow.md`](docs/mechinterp_workflow.md)
- Usage / CLI recipes: [`docs/USAGE.md`](docs/USAGE.md)
- Security notes: [`docs/SECURITY.md`](docs/SECURITY.md)
- Training notes (optional): [`docs/TRAINING.md`](docs/TRAINING.md)
- Project one-pager: [`docs/PROJECT_ONEPAGER.md`](docs/PROJECT_ONEPAGER.md)
- Resume bullets: [`docs/RESUME_BULLETS.md`](docs/RESUME_BULLETS.md)

**Model at a glance**
![SplatGPT architecture (simplified)](docs/assets/model.svg)
(Styled source: `docs/assets/model.html`)

## Main contributions

- **SplatGPT (`SetCompletionModel`)**: a permutation-invariant, SetTransformer-
  style architecture with weapon-conditioned embeddings and a multi-label
  output head, trained with aggressive randomized masking (Full/Ultra
  checkpoints) (`src/splatnlp/model/models.py`).
- **Constraint-aware decoding**: greedy closure + beam search + an exact
  allocator that converts token probabilities into legal builds under slot/AP
  constraints (`src/splatnlp/utils/reconstruct/`).
- **Evaluation that matches reality**: multi-reference scoring for
  many-valid-answer settings plus diagnostics for “coach-mode edits” vs strict
  completion (`docs/sendou_tier1_eval_report.md`).
- **SAE interpretability**: an SAE trained on pooled activations, probe-mode
  hooks (read features without changing outputs), and feature→token influence
  readouts (`src/splatnlp/monosemantic_sae/`).
- **Reproducibility**: CI tests, a Colab demo, and an artifact downloader for
  pretrained checkpoints (`src/splatnlp/utils/download_artifacts.py`).

## Results: Sendou Tier‑1 reconstruction from partial info

**Tier‑1** = curated expert builds from sendou.ink (top competitive players).

**Mask 6** = hardest partial-observation setting: drop 6 of the 12 gear
slot-items (mains/subs) at random, then ask the system to complete a full legal
build.

`completion_slot_acc` = fraction of the **missing** slot-items recovered by the
decoded Top‑1 build. Because many completions are valid, we also report a
**multi-reference** variant that scores against the closest Tier‑1 build
consistent with the observed slot-items.

Mask 6 (n=502):

| method | score |
| --- | ---: |
| random | 0.236 |
| conditional (weapon-scoped) | 0.630 |
| full | 0.656 |
| ultra | **0.673** |

Full tables, overlap/no-overlap slices, and edit behavior diagnostics:
[`docs/sendou_tier1_eval_report.md`](docs/sendou_tier1_eval_report.md).

## Interpretability highlight: SAE concepts → token influence

The Ultra SAE is trained on SplatGPT’s pooled 512‑D representation. In probe
mode we can read sparse feature activations during inference, then translate a
feature into “what tokens it pushes toward/away from” using decoder/output-layer
geometry.

Reproduce in [`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb)
(Ultra + SAE section). For deeper feature investigation workflows, see
[`docs/mechinterp_workflow.md`](docs/mechinterp_workflow.md).

## Product analog: workflow / dashboard recommendation (private-data domain)

Although this repo is framed around Splatoon gear builds, the core problem is
**set completion under constraints from partial observations**.

A close product analog is **workflow/dashboard recommendation**:
- Tokens ↔ dashboard widgets (with discrete size/bucket tokens)
- Context ↔ user type / role / task (“weapon id” here)
- Input ↔ a partial dashboard (some widgets chosen, some missing, some suboptimal)
- Output ↔ a completed (or improved) full dashboard configuration
- Data ↔ noisy usage logs; many valid solutions per context (multi-reference)

This project uses a fully public dataset as a proxy: constrained vocabulary,
compositional structure, and real-world noise. The “coach-mode” behavior
(allowing limited edits rather than strict preservation) maps to proposing a
better dashboard close to the user’s intent.

## Reproducibility

**Colab (recommended)**
- Open [`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb) and Run all.

**Local CPU quickstart**

1. Install deps and download pretrained artifacts:

   ```bash
   poetry install --with dev
   poetry run python -m splatnlp.utils.download_artifacts \
     --dataset-dir dataset_v2 --include-ultra-sae
   ```

2. Run a one-off inference:

   ```bash
   poetry run python - <<'PY'
   import json, torch
   from pathlib import Path
   from splatnlp.model.models import SetCompletionModel
   from splatnlp.serve.tokenize import tokenize_build
   from splatnlp.utils.constants import NULL, PAD
   
   base = Path("saved_models/dataset_v2")
   params = json.loads((base / "model_params.json").read_text())
   vocab = json.loads((base / "vocab.json").read_text())
   weapon_vocab = json.loads((base / "weapon_vocab.json").read_text())
   
   model = SetCompletionModel(**params)
   model.load_state_dict(torch.load(base / "model_ultra.pth", map_location="cpu"))
   model.eval()
   
   weapon_id = "weapon_id_8000"
   tokens = tokenize_build(
       {"ink_saver_main": 6, "run_speed_up": 12, "intensify_action": 12}
   )
   x = torch.tensor([[vocab[t] for t in tokens]])
   w = torch.tensor([[weapon_vocab[weapon_id]]])
   mask = x == params.get("pad_token_id", vocab[PAD])
   
   with torch.no_grad():
       probs = torch.sigmoid(model(x, w, key_padding_mask=mask)).squeeze(0)
   
   inv_vocab = {v: k for k, v in vocab.items()}
   skip = {vocab.get(PAD), vocab.get(NULL)}
   top = torch.topk(probs, k=12)
   rows = [
       (inv_vocab[i], float(p))
       for i, p in zip(top.indices.tolist(), top.values.tolist())
       if i not in skip
   ]
   print("context tokens:", tokens)
   print("top preds:", rows[:8])
   PY
   ```

**Training (optional)**
- Full training notes: [`docs/TRAINING.md`](docs/TRAINING.md)
- Longer recipes (serving, eval scripts, CLI examples):
  [`docs/USAGE.md`](docs/USAGE.md)

**Tests**
- `poetry run pytest -q`

## Security note

- Checkpoints are loaded with `torch.load` (pickle); only load artifacts you
  trust.
- The FastAPI server is intended for local demos; if you expose it, add auth,
  rate limiting, and restrict artifact sources. Details:
  [`docs/SECURITY.md`](docs/SECURITY.md).

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
See the [LICENSE](LICENSE) file for details.

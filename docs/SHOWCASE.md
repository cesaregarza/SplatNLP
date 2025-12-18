# Showcase — SplatNLP

If you only have 5 minutes:

1. Open the Colab demo and “Run all”:
   - `notebooks/colab_demo.ipynb`
   - https://colab.research.google.com/github/cesaregarza/SplatNLP/blob/main/notebooks/colab_demo.ipynb
2. Skim the build visualization output (baseline vs completion).
3. Skim the “Ultra + SAE” section (feature readouts + steering).

## What to look for

- **Set-structured prediction**: the model is *multi-label* (predicts every
  token independently) rather than autoregressive next-token prediction.
- **Hard constraints**: beam search + an exact allocator turn token
  probabilities into a **legal build** (slot rules + AP budget).
- **MechInterp hooks**: an SAE trained on a pooled 512-D activation exposes a
  sparse “feature space” you can read during inference.
- **Causal steering**: the demo performs an intervention on one SAE feature and
  shows how token probabilities / decoded builds change.

## Code map (best files)

- Model (`SetCompletionModel` / SplatGPT): `src/splatnlp/model/models.py`
- Constraint-aware decoding:
  - `src/splatnlp/utils/reconstruct/beam_search.py`
  - `src/splatnlp/utils/reconstruct/allocator.py`
- SAE + hook (probe vs recon/edit): `src/splatnlp/monosemantic_sae/hooks.py`
- SAE model: `src/splatnlp/monosemantic_sae/models.py`
- Activation server (for heavier mechinterp workflows):
  `src/splatnlp/mechinterp/server/activation_server.py`

## Good discussion prompts

- Why is this framed as multi-label set completion vs an autoregressive LM?
- What’s the right way to evaluate a constraint-aware decoder (beyond accuracy)?
- What does “feature steering” *not* claim (and what would make it more
  convincing)?
- If you had one more week, what experiment would you run to validate the SAE
  features as meaningful abstractions?

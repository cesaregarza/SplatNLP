# Resume bullets - SplatNLP

- Built an end-to-end **constraint-aware set completion** system for Splatoon 3
  gear builds (SplatGPT): weapon-conditioned set encoder + beam search + exact
  allocator to decode token probabilities into **legal builds under hard
  constraints** (3 mains / 9 subs, AP budget, slot legality).

- Designed an evaluation harness for **reconstruction from partial
  information** with strong baselines (weapon-scoped conditional + random fill)
  and diagnostics (Top-1 vs best-of-k, multi-reference Tier-1 set scoring,
  overlap/no-overlap slices) for a **multi-solution** domain.

- Implemented mechanistic interpretability tooling via a **Sparse Autoencoder**
  over a pooled 512-D representation (Ultra SAE: 24,576 features) with hooks for
  feature readout and reconstruction, plus automated labeling triage using
  graph-ranking (PageRank over token co-occurrence in high-activation examples)
  to prioritize high-signal concepts.

## Evidence

- Colab demo: `notebooks/colab_demo.ipynb`
- Eval report: `docs/sendou_tier1_eval_report.md`
- MechInterp workflow: `docs/mechinterp_workflow.md`
- One-pager: `docs/PROJECT_ONEPAGER.md`

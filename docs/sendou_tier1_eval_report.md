# Sendou Tier‑1 Build Reconstruction Eval (Full vs Ultra vs Baselines)

This report summarizes the current Sendou build “reconstruction from partial
info” evaluation, including statistical comparisons between:

- `full` model
- `ultra` model
- `conditional` (weapon‑scoped conditional baseline)
- `random` (true random fill baseline, conditioned only on mask count)

All results below correspond to **best‑of‑3** model outputs (`top_k=3`), i.e.
metrics are computed on the best scoring build among the top‑3 beam outputs.

## Quick Method Overview

- **Full model (`full`)**: Set-completion model trained on the same dataset as
  Ultra, but with a shorter training schedule (**5 subsets/epoch × 9 epochs =
  45 subsets per support**). At eval time it predicts ability tokens from the
  partially observed context and reconstructs a legal build via beam search +
  allocator (best‑of‑3).
- **Ultra model (`ultra`)**: Same dataset as Full, but trained much longer
  (**20 subsets/epoch × 20 epochs = 400 subsets per support**). Uses the same
  decoding procedure (beam search + allocator, best‑of‑3). Despite the extra
  training, it is statistically indistinguishable from Full on this eval.
- **Conditional baseline (`conditional`)**: A weapon‑scoped “most common
  completion” baseline. It searches only within tiers 2–3 builds for the same
  weapon and returns the most frequent build that *contains* the observed
  slot‑item multiset; if none match, it falls back to the weapon’s mode build.
- **Random baseline (`random`)**: A “true” baseline that preserves the observed
  slot‑items and fills the missing mains/subs by uniform sampling over ability
  families, subject only to legality constraints (main‑only canonical slot
  availability). It is conditioned only on the mask count (and whether missing
  slots are mains vs subs).

## Executive Summary

- **Full and Ultra are statistically indistinguishable** on this task across
  masks 1–6 for `best_accuracy`, and also for masked‑slot completion
  (`completion_slot_acc`).
- **Weapon‑scoped conditional is very strong** at predicting missing
  slot‑items (≈ **0.57–0.61** masked‑slot recovery across masks 1–6), but it is
  much worse on `best_accuracy` because it sometimes **fails to preserve the
  observed context** (falls back to weapon mode build when no candidate build
  contains the observed multiset).
- **Random fill is correctly “bad” at reconstruction**: for mask=1 it recovers
  the missing slot‑item only **~5.8%** of the time (as expected for uniform
  guessing among many abilities), but it can still have **high AP‑based**
  `best_accuracy` when only a small number of slots are missing.
- There is **exact build leakage across tiers**: **~21%** (105/502) of tier‑1
  builds have an identical (weapon, abilities) signature present in tiers 2–3.
  Conditional becomes near‑memorization on this subset.
- **The comparison is intentionally “unfair”, and that’s a feature**: tiers 2–3
  are still elite builds (and partially overlap tier‑1). This creates one of the
  few evaluations that has enough signal to meaningfully stress models, so we
  keep the strong baseline regime and also report overlap/no‑overlap slices for
  transparency.

## What’s Being Evaluated

Given:
- a weapon ID (as a vocab token `weapon_id_X`)
- a partially observed set of abilities (obtained by dropping N slot‑items from
  a full tier‑1 build)

…predict a complete legal build.

The evaluation is deliberately “brutal”: models must reconstruct expert builds
from limited information and must still respect game legality constraints.

## Data

**Source**
- `test_data/abilities-with-weapons.csv`

**Weapon ID normalization**
- Sendou `weaponSplId` values are mapped to their **reference kit** using
  `docs/weapon_info.json` (`reference_id`) so reskins collapse to the same
  weapon token.
- Builds with weapon IDs not present in the model’s weapon vocab
  (`saved_models/dataset_v0_2_full/weapon_vocab.json`) are dropped.

**Tier split (anti‑leak attempt)**
- Train pool for priors/baselines: tiers **2 & 3**
- Eval set: tier **1** only

After filtering and deduplication:
- Total builds: 3349
- Train builds (tiers 2–3): 2847
- Eval builds (tier 1): 502

**Residual overlap**

Even with tier splitting, there is still exact duplication:
- 105/502 (≈21%) of eval builds have an identical `(weapon_token, abilities_ap)`
  signature appearing in tiers 2–3.

The stats tooling reports results on:
- `all` (502 builds per mask)
- `no_overlap` (397 builds per mask)
- `overlap` (105 builds per mask)

## Case Generation (Masking)

Each eval build is converted from AP dict → slot‑items (e.g.
`swim_speed_up_sub`, `last_ditch_effort_main`) and then **N slot‑items are
randomly dropped** with a deterministic RNG seed.

Important: the *models* do not see exact slot‑items; the context they receive is
tokenized via `tokenize_build()`, which encodes **threshold tokens** (capstones)
instead of exact counts.

### How often are mains vs subs dropped?

Distribution of missing mains/subs in the eval cases:

| mask | missing mains/subs | frac |
| ---: | :----------------- | ---: |
| 1 | 0/1 | 0.775 |
| 1 | 1/0 | 0.225 |
| 2 | 0/2 | 0.546 |
| 2 | 1/1 | 0.410 |
| 2 | 2/0 | 0.044 |
| 3 | 0/3 | 0.353 |
| 3 | 1/2 | 0.538 |
| 3 | 2/1 | 0.102 |
| 3 | 3/0 | 0.008 |
| 4 | 0/4 | 0.231 |
| 4 | 1/3 | 0.534 |
| 4 | 2/2 | 0.217 |
| 4 | 3/1 | 0.018 |
| 5 | 0/5 | 0.167 |
| 5 | 1/4 | 0.492 |
| 5 | 2/3 | 0.301 |
| 5 | 3/2 | 0.040 |
| 6 | 0/6 | 0.076 |
| 6 | 1/5 | 0.434 |
| 6 | 2/4 | 0.398 |
| 6 | 3/3 | 0.092 |

This matters because:
- guessing a missing **main** is much harder than guessing a missing **sub**
  (and it hurts AP‑based scores more).

## Methods Compared

### random (true baseline)

Starts from the observed slot‑items and fills the missing mains/subs by sampling
ability families uniformly:

- Missing subs: uniform over `STANDARD_ABILITIES`
- Missing mains: uniform over `STANDARD_ABILITIES` plus any main‑only abilities
  whose canonical slots are still free

This baseline is conditioned only on “how many slots are missing” (and whether
those missing slots are mains vs subs).

### conditional (weapon‑scoped conditional baseline)

Builds a per‑weapon list of candidate builds from tiers 2–3, then:

- If any candidate build contains the observed multiset of slot‑items, choose
  the most frequent such candidate.
- Otherwise fall back to the weapon’s most common (“mode”) build.

This is weapon‑scoped (no pooling across all weapons), but it can still behave
like memorization on the overlap subset.

### full / ultra (models)

Both use the same decoding:
- Beam search with `beam_size=3`, `max_steps=8`, `top_k=3`
- Decoding predicts ability tokens, then uses the allocator to construct a legal
  build.

Checkpoints:
- Full: `saved_models/dataset_v0_2_full/model.pth`
- Ultra: `saved_models/dataset_v0_2_super/clean_slate.pth`

## Metrics

### best_accuracy (AP-based build similarity)

Defined as:

```
best_accuracy = 1 - (sum_abs_AP_diff / (57 * 2))
```

Where `sum_abs_AP_diff` sums the absolute AP differences per ability family
between truth and prediction. This is the repo’s existing AP‑based “accuracy”
signal.

Note: this metric can look high even for weak completions when only a small
number of slot‑items are missing.

### completion_slot_acc (masked-slot recovery)

This isolates **only the missing slot‑items**:

- Let `missing = truth_slots - observed_slots`
- Let `added = pred_slots - observed_slots`

Then:

```
completion_slot_acc = |added ∩ missing| / |missing|
```

This prevents “cheating” by crediting a method for slot‑items that were already
visible in the context.

## Statistical Methodology

All comparisons are **paired** (same case_id across methods):

- Mean ± 95% bootstrap CI (10,000 resamples)
- Paired t-test p-value
- Wilcoxon signed-rank p-value

Outputs were produced by:
- `src/splatnlp/eval/sendou_compare.py`
- `src/splatnlp/eval/sendou_stats.py`

## Results (All cases, n=502 per mask)

### completion_slot_acc (mean [95% CI])

| mask | random | conditional | full | ultra |
| ---: | :----- | :---------- | :--- | :---- |
| 1 | 0.0578 [0.0378,0.0797] | 0.5697 [0.5259,0.6135] | 0.5717 [0.5279,0.6155] | 0.5518 [0.5080,0.5956] |
| 2 | 0.1056 [0.0867,0.1255] | 0.5777 [0.5458,0.6086] | 0.5807 [0.5508,0.6116] | 0.5956 [0.5647,0.6265] |
| 3 | 0.1288 [0.1129,0.1461] | 0.5671 [0.5392,0.5950] | 0.5837 [0.5591,0.6082] | 0.5870 [0.5611,0.6129] |
| 4 | 0.1434 [0.1285,0.1579] | 0.5817 [0.5568,0.6061] | 0.6046 [0.5822,0.6265] | 0.6071 [0.5857,0.6290] |
| 5 | 0.1689 [0.1558,0.1825] | 0.6104 [0.5880,0.6327] | 0.6163 [0.5968,0.6359] | 0.6179 [0.5984,0.6375] |
| 6 | 0.1936 [0.1809,0.2062] | 0.5797 [0.5581,0.6006] | 0.6149 [0.5969,0.6325] | 0.6169 [0.5983,0.6351] |

Key read:
- Conditional is strong (≈0.57–0.61).
- Models are only modestly better on missing-slot recovery, but consistently so
  at higher masks.
- Full and Ultra are extremely close.

### best_accuracy (mean [95% CI])

| mask | random | conditional | full | ultra |
| ---: | :----- | :---------- | :--- | :---- |
| 1 | 0.9235 [0.9187,0.9282] | 0.7454 [0.7275,0.7628] | 0.9351 [0.9304,0.9396] | 0.9359 [0.9316,0.9400] |
| 2 | 0.8491 [0.8422,0.8560] | 0.7580 [0.7408,0.7749] | 0.9166 [0.9113,0.9219] | 0.9174 [0.9121,0.9226] |
| 3 | 0.7771 [0.7694,0.7848] | 0.7624 [0.7448,0.7797] | 0.8986 [0.8918,0.9052] | 0.8970 [0.8902,0.9038] |
| 4 | 0.7124 [0.7040,0.7206] | 0.7681 [0.7509,0.7848] | 0.8738 [0.8659,0.8816] | 0.8753 [0.8674,0.8831] |
| 5 | 0.6492 [0.6401,0.6581] | 0.7801 [0.7639,0.7953] | 0.8607 [0.8521,0.8691] | 0.8624 [0.8538,0.8709] |
| 6 | 0.5918 [0.5823,0.6016] | 0.7636 [0.7482,0.7789] | 0.8334 [0.8234,0.8433] | 0.8344 [0.8245,0.8440] |

Important read:
- Random looks “strong” at low masks because it always preserves the observed
  slots; `best_accuracy` is not a pure completion metric.
- Conditional is much worse than random at mask=1 on `best_accuracy` because it
  sometimes **does not preserve observed slots** (fallback to mode build).

## Paired Comparisons (Highlights)

### Full vs Ultra

- `best_accuracy`: no meaningful difference at any mask (mean diffs ~0, CIs
  cross 0, p-values not significant).
- `completion_slot_acc`: no meaningful difference at any mask.

Interpretation: on this eval, both models learn essentially the same predictive
signal, and the bottleneck is likely the (1) coarseness of the context tokens
and (2) the allocator/beam search search space rather than model capacity.

### Models vs Conditional (completion_slot_acc)

At low masks (1–2), conditional is already close to the models.
At higher masks, models pull ahead; for example:

- mask 6: full − conditional ≈ +0.035 (p≈2e‑3 paired t-test; p≈2e‑3 Wilcoxon)

### Random vs Everything

Random is a sanity check:
- Extremely low missing-slot recovery, rising slowly with mask count
  (more chances to guess some missing items).
- High `best_accuracy` at low masks due to preserving the observed context.

## Overlap vs No-Overlap Subsets

The overlap subset (exact (weapon, build) signature appears in train tiers)
creates a “memorization-friendly” regime:

- Conditional becomes near‑perfect on missing-slot recovery for low masks.
- Models remain strong but are not expected to beat pure memorization here.

The no-overlap subset is closer to the intended “fair” evaluation:
- Conditional drops materially on missing-slot recovery.
- Full/Ultra stay strong and generally remain above conditional.

### completion_slot_acc on overlap builds (n=105 per mask)

| mask | random | conditional | full | ultra |
| ---: | :----- | :---------- | :--- | :---- |
| 1 | 0.0381 [0.0095,0.0762] | 0.8952 [0.8286,0.9524] | 0.6286 [0.5333,0.7143] | 0.6095 [0.5143,0.7048] |
| 2 | 0.1524 [0.1048,0.2048] | 0.8762 [0.8238,0.9238] | 0.6762 [0.6095,0.7381] | 0.7238 [0.6667,0.7810] |
| 3 | 0.1270 [0.0921,0.1651] | 0.8857 [0.8444,0.9238] | 0.6921 [0.6444,0.7365] | 0.6921 [0.6381,0.7460] |
| 4 | 0.1595 [0.1286,0.1905] | 0.8476 [0.8048,0.8881] | 0.6857 [0.6381,0.7310] | 0.7000 [0.6548,0.7452] |
| 5 | 0.1810 [0.1543,0.2095] | 0.8667 [0.8324,0.8990] | 0.6857 [0.6400,0.7276] | 0.7029 [0.6590,0.7448] |
| 6 | 0.2048 [0.1794,0.2317] | 0.8063 [0.7603,0.8492] | 0.6762 [0.6333,0.7175] | 0.6937 [0.6508,0.7365] |

### completion_slot_acc on no-overlap builds (n=397 per mask)

| mask | random | conditional | full | ultra |
| ---: | :----- | :---------- | :--- | :---- |
| 1 | 0.0630 [0.0403,0.0882] | 0.4836 [0.4358,0.5315] | 0.5567 [0.5063,0.6071] | 0.5365 [0.4887,0.5844] |
| 2 | 0.0932 [0.0743,0.1134] | 0.4987 [0.4647,0.5327] | 0.5554 [0.5202,0.5894] | 0.5617 [0.5252,0.5970] |
| 3 | 0.1293 [0.1108,0.1478] | 0.4828 [0.4542,0.5105] | 0.5550 [0.5273,0.5835] | 0.5592 [0.5315,0.5877] |
| 4 | 0.1392 [0.1228,0.1555] | 0.5113 [0.4861,0.5359] | 0.5831 [0.5586,0.6083] | 0.5825 [0.5586,0.6064] |
| 5 | 0.1657 [0.1511,0.1809] | 0.5426 [0.5204,0.5642] | 0.5980 [0.5763,0.6196] | 0.5955 [0.5738,0.6171] |
| 6 | 0.1906 [0.1763,0.2049] | 0.5197 [0.4987,0.5399] | 0.5987 [0.5793,0.6184] | 0.5966 [0.5764,0.6163] |

## Reproducibility

Cached eval outputs:
- Masks 1–3:
  `tmp_results/sendou_compare_train2-3_eval1_masks1-2-3_limit0_seed42_beam3_steps8_top3.json`
- Masks 4–6:
  `tmp_results/sendou_compare_train2-3_eval1_masks4-5-6_limit0_seed42_beam3_steps8_top3.json`

Stats output (this report’s source of numbers):
- `tmp_results/sendou_stats_train2-3_eval1_masks1-6_seed0_boot10000_addedCompletion.json`

Commands:

```bash
poetry run python -m splatnlp.eval.sendou_compare \
  --methods conditional random full ultra \
  --train-tiers 2 3 --eval-tiers 1 \
  --masks 1 2 3 --limit 0 --seed 42 \
  --beam-size 3 --max-steps 8 --top-k 3

poetry run python -m splatnlp.eval.sendou_compare \
  --methods conditional random full ultra \
  --train-tiers 2 3 --eval-tiers 1 \
  --masks 4 5 6 --limit 0 --seed 42 \
  --beam-size 3 --max-steps 8 --top-k 3

poetry run python -m splatnlp.eval.sendou_stats \
  --in \
    tmp_results/sendou_compare_train2-3_eval1_masks1-2-3_limit0_seed42_beam3_steps8_top3.json \
    tmp_results/sendou_compare_train2-3_eval1_masks4-5-6_limit0_seed42_beam3_steps8_top3.json \
  --metrics best_accuracy completion_slot_acc exact_hit \
  --bootstrap 10000 --ci 0.95 --seed 0 --split-overlap \
  --out tmp_results/sendou_stats_train2-3_eval1_masks1-6_seed0_boot10000_addedCompletion.json
```

## Recommendations / Next Steps

1. **Add a context-preservation metric**
   - e.g. `observed_slot_recall = |pred ∩ observed| / |observed|`
   - This would explain why conditional has good completion but poor
     `best_accuracy` at low masks.

2. **Make conditional “always respect context”**
   - Instead of falling back to weapon mode build when no candidate contains the
     observed multiset, choose the candidate with maximum multiset overlap, or
     fill the remaining slots randomly while preserving the observed ones.

3. **De-leak the train pool**
   - For “strict no memorization”, remove from tiers 2–3 any build signature
     that appears in tier‑1 (per weapon), then recompute baselines.

4. **Increase `top_k` if desired**
   - If you want “best-of-5”, rerun `sendou_compare.py` with `--top-k 5` (more
     expensive, but should improve models and not baselines).

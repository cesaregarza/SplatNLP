# Sendou Tier-1 Build Reconstruction Eval (Full vs Ultra vs Baselines)

This report summarizes the current Sendou build "reconstruction from partial
info" evaluation, including statistical comparisons between:

- `full` model
- `ultra` model
- `conditional` (weapon-scoped conditional baseline)
- `random` (true random fill baseline, conditioned only on mask count)

Most tables report the original **oracle best-of-3** model outputs
(`top_k=3`), i.e. metrics are computed on the best scoring build among the
top-3 beam outputs. This report also includes **Top-1** (non-oracle) metrics
and a **multi-reference Tier-1 set** scoring view (see "Additional diagnostic
metrics").

## Quick Method Overview

- **Full model (`full`)**: Set-completion model trained on the same dataset as
  Ultra, but with a shorter training schedule (**5 subsets/epoch x 9 epochs =
  45 subsets per support**). At eval time it predicts ability tokens from the
  partially observed context and reconstructs a legal build via beam search +
  allocator (best-of-3).
- **Ultra model (`ultra`)**: Same dataset as Full, but trained much longer
  (**20 subsets/epoch x 20 epochs = 400 subsets per support**). Uses the same
  decoding procedure (beam search + allocator, best-of-3). On the original
  single-reference oracle metrics it is very close to Full, but diagnostic
  metrics show a small Ultra advantage.
- **Conditional baseline (`conditional`)**: A weapon-scoped "most common
  completion" baseline. It searches only within tiers 2-3 builds for the same
  weapon and returns the most frequent build that *contains* the observed
  slot-item multiset; if none match, it falls back to the weapon's mode build.
- **Random baseline (`random`)**: A "true" baseline that preserves the observed
  slot-items and fills the missing mains/subs by uniform sampling over ability
  families, subject only to legality constraints (main-only canonical slot
  availability). It is conditioned only on the mask count (and whether missing
  slots are mains vs subs).

## Executive Summary

- **Single-reference, oracle best-of-3:** Full and Ultra are extremely close on
  `best_accuracy` and `completion_slot_acc` across masks 1-6.
- **Top-1 + multi-reference Tier-1 set:** Ultra is modestly stronger on
  `tier1_set_completion_slot_acc_top1` (slot-item-based), especially at higher
  masks and on the divergence slice (cases where Full and Ultra top-1 disagree).
  Example (all cases, mask 6): Full 0.6557 vs Ultra 0.6730 (Full - Ultra
  -0.0173; paired t-test p=0.009; Wilcoxon p=0.004).
- **Weapon-scoped conditional is strong** at predicting missing slot-items, but
  it often violates the observed slot-item context due to mode-build fallback,
  which hurts AP-based similarity at low masks.
- **Random fill is correctly "bad" at reconstruction**: for mask=1 it recovers
  the missing slot-item only ~5.8% of the time, but it can still have high
  `best_accuracy` when only a small number of slots are missing.
- There is **exact build leakage across tiers**: ~21% (105/502) of Tier-1
  builds have an identical (weapon, abilities) signature present in tiers 2-3.
  Conditional becomes near-memorization on this subset.
- **The comparison is intentionally "unfair", and that's a feature**: tiers 2-3
  are still elite builds (and partially overlap Tier-1). This creates one of
  the few evaluations that has enough signal to stress models, so we keep the
  strong baseline regime and also report overlap/no-overlap slices.

## What's Being Evaluated

Given:
- a weapon ID (as a vocab token `weapon_id_X`)
- a partially observed set of abilities (obtained by dropping N slot-items from
  a full Tier-1 build)

...predict a complete legal build.

The evaluation is deliberately "brutal": models must reconstruct expert builds
from limited information and must still respect game legality constraints.

## Data

**Source**
- `test_data/abilities-with-weapons.csv`

**Weapon ID normalization**
- Sendou `weaponSplId` values are mapped to their **reference kit** using
  `docs/weapon_info.json` (`reference_id`) so reskins collapse to the same
  weapon token.
- Builds with weapon IDs not present in the model's weapon vocab
  (`saved_models/dataset_v0_2_full/weapon_vocab.json`) are dropped.

**Tier split (anti-leak attempt)**
- Train pool for priors/baselines: tiers **2 & 3**
- Eval set: tier **1** only

After filtering and deduplication:
- Total builds: 3349
- Train builds (tiers 2-3): 2847
- Eval builds (tier 1): 502

**Residual overlap**

Even with tier splitting, there is still exact duplication:
- 105/502 (~21%) of eval builds have an identical `(weapon_token, abilities_ap)`
  signature appearing in tiers 2-3.

The stats tooling reports results on:
- `all` (502 builds per mask)
- `no_overlap` (397 builds per mask)
- `overlap` (105 builds per mask)

## Case Generation (Masking)

Each eval build is converted from AP dict -> slot-items (e.g.
`swim_speed_up_sub`, `last_ditch_effort_main`) and then **N slot-items are
randomly dropped** with a deterministic RNG seed.

Important: the *models* do not see exact slot-items; the context they receive is
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
  (and it hurts AP-based scores more).

## Methods Compared

### random (true baseline)

Starts from the observed slot-items and fills the missing mains/subs by sampling
ability families uniformly:

- Missing subs: uniform over `STANDARD_ABILITIES`
- Missing mains: uniform over `STANDARD_ABILITIES` plus any main-only abilities
  whose canonical slots are still free

This baseline is conditioned only on "how many slots are missing" (and whether
those missing slots are mains vs subs).

### conditional (weapon-scoped conditional baseline)

Builds a per-weapon list of candidate builds from tiers 2-3, then:

- If any candidate build contains the observed multiset of slot-items, choose
  the most frequent such candidate.
- Otherwise fall back to the weapon's most common ("mode") build.

This is weapon-scoped (no pooling across all weapons), but it can still behave
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
between truth and prediction. This is the repo's existing AP-based "accuracy"
signal.

Note: this metric can look high even for weak completions when only a small
number of slot-items are missing.

### completion_slot_acc (masked-slot recovery)

This isolates **only the missing slot-items**:

- Let `missing = truth_slots - observed_slots`
- Let `added = pred_slots - observed_slots`

Then:

```
completion_slot_acc = |added & missing| / |missing|
```

This prevents "cheating" by crediting a method for slot-items that were already
visible in the context.

### Additional diagnostic metrics (implemented in sendou_stats.py)

The tables in this report use the original single-reference, best-of-k framing
that `sendou_compare.py` produces by default. For analysis, `sendou_stats.py`
also supports:

Top-1 is available without re-running inference because the cached compare JSON
stores both `predicted_top1_achieved_ap` and `predicted_best_achieved_ap` per
case (for `top_k=3`).

**Top-1 (non-oracle) metrics**
- `top1_best_accuracy`
- `top1_completion_slot_acc`
- `top1_exact_hit`

**Context preservation**
- `top1_observed_slot_recall = |pred_slots & observed_slots| / |observed_slots|`
- `top1_context_violation = 1` if any observed slot-item is missing in `pred`

**Multi-reference Tier-1 set scoring (wrong-but-good)**

Single-reference scoring punishes alternative Tier-1 builds that are still
plausible given the observed context. To reduce that failure mode,
`sendou_stats.py` can score predictions against the closest Tier-1 build for
the same weapon that is consistent with the observed slot-item multiset:

- `tier1_set_best_accuracy` / `tier1_set_best_accuracy_top1`
- `tier1_set_completion_slot_acc` / `tier1_set_completion_slot_acc_top1`
- `tier1_set_exact_hit` / `tier1_set_exact_hit_top1`

A Tier-1 reference build is eligible iff it contains all observed slot-items as
a multiset (exact main/sub slot-items).

When both methods are present, `sendou_stats.py` also reports a
`full_ultra_divergent` slice (cases where Full and Ultra top-1 predictions
differ), plus overlap/no-overlap sub-slices if `--split-overlap` is enabled.

## Statistical Methodology

All comparisons are **paired** (same case_id across methods):

- Mean with 95% bootstrap CI (10,000 resamples)
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
- Conditional is strong (~0.57-0.61).
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
- Random looks "strong" at low masks because it always preserves the observed
  slots; `best_accuracy` is not a pure completion metric.
- Conditional is much worse than random at mask=1 on `best_accuracy` because it
  sometimes **does not preserve observed slots** (fallback to mode build).

## Diagnostic Results (Top-1 + Multi-reference Tier-1 Set)

These diagnostics use the cached `sendou_compare.py` outputs plus
`sendou_stats.py`; no re-run of model inference is needed.

### Top-1 completion_slot_acc (mean [95% CI])

| mask | random | conditional | full | ultra |
| ---: | :--- | :--- | :--- | :--- |
| 1 | 0.0578 [0.0378,0.0797] | 0.5697 [0.5259,0.6135] | 0.4721 [0.4283,0.5159] | 0.4801 [0.4363,0.5239] |
| 2 | 0.1056 [0.0867,0.1255] | 0.5777 [0.5458,0.6086] | 0.4950 [0.4631,0.5269] | 0.5179 [0.4861,0.5488] |
| 3 | 0.1288 [0.1122,0.1454] | 0.5671 [0.5392,0.5950] | 0.5279 [0.5033,0.5525] | 0.5405 [0.5153,0.5657] |
| 4 | 0.1434 [0.1295,0.1579] | 0.5817 [0.5563,0.6071] | 0.5433 [0.5209,0.5652] | 0.5568 [0.5339,0.5792] |
| 5 | 0.1689 [0.1558,0.1817] | 0.6104 [0.5884,0.6327] | 0.5777 [0.5586,0.5968] | 0.5845 [0.5645,0.6028] |
| 6 | 0.1936 [0.1809,0.2062] | 0.5797 [0.5584,0.6006] | 0.5780 [0.5601,0.5959] | 0.5876 [0.5694,0.6062] |

Key read:
- Top-1 is lower than oracle best-of-3 for the models (expected).
- Ultra is slightly higher than Full across masks; the clearest gap is mask 2
  (paired t-test p=0.038; Wilcoxon p=0.080).

### Tier-1 set completion_slot_acc_top1 (mean [95% CI])

This scores each prediction against the closest Tier-1 build (same weapon)
that contains the observed slot-item multiset.

| mask | random | conditional | full | ultra |
| ---: | :--- | :--- | :--- | :--- |
| 1 | 0.0578 [0.0378,0.0797] | 0.5837 [0.5398,0.6255] | 0.4880 [0.4442,0.5319] | 0.4980 [0.4542,0.5418] |
| 2 | 0.1116 [0.0916,0.1325] | 0.6026 [0.5707,0.6345] | 0.5149 [0.4831,0.5458] | 0.5408 [0.5100,0.5717] |
| 3 | 0.1454 [0.1288,0.1627] | 0.6142 [0.5870,0.6415] | 0.5677 [0.5425,0.5930] | 0.5837 [0.5578,0.6089] |
| 4 | 0.1683 [0.1534,0.1838] | 0.6389 [0.6140,0.6633] | 0.6006 [0.5772,0.6235] | 0.6185 [0.5961,0.6409] |
| 5 | 0.1984 [0.1849,0.2124] | 0.6673 [0.6454,0.6892] | 0.6367 [0.6163,0.6562] | 0.6446 [0.6247,0.6641] |
| 6 | 0.2364 [0.2224,0.2507] | 0.6637 [0.6434,0.6839] | 0.6557 [0.6365,0.6746] | 0.6730 [0.6541,0.6919] |

Key read:
- Multi-reference scoring is materially higher than single-reference at higher
  masks, consistent with a multi-solution domain.
- Ultra > Full for every mask. The strongest gap is mask 6: Full - Ultra
  -0.0173 (paired t-test p=0.009; Wilcoxon p=0.004).

### Divergence slice (Full != Ultra, Top-1)

Number of divergent cases per mask:
- mask 1: 229
- mask 2: 269
- mask 3: 282
- mask 4: 302
- mask 5: 302
- mask 6: 313

Tier-1 set completion_slot_acc_top1 on the divergence slice:

| mask | full | ultra |
| ---: | :--- | :--- |
| 1 | 0.4061 [0.3450,0.4672] | 0.4279 [0.3668,0.4934] |
| 2 | 0.4888 [0.4498,0.5297] | 0.5372 [0.4963,0.5781] |
| 3 | 0.5331 [0.4988,0.5674] | 0.5615 [0.5284,0.5946] |
| 4 | 0.5811 [0.5513,0.6109] | 0.6109 [0.5811,0.6391] |
| 5 | 0.6159 [0.5907,0.6424] | 0.6291 [0.6033,0.6550] |
| 6 | 0.6198 [0.5948,0.6438] | 0.6475 [0.6230,0.6715] |

Key read:
- The Ultra advantage is larger when the models disagree. In particular:
  - mask 2: Full - Ultra -0.0483 (paired t-test p=0.022; Wilcoxon p=0.030)
  - mask 6: Full - Ultra -0.0277 (paired t-test p=0.009; Wilcoxon p=0.0016)

### No-overlap check (Tier-1 set, Top-1)

Tier-1 set completion_slot_acc_top1 on the no-overlap slice:

| mask | full | ultra |
| ---: | :--- | :--- |
| 1 | 0.4610 [0.4131,0.5088] | 0.4836 [0.4358,0.5340] |
| 2 | 0.4798 [0.4446,0.5151] | 0.5088 [0.4748,0.5428] |
| 3 | 0.5416 [0.5139,0.5701] | 0.5550 [0.5273,0.5827] |
| 4 | 0.5863 [0.5598,0.6121] | 0.6033 [0.5775,0.6285] |
| 5 | 0.6217 [0.6000,0.6443] | 0.6252 [0.6030,0.6463] |
| 6 | 0.6440 [0.6230,0.6646] | 0.6566 [0.6364,0.6767] |

### Context preservation (Top-1)

top1_observed_slot_recall (mean [95% CI]):

| mask | random | conditional | full | ultra |
| ---: | :--- | :--- | :--- | :--- |
| 1 | 1.0000 [1.0000,1.0000] | 0.6956 [0.6749,0.7162] | 0.8589 [0.8461,0.8716] | 0.8558 [0.8435,0.8680] |
| 2 | 1.0000 [1.0000,1.0000] | 0.7293 [0.7074,0.7508] | 0.8616 [0.8492,0.8735] | 0.8711 [0.8592,0.8829] |
| 3 | 1.0000 [1.0000,1.0000] | 0.7590 [0.7368,0.7809] | 0.8776 [0.8656,0.8891] | 0.8727 [0.8606,0.8849] |
| 4 | 1.0000 [1.0000,1.0000] | 0.7839 [0.7615,0.8065] | 0.8693 [0.8556,0.8822] | 0.8720 [0.8583,0.8852] |
| 5 | 1.0000 [1.0000,1.0000] | 0.8187 [0.7954,0.8409] | 0.8782 [0.8645,0.8919] | 0.8853 [0.8714,0.8987] |
| 6 | 1.0000 [1.0000,1.0000] | 0.8612 [0.8396,0.8818] | 0.8831 [0.8689,0.8964] | 0.8918 [0.8778,0.9054] |

top1_context_violation (mean [95% CI]):

| mask | random | conditional | full | ultra |
| ---: | :--- | :--- | :--- | :--- |
| 1 | 0.0000 [0.0000,0.0000] | 0.7390 [0.6992,0.7769] | 0.6813 [0.6414,0.7211] | 0.7052 [0.6653,0.7430] |
| 2 | 0.0000 [0.0000,0.0000] | 0.6434 [0.6016,0.6853] | 0.6614 [0.6215,0.7032] | 0.6474 [0.6056,0.6892] |
| 3 | 0.0000 [0.0000,0.0000] | 0.5578 [0.5139,0.6016] | 0.5777 [0.5359,0.6215] | 0.5777 [0.5359,0.6215] |
| 4 | 0.0000 [0.0000,0.0000] | 0.4861 [0.4422,0.5299] | 0.5378 [0.4940,0.5817] | 0.5458 [0.5040,0.5896] |
| 5 | 0.0000 [0.0000,0.0000] | 0.4004 [0.3566,0.4442] | 0.4741 [0.4303,0.5179] | 0.4562 [0.4143,0.5000] |
| 6 | 0.0000 [0.0000,0.0000] | 0.2888 [0.2490,0.3287] | 0.4422 [0.3984,0.4861] | 0.4064 [0.3625,0.4502] |

Key read:
- `random` is perfect on these by construction: it starts from the observed
  slot-items and only fills the missing ones, so it never drops observed items.
- `conditional` is much worse because it sometimes falls back to a mode build
  that does not contain the observed multiset.

## Paired Comparisons (Highlights)

### Full vs Ultra

- Oracle best-of-3 (single-reference):
  - `best_accuracy`: no meaningful difference at any mask (mean diffs ~0, CIs
    cross 0, p-values not significant).
  - `completion_slot_acc`: no meaningful difference at any mask.
- Diagnostic views:
  - `top1_completion_slot_acc`: Ultra is slightly higher; the clearest gap is
    mask 2 (paired t-test p=0.038; Wilcoxon p=0.080).
  - `tier1_set_completion_slot_acc_top1`: Ultra is higher for every mask; mask
    6 has the strongest gap (paired t-test p=0.009; Wilcoxon p=0.004), and the
    divergence slice shows larger differences.

Interpretation: the original single-reference oracle view largely hides
differences between Full and Ultra, while Top-1 and multi-reference scoring
surface a modest Ultra advantage (especially when the models disagree), which
is consistent with "wrong-but-good" alternatives in a multi-solution domain.

### Models vs Conditional (completion_slot_acc)

At low masks (1-2), conditional is already close to the models.
At higher masks, models pull ahead; for example:

- mask 6: full - conditional ~ +0.035 (p~2e-3 paired t-test; p~2e-3 Wilcoxon)

### Random vs Everything

Random is a sanity check:
- Extremely low missing-slot recovery, rising slowly with mask count
  (more chances to guess some missing items).
- High `best_accuracy` at low masks due to preserving the observed context.

## Overlap vs No-Overlap Subsets

The overlap subset (exact (weapon, build) signature appears in train tiers)
creates a "memorization-friendly" regime:

- Conditional becomes near-perfect on missing-slot recovery for low masks.
- Models remain strong but are not expected to beat pure memorization here.

The no-overlap subset is closer to the intended "fair" evaluation:
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
- Masks 1-3:
  `tmp_results/sendou_compare_train2-3_eval1_masks1-2-3_limit0_seed42_beam3_steps8_top3.json`
- Masks 4-6:
  `tmp_results/sendou_compare_train2-3_eval1_masks4-5-6_limit0_seed42_beam3_steps8_top3.json`

Stats outputs (this report's source of numbers):
- Oracle single-reference tables:
  `tmp_results/sendou_stats_train2-3_eval1_masks1-6_seed0_boot10000_addedCompletion.json`
- Diagnostic tables (Top-1 + Tier-1 set):
  `tmp_results/sendou_stats_train2-3_eval1_masks1-6_seed0_boot10000_multirefTier1SlotBased.json`

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

# Diagnostic metrics (uses the same compare JSONs, no re-run needed):
poetry run python -m splatnlp.eval.sendou_stats \
  --in \
    tmp_results/sendou_compare_train2-3_eval1_masks1-2-3_limit0_seed42_beam3_steps8_top3.json \
    tmp_results/sendou_compare_train2-3_eval1_masks4-5-6_limit0_seed42_beam3_steps8_top3.json \
  --metrics \
    top1_best_accuracy top1_completion_slot_acc top1_exact_hit \
    top1_observed_slot_recall top1_context_violation \
    tier1_set_best_accuracy_top1 tier1_set_completion_slot_acc_top1 \
  --bootstrap 10000 --ci 0.95 --seed 0 --split-overlap \
  --out tmp_results/sendou_stats_train2-3_eval1_masks1-6_seed0_boot10000_multirefTier1SlotBased.json
```

## Recommendations / Next Steps

1. **Report context-preservation metrics**
   - Implemented in `sendou_stats.py` as `top1_observed_slot_recall` and
     `top1_context_violation`.
   - This explains why conditional can have good completion but poor
     `best_accuracy` at low masks.

2. **Make conditional "always respect context"**
   - Instead of falling back to weapon mode build when no candidate contains the
     observed multiset, choose the candidate with maximum multiset overlap, or
     fill the remaining slots randomly while preserving the observed ones.

3. **De-leak the train pool**
   - For "strict no memorization", remove from tiers 2-3 any build signature
     that appears in Tier-1 (per weapon), then recompute baselines.

4. **Increase `top_k` if desired**
   - If you want "best-of-5", rerun `sendou_compare.py` with `--top-k 5` (more
     expensive, but should improve models and not baselines).

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
  slot-item multiset; if none match, it uses the weapon's mode build as a
  template but preserves the observed slot-items and fills only the missing
  slots.
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
- **Weapon-scoped conditional (context-preserving)** never drops observed
  slot-items, but it is notably weaker than the models on `completion_slot_acc`
  outside the overlap slice (it becomes near-memorization only on overlap).
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
| 1 | 0.0578 [0.0378,0.0797] | 0.3546 [0.3127,0.3964] | 0.5717 [0.5299,0.6155] | 0.5518 [0.5080,0.5956] |
| 2 | 0.1056 [0.0867,0.1255] | 0.4233 [0.3894,0.4572] | 0.5807 [0.5488,0.6116] | 0.5956 [0.5647,0.6255] |
| 3 | 0.1288 [0.1122,0.1454] | 0.4542 [0.4243,0.4841] | 0.5837 [0.5598,0.6076] | 0.5870 [0.5611,0.6122] |
| 4 | 0.1434 [0.1295,0.1579] | 0.5115 [0.4846,0.5378] | 0.6046 [0.5827,0.6265] | 0.6071 [0.5852,0.6290] |
| 5 | 0.1689 [0.1558,0.1821] | 0.5578 [0.5339,0.5813] | 0.6163 [0.5968,0.6363] | 0.6179 [0.5984,0.6375] |
| 6 | 0.1936 [0.1813,0.2062] | 0.5468 [0.5246,0.5691] | 0.6149 [0.5966,0.6328] | 0.6169 [0.5983,0.6348] |

Key read:
- Conditional is near-memorization on overlap, but much weaker on no-overlap.
- Models outperform conditional on missing-slot recovery across masks 1-6.
- Full and Ultra are extremely close.

### best_accuracy (mean [95% CI])

| mask | random | conditional | full | ultra |
| ---: | :----- | :---------- | :--- | :---- |
| 1 | 0.9235 [0.9187,0.9281] | 0.9504 [0.9456,0.9551] | 0.9351 [0.9306,0.9397] | 0.9359 [0.9315,0.9400] |
| 2 | 0.8491 [0.8423,0.8558] | 0.9100 [0.9031,0.9168] | 0.9166 [0.9112,0.9220] | 0.9174 [0.9123,0.9227] |
| 3 | 0.7771 [0.7693,0.7850] | 0.8711 [0.8621,0.8801] | 0.8986 [0.8918,0.9052] | 0.8970 [0.8902,0.9038] |
| 4 | 0.7124 [0.7038,0.7208] | 0.8514 [0.8417,0.8611] | 0.8738 [0.8659,0.8817] | 0.8753 [0.8674,0.8830] |
| 5 | 0.6492 [0.6401,0.6582] | 0.8378 [0.8274,0.8479] | 0.8607 [0.8521,0.8690] | 0.8624 [0.8540,0.8710] |
| 6 | 0.5918 [0.5821,0.6018] | 0.8001 [0.7881,0.8118] | 0.8334 [0.8235,0.8430] | 0.8344 [0.8248,0.8439] |

Important read:
- Random looks "strong" at low masks because it always preserves the observed
  slots; `best_accuracy` is not a pure completion metric.
- Conditional also preserves observed slot-items by construction, so its strong
  `best_accuracy` at low masks is expected; use `completion_slot_acc` (and the
  Tier-1 set variant) to focus on missing-slot recovery.

## Diagnostic Results (Top-1 + Multi-reference Tier-1 Set)

These diagnostics use the cached `sendou_compare.py` outputs plus
`sendou_stats.py`; no re-run of model inference is needed.

### Top-1 completion_slot_acc (mean [95% CI])

| mask | random | conditional | full | ultra |
| ---: | :--- | :--- | :--- | :--- |
| 1 | 0.0578 [0.0378,0.0797] | 0.3546 [0.3127,0.3964] | 0.4721 [0.4283,0.5159] | 0.4801 [0.4363,0.5239] |
| 2 | 0.1056 [0.0867,0.1255] | 0.4233 [0.3904,0.4562] | 0.4950 [0.4641,0.5259] | 0.5179 [0.4871,0.5488] |
| 3 | 0.1288 [0.1129,0.1454] | 0.4542 [0.4250,0.4841] | 0.5279 [0.5033,0.5525] | 0.5405 [0.5153,0.5657] |
| 4 | 0.1434 [0.1295,0.1579] | 0.5115 [0.4846,0.5378] | 0.5433 [0.5209,0.5652] | 0.5568 [0.5339,0.5792] |
| 5 | 0.1689 [0.1558,0.1821] | 0.5578 [0.5343,0.5817] | 0.5777 [0.5586,0.5968] | 0.5845 [0.5653,0.6036] |
| 6 | 0.1936 [0.1809,0.2065] | 0.5468 [0.5246,0.5687] | 0.5780 [0.5601,0.5956] | 0.5876 [0.5687,0.6062] |

Key read:
- Top-1 is lower than oracle best-of-3 for the models (expected).
- Ultra is slightly higher than Full across masks; the clearest gap is mask 2
  (paired t-test p=0.038; Wilcoxon p=0.080).

### Tier-1 set completion_slot_acc_top1 (mean [95% CI])

This scores each prediction against the closest Tier-1 build (same weapon)
that contains the observed slot-item multiset.

| mask | random | conditional | full | ultra |
| ---: | :--- | :--- | :--- | :--- |
| 1 | 0.0578 [0.0378,0.0797] | 0.3685 [0.3267,0.4104] | 0.4880 [0.4442,0.5319] | 0.4980 [0.4542,0.5418] |
| 2 | 0.1116 [0.0916,0.1315] | 0.4472 [0.4133,0.4821] | 0.5149 [0.4841,0.5458] | 0.5408 [0.5100,0.5717] |
| 3 | 0.1454 [0.1282,0.1627] | 0.4973 [0.4675,0.5279] | 0.5677 [0.5425,0.5930] | 0.5837 [0.5578,0.6089] |
| 4 | 0.1683 [0.1534,0.1838] | 0.5657 [0.5388,0.5931] | 0.6006 [0.5772,0.6235] | 0.6185 [0.5961,0.6409] |
| 5 | 0.1984 [0.1845,0.2127] | 0.6131 [0.5892,0.6375] | 0.6367 [0.6163,0.6566] | 0.6446 [0.6251,0.6641] |
| 6 | 0.2364 [0.2221,0.2503] | 0.6301 [0.6082,0.6524] | 0.6557 [0.6371,0.6746] | 0.6730 [0.6544,0.6912] |

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
| 1 | 0.4061 [0.3450,0.4716] | 0.4279 [0.3624,0.4934] |
| 2 | 0.4888 [0.4480,0.5297] | 0.5372 [0.4981,0.5781] |
| 3 | 0.5331 [0.5000,0.5662] | 0.5615 [0.5272,0.5946] |
| 4 | 0.5811 [0.5505,0.6109] | 0.6109 [0.5820,0.6391] |
| 5 | 0.6159 [0.5901,0.6417] | 0.6291 [0.6026,0.6550] |
| 6 | 0.6198 [0.5953,0.6438] | 0.6475 [0.6225,0.6720] |

Key read:
- The Ultra advantage is larger when the models disagree. In particular:
  - mask 2: Full - Ultra -0.0483 (paired t-test p=0.022; Wilcoxon p=0.030)
  - mask 6: Full - Ultra -0.0277 (paired t-test p=0.009; Wilcoxon p=0.0016)

### No-overlap check (Tier-1 set, Top-1)

Tier-1 set completion_slot_acc_top1 on the no-overlap slice:

| mask | full | ultra |
| ---: | :--- | :--- |
| 1 | 0.4610 [0.4131,0.5113] | 0.4836 [0.4358,0.5315] |
| 2 | 0.4798 [0.4446,0.5151] | 0.5088 [0.4736,0.5441] |
| 3 | 0.5416 [0.5139,0.5693] | 0.5550 [0.5273,0.5819] |
| 4 | 0.5863 [0.5605,0.6121] | 0.6033 [0.5775,0.6291] |
| 5 | 0.6217 [0.5995,0.6433] | 0.6252 [0.6030,0.6463] |
| 6 | 0.6440 [0.6230,0.6646] | 0.6566 [0.6364,0.6772] |

### Context preservation (Top-1)

top1_observed_slot_recall (mean [95% CI]):

| mask | random | conditional | full | ultra |
| ---: | :--- | :--- | :--- | :--- |
| 1 | 1.0000 [1.0000,1.0000] | 1.0000 [1.0000,1.0000] | 0.8589 [0.8461,0.8716] | 0.8558 [0.8435,0.8680] |
| 2 | 1.0000 [1.0000,1.0000] | 1.0000 [1.0000,1.0000] | 0.8616 [0.8492,0.8737] | 0.8711 [0.8592,0.8829] |
| 3 | 1.0000 [1.0000,1.0000] | 1.0000 [1.0000,1.0000] | 0.8776 [0.8659,0.8891] | 0.8727 [0.8603,0.8849] |
| 4 | 1.0000 [1.0000,1.0000] | 1.0000 [1.0000,1.0000] | 0.8693 [0.8556,0.8822] | 0.8720 [0.8583,0.8852] |
| 5 | 1.0000 [1.0000,1.0000] | 1.0000 [1.0000,1.0000] | 0.8782 [0.8645,0.8916] | 0.8853 [0.8717,0.8987] |
| 6 | 1.0000 [1.0000,1.0000] | 1.0000 [1.0000,1.0000] | 0.8831 [0.8695,0.8967] | 0.8918 [0.8778,0.9057] |

top1_context_violation (mean [95% CI]):

| mask | random | conditional | full | ultra |
| ---: | :--- | :--- | :--- | :--- |
| 1 | 0.0000 [0.0000,0.0000] | 0.0000 [0.0000,0.0000] | 0.6813 [0.6414,0.7211] | 0.7052 [0.6653,0.7430] |
| 2 | 0.0000 [0.0000,0.0000] | 0.0000 [0.0000,0.0000] | 0.6614 [0.6215,0.7032] | 0.6474 [0.6056,0.6892] |
| 3 | 0.0000 [0.0000,0.0000] | 0.0000 [0.0000,0.0000] | 0.5777 [0.5339,0.6215] | 0.5777 [0.5339,0.6195] |
| 4 | 0.0000 [0.0000,0.0000] | 0.0000 [0.0000,0.0000] | 0.5378 [0.4940,0.5817] | 0.5458 [0.5040,0.5896] |
| 5 | 0.0000 [0.0000,0.0000] | 0.0000 [0.0000,0.0000] | 0.4741 [0.4303,0.5160] | 0.4562 [0.4143,0.5000] |
| 6 | 0.0000 [0.0000,0.0000] | 0.0000 [0.0000,0.0000] | 0.4422 [0.3984,0.4861] | 0.4064 [0.3645,0.4502] |

Key read:
- `random` and `conditional` are perfect on these by construction: they start
  from the observed slot-items and only fill missing ones.
- Models are not perfect here because they do not see exact slot-items (only
  threshold tokens) and because reconstruction is constrained by legality.

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

Conditional is substantially weaker than the models for every mask on
`completion_slot_acc`, especially on the no-overlap slice at low masks. For
example:

- mask 1 (all): full - conditional ~ +0.217
- mask 6 (all): full - conditional ~ +0.068

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
| 1 | 0.0381 [0.0095,0.0762] | 0.8952 [0.8381,0.9524] | 0.6286 [0.5333,0.7143] | 0.6095 [0.5143,0.7048] |
| 2 | 0.1524 [0.1048,0.2048] | 0.8762 [0.8238,0.9238] | 0.6762 [0.6143,0.7381] | 0.7238 [0.6667,0.7810] |
| 3 | 0.1270 [0.0921,0.1652] | 0.8857 [0.8444,0.9238] | 0.6921 [0.6444,0.7397] | 0.6921 [0.6381,0.7429] |
| 4 | 0.1595 [0.1286,0.1905] | 0.8476 [0.8024,0.8881] | 0.6857 [0.6381,0.7310] | 0.7000 [0.6571,0.7429] |
| 5 | 0.1810 [0.1524,0.2095] | 0.8667 [0.8305,0.8990] | 0.6857 [0.6419,0.7295] | 0.7029 [0.6590,0.7448] |
| 6 | 0.2048 [0.1778,0.2317] | 0.8063 [0.7603,0.8508] | 0.6762 [0.6333,0.7190] | 0.6937 [0.6508,0.7349] |

### completion_slot_acc on no-overlap builds (n=397 per mask)

| mask | random | conditional | full | ultra |
| ---: | :----- | :---------- | :--- | :---- |
| 1 | 0.0630 [0.0403,0.0882] | 0.2116 [0.1713,0.2519] | 0.5567 [0.5088,0.6045] | 0.5365 [0.4887,0.5844] |
| 2 | 0.0932 [0.0743,0.1134] | 0.3035 [0.2733,0.3338] | 0.5554 [0.5202,0.5907] | 0.5617 [0.5264,0.5970] |
| 3 | 0.1293 [0.1117,0.1486] | 0.3401 [0.3140,0.3661] | 0.5550 [0.5273,0.5835] | 0.5592 [0.5306,0.5877] |
| 4 | 0.1392 [0.1234,0.1555] | 0.4225 [0.3980,0.4477] | 0.5831 [0.5586,0.6077] | 0.5825 [0.5579,0.6071] |
| 5 | 0.1657 [0.1506,0.1814] | 0.4761 [0.4534,0.4982] | 0.5980 [0.5758,0.6202] | 0.5955 [0.5733,0.6166] |
| 6 | 0.1906 [0.1763,0.2053] | 0.4782 [0.4572,0.4992] | 0.5987 [0.5789,0.6184] | 0.5966 [0.5772,0.6167] |

## Reproducibility

Cached eval outputs:
- Masks 1-3:
  `tmp_results/sendou_compare_train2-3_eval1_masks1-2-3_limit0_seed42_beam3_steps8_top3_condfix.json`
- Masks 4-6:
  `tmp_results/sendou_compare_train2-3_eval1_masks4-5-6_limit0_seed42_beam3_steps8_top3_condfix.json`

Stats outputs (this report's source of numbers):
- Oracle single-reference tables:
  `tmp_results/sendou_stats_train2-3_eval1_masks1-6_seed0_boot10000_addedCompletion_condfix.json`
- Diagnostic tables (Top-1 + Tier-1 set):
  `tmp_results/sendou_stats_train2-3_eval1_masks1-6_seed0_boot10000_multirefTier1SlotBased_condfix.json`

Commands:

```bash
poetry run python -m splatnlp.eval.sendou_compare \
  --methods conditional random full ultra \
  --train-tiers 2 3 --eval-tiers 1 \
  --masks 1 2 3 --limit 0 --seed 42 \
  --beam-size 3 --max-steps 8 --top-k 3 \
  --out tmp_results/sendou_compare_train2-3_eval1_masks1-2-3_limit0_seed42_beam3_steps8_top3_condfix.json

poetry run python -m splatnlp.eval.sendou_compare \
  --methods conditional random full ultra \
  --train-tiers 2 3 --eval-tiers 1 \
  --masks 4 5 6 --limit 0 --seed 42 \
  --beam-size 3 --max-steps 8 --top-k 3 \
  --out tmp_results/sendou_compare_train2-3_eval1_masks4-5-6_limit0_seed42_beam3_steps8_top3_condfix.json

poetry run python -m splatnlp.eval.sendou_stats \
  --in \
    tmp_results/sendou_compare_train2-3_eval1_masks1-2-3_limit0_seed42_beam3_steps8_top3_condfix.json \
    tmp_results/sendou_compare_train2-3_eval1_masks4-5-6_limit0_seed42_beam3_steps8_top3_condfix.json \
  --metrics best_accuracy completion_slot_acc exact_hit \
  --bootstrap 10000 --ci 0.95 --seed 0 --split-overlap \
  --out tmp_results/sendou_stats_train2-3_eval1_masks1-6_seed0_boot10000_addedCompletion_condfix.json

# Diagnostic metrics (uses the same compare JSONs, no re-run needed):
poetry run python -m splatnlp.eval.sendou_stats \
  --in \
    tmp_results/sendou_compare_train2-3_eval1_masks1-2-3_limit0_seed42_beam3_steps8_top3_condfix.json \
    tmp_results/sendou_compare_train2-3_eval1_masks4-5-6_limit0_seed42_beam3_steps8_top3_condfix.json \
  --metrics \
    top1_best_accuracy top1_completion_slot_acc top1_exact_hit \
    top1_observed_slot_recall top1_context_violation \
    tier1_set_best_accuracy_top1 tier1_set_completion_slot_acc_top1 \
  --bootstrap 10000 --ci 0.95 --seed 0 --split-overlap \
  --out tmp_results/sendou_stats_train2-3_eval1_masks1-6_seed0_boot10000_multirefTier1SlotBased_condfix.json
```

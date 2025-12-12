---
name: mechinterp-investigator
description: Orchestrate a systematic research program to investigate and meaningfully label SAE features
---

# MechInterp Investigator

This skill guides a systematic investigation of SAE features to arrive at meaningful, non-trivial labels. It orchestrates the other mechinterp skills into a coherent research workflow.

## Phase 0: Triage (ALWAYS START HERE)

**Goal:** Quickly filter out weak/auxiliary features that don't warrant deep investigation.

**Time:** 1-2 minutes

Many SAE features have minimal influence on model outputs. Triage identifies these early so you can skip expensive analysis.

### Step 0.1: Check Decoder Weight Percentile

```python
import torch

sae_path = '/mnt/e/dev_spillover/SplatNLP/sae_runs/run_20250704_191557/sae_model_final.pth'
sae_checkpoint = torch.load(sae_path, map_location='cpu', weights_only=True)
decoder_weight = sae_checkpoint['decoder.weight']  # [512, 24576]

# Get this feature's max absolute decoder weight
feature_decoder = decoder_weight[:, FEATURE_ID]
max_abs = torch.abs(feature_decoder).max().item()

# Compare to all features
all_max_abs = torch.abs(decoder_weight).max(dim=0).values
percentile = (all_max_abs < max_abs).float().mean() * 100

print(f"Feature {FEATURE_ID} decoder weight percentile: {percentile:.1f}%")
```

| Percentile | Action |
|------------|--------|
| < 10% | **Likely weak** - check overview structure |
| 10-25% | Borderline - overview decides |
| > 25% | Proceed to Phase 1 (Overview) |

### Step 0.2: Quick Overview Check (if <10%)

If decoder percentile < 10%, run a quick overview:

```bash
poetry run python -m splatnlp.mechinterp.cli.overview_cli \
    --feature-id {FEATURE_ID} --model ultra --top-k 10
```

**Signs of clear structure (proceed to Phase 1):**
- One family dominates (>40% of breakdown)
- Strong weapon concentration (>50% one weapon)
- Clear binary ability pattern
- Top PageRank token has score > 0.20

**Signs of no structure (label as weak):**
- Family breakdown is flat (all <15%)
- Weapons are diverse
- Top PageRank score < 0.10
- High sparsity (>99%) with no clear pattern

### Triage Decision

```
Decoder percentile < 10% AND no clear structure in overview?
  │
  Yes → Label as "Weak/Aux Feature {ID}" and STOP
  │
  No → Proceed to Phase 1 (Overview)
```

### Weak Feature Label Format

```json
{
  "dashboard_name": "Weak/Aux Feature {ID}",
  "dashboard_category": "auxiliary",
  "dashboard_notes": "TRIAGE: Decoder weight {X}th percentile, no clear structure in overview. Skipped deep dive.",
  "hypothesis_confidence": 0.0,
  "source": "claude code (triage)"
}
```

### When to Override Triage

Even with low decoder weights, proceed if:
- The feature is part of a cluster you're investigating
- You have external reason to believe it's important
- You're doing exhaustive analysis of a subset

---

## ⚠️ Deep Dive Basics

A proper deep dive requires **experiments**, not just reading overview data. The overview shows correlations; experiments reveal causation.

### Minimum Requirements for a Deep Dive

| Step | What to Do | Why |
|------|------------|-----|
| 1. Overview | Run overview to see correlations | Generate hypotheses |
| 2. 1D Sweeps | Test top 3-5 families with 1D sweeps | Find causal drivers (scaling abilities) |
| 3. Binary Check | For binary abilities (Comeback, Stealth Jump, LDE, Haunt, etc.), check presence rate | Binary abilities show delta=0 in sweeps but may still be characteristic |
| 4. Bottom Tokens | Check suppressors from overview | What the feature AVOIDS is often more informative |
| 5. 2D Heatmaps | Test interactions between primary driver and correlated tokens | Verify if correlations are causal or spurious |

### Binary Abilities Need Special Handling

**Binary abilities** (you have them or you don't) show **delta=0 in 1D sweeps** because there's no scaling. This does NOT mean they're unimportant.

| Binary Abilities |
|------------------|
| Comeback, Stealth Jump, Last-Ditch Effort, Haunt, Ninja Squid, Respawn Punisher, Object Shredder, Drop Roller, Opening Gambit, Tenacity |

**To evaluate binary abilities:**
1. Check PageRank score (correlation strength)
2. Check presence rate: What % of high-activation examples contain it?
3. Compare mean activation WITH vs WITHOUT the binary token
4. Run 2D heatmap: `scaling_ability × binary_ability` to see conditional effect

### Binary Ability Analysis Protocol (CRITICAL)

Binary abilities can have **strong conditional effects** that ONLY show up in 2D analysis. Here's the exact methodology:

**Step 1: Check presence rate enrichment**
```python
from splatnlp.mechinterp.skill_helpers import load_context
import polars as pl

ctx = load_context('ultra')
df = ctx.db.get_all_feature_activations_for_pagerank(FEATURE_ID)

# Find binary token ID
binary_id = None
for tok_id, tok_name in ctx.inv_vocab.items():
    if tok_name == 'comeback':  # or stealth_jump, etc.
        binary_id = tok_id
        break

# Calculate enrichment
threshold = df['activation'].quantile(0.90)  # Top 10%
high_df = df.filter(pl.col('activation') >= threshold)

with_binary_all = df.filter(pl.col('ability_input_tokens').list.contains(binary_id))
with_binary_high = high_df.filter(pl.col('ability_input_tokens').list.contains(binary_id))

baseline_rate = len(with_binary_all) / len(df)
high_rate = len(with_binary_high) / len(high_df)
enrichment = high_rate / baseline_rate

print(f"Baseline presence: {baseline_rate:.1%}")
print(f"High-activation presence: {high_rate:.1%}")
print(f"Enrichment ratio: {enrichment:.2f}x")
# Enrichment > 1.5x suggests binary ability is characteristic
```

**Step 2: Check mean activation WITH vs WITHOUT**
```python
with_binary = df.filter(pl.col('ability_input_tokens').list.contains(binary_id))
without_binary = df.filter(~pl.col('ability_input_tokens').list.contains(binary_id))

mean_with = with_binary['activation'].mean()
mean_without = without_binary['activation'].mean()
delta = mean_with - mean_without

print(f"Mean WITH: {mean_with:.4f}")
print(f"Mean WITHOUT: {mean_without:.4f}")
print(f"Delta: {delta:+.4f}")
# Delta > 0.03 suggests meaningful effect
```

**Step 3: Run 2D heatmap (MOST IMPORTANT)**

Binary abilities can have **conditional effects** that vary by the scaling ability level:

```python
# Manual 2D analysis for binary abilities
# (The built-in 2D heatmap may not handle binary tokens correctly)

scaling_ids = {3: 48, 6: 49, 12: 50, 21: 53, 29: 80}  # ISM example
binary_id = 27  # Comeback

print("Scaling | No Binary | With Binary | Delta")
print("-" * 50)

for level, tok_id in scaling_ids.items():
    level_df = df.filter(pl.col('ability_input_tokens').list.contains(tok_id))

    with_binary = level_df.filter(pl.col('ability_input_tokens').list.contains(binary_id))
    without_binary = level_df.filter(~pl.col('ability_input_tokens').list.contains(binary_id))

    mean_with = with_binary['activation'].mean() if len(with_binary) > 0 else 0
    mean_without = without_binary['activation'].mean() if len(without_binary) > 0 else 0
    delta = mean_with - mean_without

    print(f"{level:>7} | {mean_without:>9.4f} | {mean_with:>11.4f} | {delta:>+.4f}")
```

**Example (Feature 13352):**
```
ISM × Comeback 2D Analysis:
ISM | No CB  | With CB | Delta
  0 | 0.066  | 0.117   | +0.051
  3 | 0.122  | 0.261   | +0.139
  6 | 0.147  | 0.352   | +0.205  ← PEAK INTERACTION
 12 | 0.094  | 0.163   | +0.069
 21 | 0.094  | 0.129   | +0.035

Interpretation: Comeback has STRONG conditional effect at ISM 3-6.
The +0.205 delta at ISM_6 means Comeback DOUBLES the activation!
1D sweep showed delta=0 because most examples have ISM=0 (low baseline).
```

**Step 4: Test combinations of binary abilities together**
```python
# Test multiple binary abilities together
binary_id_1 = 27  # e.g., comeback
binary_id_2 = 1   # e.g., stealth_jump

both = df.filter(
    pl.col('ability_input_tokens').list.contains(binary_id_1) &
    pl.col('ability_input_tokens').list.contains(binary_id_2)
)
neither = df.filter(
    ~pl.col('ability_input_tokens').list.contains(binary_id_1) &
    ~pl.col('ability_input_tokens').list.contains(binary_id_2)
)

# Then do 2D analysis at each scaling level
# Combinations can have stronger effects than individual abilities!
```

**Key Insight:** Binary abilities may have stronger effects when combined. Always test combinations, not just individual tokens.

### Additional Learnings

1. **Conditional effects can be much stronger than marginal effects**: A feature might show ISM with only 0.069 max_delta in 1D sweeps, but a binary ability combination at moderate ISM could produce +0.335 delta - the interaction effect can be 5x stronger than the marginal effect. 1D sweeps can dramatically underestimate a feature's true behavior.

2. **Depletion is informative**: If a binary ability shows enrichment < 1.0 (e.g., 0.72x), the feature actively *avoids* that ability. This is meaningful for interpretation - it tells you what the feature excludes, not just what it includes.

3. **Manual 2D analysis required for binary tokens**: The `Family2DHeatmapRunner` uses `parse_token()` which expects `family_name_AP` format, but binary abilities appear as just the token name (e.g., `comeback` not `comeback_10`). Use manual 2D analysis code for binary abilities (see protocol above).

4. **"Weak feature" needs decoder weight check**: A feature with weak activation effects (max_delta < 0.03) might still have high influence on outputs. Remember: **net influence = activation strength × decoder weight**. Before labeling as "weak", check the feature's decoder weights to the output tokens it contributes to. A "weak activation" feature with high decoder weights may actually be important.

5. **Watch for error-correction features**: If 1D sweeps show small deltas or effects only in unusual rung combinations, the feature may fire when prerequisites are MISSING (OOD detection). Test "explains-away" behavior by comparing activation when low-level evidence is present vs missing. Example: Does feature fire MORE when SCU_3 is absent from a high-SCU build?

6. **Beware of flanderization in top activations**: The top 100 activations over-emphasize extreme cases. The TRUE concept often lives in the **mid-activation range (25-75th percentile)**. Always compare mid vs top activation regions - if they show different weapon/ability patterns, label the mid-range concept and note the extremes as "super-stimuli".

### What Counts as Evidence

| Evidence Type | Strength | Example |
|---------------|----------|---------|
| 1D sweep max_delta > 0.05 | Strong causal | "ISM drives this feature" |
| 1D sweep max_delta 0.02-0.05 | Weak causal | "ISM has minor effect" |
| 1D sweep max_delta < 0.02 | Negligible | "ISM doesn't drive this" |
| Binary delta = 0 | Inconclusive | Need presence rate check |
| High PageRank + low delta | Spurious correlation | Token co-occurs but doesn't cause |
| 2D heatmap shows conditional effect | Interaction confirmed | "X matters only when Y is high" |
| Bottom tokens (suppressors) | Avoidance pattern | "Feature avoids death-perks" |
| Higher activation when prerequisite MISSING | Error-correction | "Fires on OOD rung combos" |
| Mid-range (25-75%) differs from top | Flanderization | "Top is super-stimuli; label mid-range" |

### Common Mistakes to Avoid

1. **Presenting overview as findings** - Overview is hypotheses, not conclusions
2. **Ignoring binary abilities** - Delta=0 doesn't mean unimportant
3. **Skipping bottom tokens** - Suppressors reveal what feature avoids
4. **Only running 1D sweeps** - 2D heatmaps needed for interaction effects
5. **Not checking weapon patterns** - Feature may be weapon-specific, not ability-specific
6. **Using only top activations** - Top 100 may be "flanderized" extremes; check mid-range (25-75%)
7. **Missing error-correction features** - Small deltas in weird rung combos may indicate OOD detection
8. **Confusing data sparsity with suppression** - Zero examples at a condition ≠ "suppression to 0" (see below)
9. **Shallow validation** - Just checking if numbers "look right" without running enrichment analysis
10. **Semantic contradictions in labels** - e.g., "Zombie" (embraces death) + "high SSU" (avoids death) is contradictory
11. **Reporting weapon percentages from top-100** - Use top 20-30% instead; top-100 can be 5-10x off (e.g., 78% vs 10%)
12. **Not checking meta archetypes** - Weapons may cluster by playstyle, not kit; use splatoon3-meta skill
13. **Assuming kit-based patterns** - Check if weapons share sub/special BEFORE assuming it's kit-related
14. **Ignoring flanderization crossover** - Note where a "super-stimulus" weapon overtakes the general pattern (usually top 5%)

### ⚠️ CRITICAL: Data Sparsity vs Suppression

**This is a common and dangerous mistake.** When you see "activation = 0" or "no effect" at some condition, ask: **Is this suppression or data sparsity?**

**Example of the mistake (Feature 1819):**
```
Original claim: "QR is HARD SUPPRESSOR - SSU_57+QR_any=0.000"
Reality: There were ZERO examples with SSU_57 + any QR in the dataset!
         The "0.000" was missing data, not suppression.
```

**How to detect data sparsity:**
```python
# ALWAYS check sample sizes when claiming suppression!
at_high_ssu = df.filter(pl.col('ability_input_tokens').list.contains(ssu_57_id))
with_qr = at_high_ssu.filter(pl.col('ability_input_tokens').list.set_intersection(qr_ids).list.len() > 0)

print(f"Examples at SSU_57 with QR: {len(with_qr)}")  # If 0, this is SPARSITY not suppression!
```

**Rule:** Never claim "suppression" unless you have ≥20 examples in the suppressed condition. Report sample sizes with all claims.

## Philosophy

A **meaningful label** should capture:
- What concept the feature encodes (not just "detects token X")
- Why the model might have learned this representation
- How it relates to strategic/tactical gameplay

**Avoid trivial labels** like:
- "SCU Detector" (just describes token presence)
- "High activation feature" (describes statistics, not meaning)

**Aim for interpretable labels** like:
- "Aggressive Slayer Build" (strategic concept)
- "Special Spam Enabler" (functional role)
- "Backline Support Kit" (playstyle archetype)

## Investigation Workflow

### Phase 0: Triage

See [Phase 0: Triage](#phase-0-triage-always-start-here) above. **Always start here.**

If feature passes triage (decoder weight ≥10% OR has clear structure), proceed to Phase 1.

### Phase 1: Initial Assessment

Run the overview and classify the feature type:

```bash
poetry run python -m splatnlp.mechinterp.cli.overview_cli \
    --feature-id {FEATURE_ID} --model {MODEL} --top-k 20
```

**Classify based on family breakdown:**

| Pattern | Type | Next Steps |
|---------|------|------------|
| One family >40% | Single-family | Check for interference, weapon specificity |
| Top 2-3 families ~20% each | Multi-family | Check synergy/redundancy, build archetype |
| Many families <15% each | Distributed | Look for meta-pattern, weapon class |
| Weapons concentrated | Weapon-specific | Weapon sweep, class analysis |

**CRITICAL**: Always check for non-monotonic effects! Higher AP doesn't always mean higher activation.

### Phase 1.5: Activation Region Analysis (CRITICAL - Anti-Flanderization)

**Don't only examine extreme activations!** High activations may be "flanderized" - exaggerated, extreme versions of the true concept that over-emphasize niche cases.

**Key insight:** The TRUE concept often lives in the **mid-activation range (25-75th percentile)**, not the top 100 examples. Top activations can mislead you into labeling a niche pattern instead of the general concept.

Run activation region analysis:

```python
from splatnlp.mechinterp.skill_helpers import load_context
import numpy as np
from collections import Counter

ctx = load_context("{MODEL}")
df = ctx.db.get_all_feature_activations_for_pagerank({FEATURE_ID})

acts = df['activation'].to_numpy()
weapons = df['weapon_id'].to_list()

# Define activation regions
regions = [
    ('Floor (≤0.01)', lambda a: a <= 0.01),
    ('Low (0.01-0.05)', lambda a: 0.01 < a <= 0.05),
    ('Mild (0.05-0.10)', lambda a: 0.05 < a <= 0.10),
    ('Moderate (0.10-0.20)', lambda a: 0.10 < a <= 0.20),
    ('High (0.20-0.35)', lambda a: 0.20 < a <= 0.35),
    ('Very High (>0.35)', lambda a: a > 0.35),
]

for region_name, filter_fn in regions:
    indices = [i for i, a in enumerate(acts) if filter_fn(a)]
    weps = [weapons[i] for i in indices]
    print(f"\n{region_name} (n={len(indices)}):")
    for wep, count in Counter(weps).most_common(5):
        name = ctx.id_to_weapon_display_name(wep)
        print(f"  {name}: {count}")
```

**Key signals to look for:**

| Pattern | Interpretation |
|---------|----------------|
| Same weapons in ALL regions | General concept (continuous feature) |
| Different weapons in moderate vs high | Super-stimuli detected |
| Diverse weapons in moderate, concentrated in high | True concept is in moderate region |
| Niche weapons only in high | High activations are "flanderized" extremes |

**Example (Feature 9971):**
```
Moderate (0.10-0.20): Splattershot (115), Wellstring (65), Sploosh (57)...
Very High (>0.35): Bloblobber (44), Glooga Deco (39), Range Blaster (28)

Interpretation: Low-moderate shows GENERAL offensive investment.
Very high shows EXTREME SCU on special-dependent weapons (super-stimuli).
Label the general concept, note the super-stimuli pattern.
```

**CRITICAL**: Always check the **Bottom Tokens (Suppressors)** section! Tokens that rarely appear in high-activation examples can reveal what the feature *avoids*:

| Suppressor Pattern | Interpretation |
|-------------------|----------------|
| Death-mitigation (QR, SS, CB) suppressed | Feature avoids "death-accepting" builds |
| Defensive (IR, SR) suppressed | Feature prefers aggressive/ranged builds |
| Mobility suppressed | Feature prefers stationary/positional play |
| Special abilities suppressed | Feature encodes non-special playstyle |

**Example**: If SCU is enhanced but `quick_respawn`, `special_saver`, and `comeback` are ALL suppressed, the feature doesn't just detect "SCU" - it detects "death-averse SCU builds" (players who stack SCU but don't plan to die).

### Phase 1.6: Weapon Distribution Analysis (CRITICAL - Anti-Flanderization)

**NEVER report weapon percentages from top-100 samples.** Top-100 is severely flanderized and can give wildly misleading weapon distributions.

**Example (Feature 14096 - Real Case):**
```
Top 100:     Dark Tetra 78%, Stamper 20%  ← WRONG, flanderized
Top 10%:     Stamper 35%, Dark Tetra 21%  ← Better but still skewed
Top 30%:     Stamper 23%, Dark Tetra 10%  ← TRUE CONCEPT
Full dataset: Stamper 9%, Dark Tetra 3.5% ← Includes noise/floor
```

**Use top 20-30% for weapon characterization:**

```python
import polars as pl
import numpy as np
from collections import Counter
from splatnlp.mechinterp.skill_helpers import load_context

ctx = load_context('ultra')
df = ctx.db.get_all_feature_activations_for_pagerank(FEATURE_ID)

# Get percentile thresholds
acts = df['activation'].to_numpy()
thresholds = {p: np.percentile(acts, p) for p in [0, 50, 70, 80, 90, 95, 99]}

# Analyze by region
regions = [
    ("Bottom 50% (noise)", 0, 50),
    ("50-70% (weak)", 50, 70),
    ("Top 30% (TRUE CONCEPT)", 70, 100),
    ("Top 10%", 90, 100),
    ("Top 1% (flanderized)", 99, 100),
]

print("Region | Top Weapons")
print("-" * 60)

for name, p_low, p_high in regions:
    t_low, t_high = thresholds[p_low], thresholds.get(p_high, float('inf'))
    if p_high == 100:
        region_df = df.filter(pl.col('activation') >= t_low)
    else:
        region_df = df.filter((pl.col('activation') >= t_low) & (pl.col('activation') < t_high))

    if len(region_df) == 0:
        continue

    weapon_counts = region_df.group_by('weapon_id').agg(
        pl.col('activation').count().alias('n')
    ).sort('n', descending=True)

    top3 = []
    for row in weapon_counts.head(3).iter_rows(named=True):
        wname = ctx.id_to_weapon_display_name(row['weapon_id'])
        pct = row['n'] / len(region_df) * 100
        top3.append(f"{wname[:12]}({pct:.0f}%)")

    print(f"{name:<25} | {', '.join(top3)}")
```

**Interpretation Guide:**

| Pattern | Meaning |
|---------|---------|
| Same weapons in top-30% and top-1% | Continuous feature, no flanderization |
| Different weapons in top-30% vs top-1% | **Flanderization detected** - label top-30% concept |
| One weapon jumps from 10% to 70%+ | That weapon is "super-stimulus" for the feature |
| Weapons consistent 50%→30%→10%→1% | Stable feature, safe to use any region |

**Rule: Report weapon percentages from top 20-30%, note if top-1% differs significantly.**

### Phase 1.7: Meta-Informed Weapon Analysis (USE AFTER WEAPON SWEEP)

After identifying top weapons, **always check if they match a known meta archetype** using the `splatoon3-meta` skill.

**Step 1: Look up weapon kits**

Check `references/weapons.md` for each top weapon's sub and special:

```python
# Top weapons from Feature 14096 (top 30%):
kits = {
    "Splatana Stamper": ("Burst Bomb", "Zipcaster"),
    "Dark Tetra Dualies": ("Autobomb", "Reefslider"),
    "Glooga Dualies": ("Splash Wall", "Booyah Bomb"),
    "Dapple Dualies Nouveau": ("Torpedo", "Reefslider"),
    "Splatana Wiper": ("Torpedo", "Ultra Stamp"),
}

# Check for shared subs/specials
from collections import Counter
subs = Counter(k[0] for k in kits.values())
specials = Counter(k[1] for k in kits.values())

# If one sub/special dominates → kit-based feature
# If diverse → playstyle-based feature
```

**Step 2: Check archetype reference**

Read `references/archetypes.md` to see if weapons match a known archetype:

| Archetype | Key Weapons | Signature Abilities |
|-----------|-------------|---------------------|
| Zombie Slayer | Tetra Dualies, Splatana Wiper | QR + Comeback + Stealth Jump |
| Stealth Slayer | Carbon Roller, Inkbrush | Ninja Squid + SSU + Stealth Jump |
| Anchor/Backline | E-liter, Hydra Splatling | Respawn Punisher + Object Shredder |
| Support/Beacon | Squid Beakon weapons | Sub Power Up + ISS + Comeback |

**Step 3: Classification decision**

```
Kit Analysis Result:
├─ Shared sub weapon? → Feature may encode SUB PLAYSTYLE
├─ Shared special? → Feature may encode SPECIAL FARMING
├─ No kit pattern + archetype match? → PLAYSTYLE FEATURE (label as archetype)
└─ No kit pattern + no archetype? → WEAPON CLASS feature (check if all dualies, all shooters, etc.)
```

**Example (Feature 14096):**
```
Top 30% weapons: Stamper, Dark Tetra, Glooga, Dapple, Wiper
Kit analysis: Diverse subs (Burst, Auto, Splash Wall, Torpedo), diverse specials
Archetype check: Dark Tetra + Splatana Wiper = "Zombie Slayer" archetype!
Conclusion: PLAYSTYLE feature encoding Zombie Slayer (death-accepting aggressive)
Label: "Zombie Slayer QR (Splatana/Dualies)" - tactical category
```

**When to invoke splatoon3-meta skill:**
- After weapon_sweep shows concentrated weapon pattern
- When top weapons seem unrelated by kit but share a playstyle
- To validate that ability patterns match expected meta builds
- To identify if weapons share archetype despite different kits

### Phase 2: Hypothesis Generation

Based on Phase 1, generate hypotheses about what the feature might encode:

**For single-family dominated features:**
- H1: Pure token detector (trivial - try to disprove)
- H2: Threshold detector (activates only at high AP)
- H3: Interaction detector (family + something else)
- H4: Weapon-conditional (family matters only for certain weapons)

**For multi-family features:**
- H1: Synergy detector (families work together)
- H2: Build archetype (strategic loadout pattern)
- H3: Playstyle indicator (aggressive, defensive, support)

**For weapon-specific features:**
- H1: Weapon class pattern (all shooters, all chargers, etc.)
- H2: Meta build (optimal loadout for that weapon)
- H3: Weapon-ability interaction

### Phase 3: Targeted Experiments

Run experiments to test hypotheses. **Available experiment types:**

| Type | Purpose |
|------|---------|
| `family_1d_sweep` | Activation across AP rungs for one family |
| `family_2d_heatmap` | Interaction between two families |
| `within_family_interference` | Detect error correction within a family |
| `weapon_sweep` | Activation by weapon (optionally conditioned on family) |
| `weapon_group_analysis` | Compare high vs low activation by weapon |
| `pairwise_interactions` | Synergy/redundancy between tokens |
| `token_influence_sweep` | Identify enhancers and suppressors across all tokens |

## ⚠️ CRITICAL: Iterative Conditional Testing Protocol

**1D sweeps can be MISLEADING for secondary abilities.** When a feature has a strong primary driver:

### The Problem

1D sweep for secondary ability (e.g., QR) across ALL contexts might show **delta ≈ 0**

**Why this happens:**
- Most contexts have LOW primary driver (e.g., low SCU) → activation already near zero
- Secondary ability can't suppress what's already zero
- The few high-primary contexts get drowned out in the average

**Example (Feature 18712):**
```
QR 1D sweep (all contexts): mean_delta = -0.0006 → "QR has no effect" ❌ WRONG!
SCU × QR 2D heatmap:
  - At SCU_15: QR_0=0.13, QR_12=0.04 → QR suppresses 70%! ✅
  - At SCU_29: QR_0=0.15, QR_12=0.04 → QR suppresses 74%! ✅
```

### The Solution: Iterative 2D Testing

**Protocol for features with a strong primary driver:**

```
1. Confirm primary driver with 1D sweep
   └─ If monotonic response confirmed → proceed to step 2

2. For EACH correlated ability in overview (top 5-10):
   └─ Run 2D heatmap: PRIMARY × SECONDARY
   └─ Check activation at EACH primary level
   └─ Look for:
      - Suppression: secondary reduces activation at high primary
      - Synergy: secondary boosts activation at high primary
      - Spurious: no conditional effect (correlation was coincidence)

3. Group findings by semantic category:
   └─ Death-mitigation (QR, SS, CB): all suppress? → "death-averse"
   └─ Mobility (SSU, RSU): all enhance? → "mobility-synergistic"
   └─ Efficiency (ISM, ISS): mixed? → test individually
```

### 2D Heatmap Interpretation Guide

| Pattern | Interpretation |
|---------|----------------|
| Peak at (high_X, 0_Y) | Y is a **suppressor** |
| Peak at (high_X, high_Y) | Y is a **synergy** |
| Flat across Y at each X | Y has **no conditional effect** (spurious) |
| Non-monotonic in X at some Y | **Interference** pattern |

### When to Use 2D vs 1D

| Scenario | Use 1D | Use 2D |
|----------|--------|--------|
| Testing primary driver | ✅ | - |
| Testing secondary abilities | ❌ MISLEADING | ✅ REQUIRED |
| Looking for interactions | - | ✅ |
| Confirming suppressor hypothesis | - | ✅ |
| Quick initial scan | ✅ (with caution) | - |

### Template: Death-Aversion Test Battery

For single-family dominated features, always test death-mitigation:

```json
// Test 1: Primary × Quick Respawn
{
  "type": "family_2d_heatmap",
  "variables": {
    "family_x": "{PRIMARY}",
    "family_y": "quick_respawn",
    "rungs_x": [0, 6, 15, 29, 41, 57],
    "rungs_y": [0, 6, 12, 21, 29]
  }
}

// Test 2: Primary × Special Saver
{
  "type": "family_2d_heatmap",
  "variables": {
    "family_x": "{PRIMARY}",
    "family_y": "special_saver",
    "rungs_x": [0, 6, 15, 29, 41, 57],
    "rungs_y": [0, 3, 6, 12, 21]
  }
}

// Test 3: Primary × Comeback (exclusive)
{
  "type": "family_2d_heatmap",
  "variables": {
    "family_x": "{PRIMARY}",
    "family_y": "comeback",
    "rungs_x": [0, 6, 15, 29, 41, 57],
    "rungs_y": [0, 10]
  }
}
```

If ALL three show suppression at Y>0, label includes "death-averse"

### Template: Error-Correction Detection

If 1D sweeps show **small deltas** or effects **only in unusual rung combinations**, test for error-correction behavior:

```python
import polars as pl
from splatnlp.mechinterp.skill_helpers import load_context

ctx = load_context('ultra')
df = ctx.db.get_all_feature_activations_for_pagerank(FEATURE_ID)

# Get token IDs for high and low rungs
# Example: SCU_57 (high) and SCU_3 (low)
high_rung_id = ctx.vocab['special_charge_up_57']
low_rung_id = ctx.vocab['special_charge_up_3']

# Compare activation when low rung is present vs missing (among high-rung builds)
high_with_low = df.filter(
    pl.col('ability_input_tokens').list.contains(high_rung_id) &
    pl.col('ability_input_tokens').list.contains(low_rung_id)
)
high_without_low = df.filter(
    pl.col('ability_input_tokens').list.contains(high_rung_id) &
    ~pl.col('ability_input_tokens').list.contains(low_rung_id)
)

mean_with = high_with_low['activation'].mean()
mean_without = high_without_low['activation'].mean()

print(f"High rung WITH low rung present: {mean_with:.4f} (n={len(high_with_low)})")
print(f"High rung WITHOUT low rung: {mean_without:.4f} (n={len(high_without_low)})")
print(f"Delta: {mean_without - mean_with:+.4f}")

# If WITHOUT > WITH, feature fires when prerequisite is MISSING = error correction!
```

**Signs of error-correction:**

| Pattern | Interpretation | Label Style |
|---------|----------------|-------------|
| Higher activation when low rung MISSING | "Explains away" missing evidence | "Error-Correction: {FAMILY}" |
| Only fires on weird rung combos | OOD detector | "OOD Detector: {PATTERN}" |
| Negative interactions in 2D heatmaps | Within-family interference | "Interference Feature: {FAMILY}" |

**Test for within-family interference (CRITICAL for single-family):**
```json
{
  "type": "within_family_interference",
  "feature_id": {FEATURE_ID},
  "model_type": "{MODEL}",
  "variables": {"family": "{FAMILY}"},
  "description": "Test for error correction within {FAMILY}"
}
```

**Test for interactions (2D heatmap):**
```json
{
  "type": "family_2d_heatmap",
  "feature_id": {FEATURE_ID},
  "model_type": "{MODEL}",
  "variables": {
    "family_x": "{FAMILY_A}",
    "family_y": "{FAMILY_B}"
  }
}
```

**Test for weapon specificity:**
```json
{
  "type": "weapon_sweep",
  "feature_id": {FEATURE_ID},
  "model_type": "{MODEL}",
  "variables": {"min_examples": 10, "top_k_weapons": 20}
}
```

**CHECKPOINT: After weapon_sweep, check for dominant weapon pattern:**

If weapon_sweep diagnostics show "DOMINANT WEAPON" warning (one weapon has >2x delta of second):

1. **Run kit_sweep** to analyze by sub weapon and special weapon:
```json
{
  "type": "kit_sweep",
  "feature_id": {FEATURE_ID},
  "model_type": "{MODEL}",
  "variables": {"min_examples": 10, "top_k": 10, "analyze_combinations": true}
}
```

2. **Use splatoon3-meta skill** to look up the dominant weapon's kit:
   - Read `.claude/skills/splatoon3-meta/references/weapons.md`
   - Find the weapon's sub weapon and special weapon

3. **Cross-reference** other high-activation weapons:
   - Do they share the same sub weapon?
   - Do they share the same special weapon?
   - If yes, the feature may encode **kit behavior** not weapon behavior

4. **Update hypothesis** based on findings:
   - If shared sub: Feature may encode sub weapon playstyle
   - If shared special: Feature may encode special spam/farming
   - If no kit pattern: Feature is truly weapon-specific

**Example**: Feature 18712 shows Octobrush Nouveau dominant. Kit lookup reveals Squid Beakon + Ink Storm. Other high weapons (Rapid Blaster, Range Blaster) also have "special-dependent" characteristics per meta → Feature encodes "SCU for Ink Storm spam" not just "Octobrush".

**Test for threshold effects:**
- Compare low-rung vs high-rung responses
- Look for non-linear jumps in activation
- Check if certain rungs REDUCE activation (interference)

### Phase 4: Synthesis

Combine findings into a coherent interpretation:

1. **What triggers activation?** (tokens, combinations, weapons)
2. **Is there structure beyond simple detection?** (interactions, thresholds)
3. **What gameplay concept does this represent?**
4. **Why would the model learn this?** (predictive value for recommendations)

### Phase 5: Label Proposal

Propose a label at the appropriate level:

| Complexity | Label Type | Example |
|------------|------------|---------|
| Trivial | Token detector | "SCU Presence" (avoid if possible) |
| Simple | Threshold detector | "High SCU Investment (29+ AP)" |
| Moderate | Interaction | "SCU + Mobility Combo" |
| Strategic | Build archetype | "Special Spam Slayer Kit" |
| Tactical | Playstyle | "Aggressive Frontline Build" |

### Phase 6: Deeper Dive (For Thorny Features)

**When to use:** If the standard deep dive (Phases 1-5) didn't produce a clear interpretation:
- All scaling effects weak (max_delta < 0.03)
- No clear primary driver
- Conflicting signals from different experiments
- Feature seems important (high contribution to outputs) but unclear why

**The Deeper Dive uses the hypothesis/state management system** for systematic exploration:

#### Step 1: Initialize Research State

```python
from splatnlp.mechinterp.state import ResearchState, Hypothesis

state = ResearchState(feature_id=FEATURE_ID, model_type="ultra")

# Add competing hypotheses based on what you've observed
state.add_hypothesis(Hypothesis(
    id="h1",
    description="Feature encodes weapon-specific pattern for Dapple Nouveau",
    status="pending"
))
state.add_hypothesis(Hypothesis(
    id="h2",
    description="Feature encodes binary ability package (Stealth + Comeback)",
    status="pending"
))
state.add_hypothesis(Hypothesis(
    id="h3",
    description="Feature has high decoder weights despite weak activation effects",
    status="pending"
))
```

#### Step 2: Check Decoder Weights

For "weak activation" features, check if they have high influence via decoder weights:

```python
# Load SAE decoder weights
import torch
sae_path = '/mnt/e/dev_spillover/SplatNLP/sae_runs/run_20250704_191557/sae_model_final.pth'
sae_checkpoint = torch.load(sae_path, map_location='cpu', weights_only=True)
decoder_weight = sae_checkpoint['decoder.weight']  # [512, 24576]

# Get this feature's decoder weights to output space
feature_decoder = decoder_weight[:, FEATURE_ID]  # [512]

# Check magnitude
print(f"Decoder weight L2 norm: {torch.norm(feature_decoder):.4f}")
print(f"Max absolute weight: {torch.abs(feature_decoder).max():.4f}")

# Compare to other features
all_norms = torch.norm(decoder_weight, dim=0)
percentile = (all_norms < torch.norm(feature_decoder)).float().mean() * 100
print(f"Percentile among all features: {percentile:.1f}%")
```

If decoder weights are high (>75th percentile), the feature may be important despite weak activation effects.

#### Step 3: Test Output Token Connections

Check which output tokens this feature most influences:

```python
# Get the model's output layer weights
# Feature activations → pooled repr → output logits
# Need to trace: SAE_feature → decoder → output_layer

from splatnlp.mechinterp.skill_helpers import load_context
ctx = load_context('ultra')

# The feature's contribution to each output token
# This requires tracing through the model architecture
# (Implementation depends on how outputs are structured)
```

#### Step 4: Run Targeted Experiments

Based on hypotheses, run specific tests:

```python
# Log experiments and findings to state
state.add_evidence(
    hypothesis_id="h1",
    experiment_type="weapon_sweep",
    finding="37% Dapple Nouveau, but also 10% .96 Gal Deco - not single-weapon",
    supports=False
)

state.add_evidence(
    hypothesis_id="h3",
    experiment_type="decoder_weight_check",
    finding="Decoder L2 norm: 0.89 (92nd percentile) - HIGH despite weak activation",
    supports=True
)
```

#### Step 5: Synthesize

```python
# Review all evidence
state.summarize()

# Update hypothesis statuses
state.update_hypothesis("h1", status="rejected")
state.update_hypothesis("h3", status="supported")

# Propose final interpretation
state.set_conclusion(
    "Feature has weak activation effects but high decoder weights. "
    "It acts as a 'fine-tuning' feature that makes small but important "
    "adjustments to output probabilities."
)
```

#### When Deeper Dive is Complete

The state object provides an audit trail of:
- What hypotheses were considered
- What experiments were run
- What evidence was found
- Why the final interpretation was chosen

This is useful for:
- Revisiting the feature later
- Explaining the interpretation to others
- Identifying if new evidence should change the interpretation

## Decision Trees

### Single-Family Dominated Feature

```
1. Run within_family_interference to check for error correction
   └─ If interference found → "Error-Correcting {FAMILY} Detector"
   └─ If enhancement patterns → "{FAMILY} Stacker (synergistic)"
   └─ If neutral → continue

2. Check for non-monotonic 1D response
   └─ If drops at certain rungs → investigate interference
   └─ If monotonic with threshold → "High {FAMILY} Investment"
   └─ If monotonic with no threshold → probably trivial

3. Run weapon_sweep to check weapon specificity
   └─ If weapon-concentrated → run weapon_group_analysis
   └─ If weapon-specific patterns → "{WEAPON_CLASS} + {FAMILY}"

4. Run 2D sweep with second-ranked family
   └─ If interaction effect → "{FAMILY_A} + {FAMILY_B} Combo"
   └─ If no interaction → try third family

5. If all trivial → label as "{FAMILY} Stacker" with note "simple detector"
```

### Multi-Family Feature

```
1. Check if families are related
   └─ All mobility (SSU, RSU, QSJ) → "Mobility Kit"
   └─ All ink efficiency (ISM, ISS, IRU) → "Efficiency Kit"
   └─ Mixed → continue

2. Run pairwise interaction analysis
   └─ Positive synergy → "Synergistic Build"
   └─ Redundancy → "Alternative Paths"

3. Check weapon breakdown
   └─ Weapon class pattern → "{CLASS} Optimal Build"

4. Consider strategic meaning
   └─ What playstyle does this combination enable?
```

## Example Investigation

**Feature 18712 (Deep Analysis):**

1. **Overview**: SCU 31%, SSU 11%, ISS 10% → Single-family dominated
2. **Hypothesis**: Could be SCU + something, or just trivial SCU detector
3. **2D Heatmap (SCU × SSU)**: Peak at SCU=57, SSU=0. Non-monotonic drops visible!
   - SCU 6→12: DROP of 0.02 (unexpected)
   - SCU 15→21: DROP of 0.01
4. **Interference Analysis**:
   - SCU_12 REDUCES SCU_51 signal by 0.10 (interference!)
   - SCU_15 ENHANCES SCU_51 signal by 0.12 (synergy!)
5. **Weapon Analysis**: Effect varies by weapon
   - weapon_id_50: SCU_3 reduces SCU_15 (-0.08)
   - weapon_id_7020: SCU_3 enhances SCU_15 (+0.03)
6. **Interpretation**: Feature detects "clean" high-SCU builds.
   - Low rungs (SCU_3, SCU_12) can contaminate the signal
   - Effect is weapon-dependent
7. **Label**: "SCU Purity Detector (weapon-conditional)" - NOT trivial!

**Key Insight**: What looked like a simple "SCU detector" actually encodes
complex error-correction behavior. Always check for interference!

## Commands Summary

```bash
# Phase 1: Overview
poetry run python -m splatnlp.mechinterp.cli.overview_cli \
    --feature-id {ID} --model ultra --top-k 20

# Phase 3a: 1D sweep for dominant family
poetry run python -m splatnlp.mechinterp.cli.planner_cli \
    --feature-id {ID} --hypothesis "Responds to {FAMILY}" --model ultra

# Phase 3b: 2D sweep for interactions
# (manually create spec with family_2d_heatmap type)

# Phase 3c: Weapon sweep
poetry run python -m splatnlp.mechinterp.cli.planner_cli \
    --feature-id {ID} --hypothesis "Weapon-specific" --model ultra

# Phase 5: Set label
poetry run python -m splatnlp.mechinterp.cli.labeler_cli label \
    --feature-id {ID} --name "{LABEL}" --category {tactical|strategic|mechanical}
```

## Labeling Categories

- **mechanical**: Low-level patterns (token presence, simple combinations)
- **tactical**: Mid-level patterns (build synergies, weapon kits)
- **strategic**: High-level patterns (playstyles, meta concepts)

## See Also

- **mechinterp-overview**: Initial feature assessment (now includes bottom tokens)
- **mechinterp-runner**: Execute experiments
- **mechinterp-next-step-planner**: Generate experiment specs
- **mechinterp-labeler**: Save labels
- **mechinterp-glossary-and-constraints**: Domain reference
- **mechinterp-ability-semantics**: Ability semantic groupings (check AFTER hypotheses)
- **splatoon3-meta**: Weapon archetypes, kit lookups, meta knowledge (USE for weapon pattern interpretation)

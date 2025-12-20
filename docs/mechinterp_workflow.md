# MechInterp Feature Investigation Workflow

A detailed guide for investigating Sparse Autoencoder (SAE) features using Claude Code to produce meaningful, interpretable labels.

## Table of Contents

1. [Introduction](#introduction)
2. [Background Concepts](#background-concepts)
3. [Tools and Setup](#tools-and-setup)
4. [Tier 0: Triage](#tier-0-triage)
5. [Tier 1: Overview](#tier-1-overview)
6. [Tier 2: Deep Dive](#tier-2-deep-dive)
7. [Tier 3: Deeper Dive](#tier-3-deeper-dive)
8. [Advanced: Beam Trace Net Contribution Analysis](#advanced-beam-trace-net-contribution-analysis)
9. [Quick Reference](#quick-reference)
10. [Common Pitfalls](#common-pitfalls)
11. [Worked Example](#worked-example)

---

## Introduction

### What is this workflow for?

This workflow helps you understand what individual features in a Sparse Autoencoder (SAE) represent. The SAE was trained on SplatNLP's gear recommendation model (SplatGPT) to decompose its internal representations into interpretable components.

Each SAE feature is a direction in the model's activation space. When we say a feature "fires" or "activates," we mean the model's internal state has a component in that direction. Our goal is to figure out **what concept each feature represents** - what patterns in the input cause it to activate?

### Why use Claude Code?

This workflow is designed to be executed with **Claude Code**, Anthropic's AI coding assistant. Claude Code can:

- Run Python experiments and analyze results
- Search through activation databases
- Track hypotheses and evidence systematically
- Remember context across a long investigation
- Use specialized skills for domain knowledge (Splatoon 3 meta, ability semantics)

The workflow uses Claude Code's **skills system** - pre-written instructions that Claude Code can invoke for specific tasks like running experiments, looking up game data, or managing research state.

### The Four-Tier Approach

Not all features require the same level of investigation:

| Tier | Name | Time | When to Use |
|------|------|------|-------------|
| 0 | Triage | 1-2 min | **Always start here** - filter out weak features |
| 1 | Overview | 5 min | Features that pass triage |
| 2 | Deep Dive | 15-30 min | Most features |
| 3 | Deeper Dive | 30-60 min | Thorny/unclear features |

**Tier 0 is critical for efficiency.** Many SAE features have minimal influence on model outputs. Triage identifies these "weak/auxiliary" features early so you can skip expensive analysis and move on.

Most features that pass triage can be labeled after Tier 2. Reserve Tier 3 for features where the standard approach doesn't yield clear answers.

---

## Background Concepts

### What is a Sparse Autoencoder (SAE)?

An SAE is a neural network that learns to compress and reconstruct another model's activations. The key property is **sparsity** - for any given input, only a small number of SAE features activate. This encourages each feature to represent a distinct, interpretable concept.

```
Input → Model → [Internal Activations] → SAE Encoder → [Sparse Features] → SAE Decoder → [Reconstructed Activations]
```

In our case:
- **Model**: SplatGPT (gear recommendation model)
- **Internal Activations**: 512-dimensional pooled representation
- **SAE Features**: 24,576 sparse features (Ultra model) or 2,048 (Full model)

### What is a "feature"?

A feature is a learned direction in the model's activation space. Each feature has:

1. **Encoder weights**: How to detect this feature from model activations
2. **Decoder weights**: How this feature contributes back to model outputs
3. **Activation pattern**: Which inputs cause this feature to fire

Our goal is to understand #3 - what inputs activate this feature?

### Key Terminology

| Term | Definition |
|------|------------|
| **Activation** | How strongly a feature fires for a given input (0 = not at all, higher = stronger) |
| **Sparsity** | Percentage of inputs where feature activation is ~0 |
| **PageRank** | Algorithm to find important tokens; tokens that appear in high-activation examples get high scores |
| **1D Sweep** | Experiment varying one ability family across AP levels |
| **2D Heatmap** | Experiment varying two ability families to find interactions |
| **Scaling ability** | Ability that can have different AP levels (3, 6, 12, 21, etc.) |
| **Binary ability** | Ability that's either present or absent, no levels (Comeback, Stealth Jump, etc.) |

### Splatoon 3 Domain Knowledge

This workflow analyzes gear builds in Splatoon 3. Key concepts:

**Ability Points (AP):**
- Main ability slot = 10 AP
- Sub ability slot = 3 AP
- Maximum = 57 AP per ability (3 mains + 9 subs)

**Ability Types:**
- **Stackable abilities**: Can invest 3-57 AP (e.g., Swim Speed Up, Special Charge Up)
- **Binary abilities**: Fixed 10 AP, either have it or don't (e.g., Comeback, Stealth Jump, Ninja Squid)

**Common Abbreviations:**
| Code | Full Name |
|------|-----------|
| SCU | Special Charge Up |
| SPU | Special Power Up |
| SSU | Swim Speed Up |
| RSU | Run Speed Up |
| ISM | Ink Saver Main |
| ISS | Ink Saver Sub |
| IRU | Ink Recovery Up |
| QR | Quick Respawn |
| QSJ | Quick Super Jump |
| IA | Intensify Action |
| SS | Special Saver |
| CB | Comeback |
| SJ | Stealth Jump |
| LDE | Last-Ditch Effort |
| RES | Ink Resistance Up |

---

## Tools and Setup

### Prerequisites

1. **Claude Code** installed and configured
2. Access to the SplatNLP repository
3. Activation database at `/mnt/e/activations_ultra_efficient`
4. SAE checkpoint at `/mnt/e/dev_spillover/SplatNLP/sae_runs/run_20250704_191557/`

### Available Skills

Claude Code has access to specialized skills for this workflow:

| Skill | Purpose |
|-------|---------|
| `mechinterp-overview` | Quick feature summary |
| `mechinterp-runner` | Execute experiments |
| `mechinterp-glossary-and-constraints` | Domain rules and terminology |
| `mechinterp-labeler` | Save labels |
| `mechinterp-state` | Track hypotheses and evidence |
| `splatoon3-meta` | Game meta knowledge |

### Starting an Investigation

Tell Claude Code which feature you want to investigate:

```
Investigate feature 13352 in the Ultra SAE model.
```

Claude Code will automatically use the appropriate skills and tools.

---

## Tier 0: Triage

**Goal:** Quickly identify weak/auxiliary features that don't warrant deep investigation.

**Time:** 1-2 minutes

**Output:** Either "Weak/Aux Feature [ID]" label OR proceed to Tier 1

### Why Triage?

The Ultra SAE has 24,576 features. Many of these have minimal influence on model outputs - they fire weakly and contribute little to predictions. Spending 15-30 minutes on a deep dive for these features is wasteful.

Tier 0 checks two things:
1. **Decoder weight magnitude** - Does this feature strongly influence outputs?
2. **Overview structure** - Does it show clear patterns worth investigating?

If both are weak, label as "Weak/Aux Feature [ID]" and move on.

### Step 0.1: Check Decoder Weight Percentile

The decoder weights determine how much a feature influences model outputs. Low decoder weights = low importance.

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
| > 25% | Proceed to Tier 1 |

### Step 0.2: Quick Overview Check

If decoder percentile < 10%, run a quick overview to check for structure:

```bash
poetry run python -m splatnlp.mechinterp.cli.overview_cli \
    --feature-id {FEATURE_ID} --model ultra --top-k 10
```

**Signs of clear structure (proceed to Tier 1):**
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
┌─────────────────────────────────┐
│ Check decoder weight percentile │
└─────────────────┬───────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
   < 10%                  ≥ 10%
        │                   │
        ▼                   ▼
┌───────────────┐    ┌─────────────┐
│ Quick overview │    │ Proceed to  │
│ structure check│    │   Tier 1    │
└───────┬───────┘    └─────────────┘
        │
   ┌────┴────┐
   │         │
No structure  Has structure
   │         │
   ▼         ▼
┌──────────────────┐  ┌─────────────┐
│ Label:           │  │ Proceed to  │
│ "Weak/Aux [ID]"  │  │   Tier 1    │
└──────────────────┘  └─────────────┘
```

### Weak Feature Label Format

For features that fail triage:

```json
{
  "feature_id": 12345,
  "dashboard_name": "Weak/Aux Feature 12345",
  "dashboard_category": "auxiliary",
  "dashboard_notes": "TRIAGE: Decoder weight 3.2nd percentile, no clear structure in overview. Skipped deep dive.",
  "hypothesis_confidence": 0.0,
  "source": "claude code (triage)"
}
```

This label:
- Clearly marks it as weak/auxiliary
- Records why it was skipped
- Can be revisited later if needed

### When to Override Triage

Even with low decoder weights, proceed to Tier 1 if:
- The feature is part of a cluster you're investigating
- You have external reason to believe it's important
- You're doing exhaustive analysis of a subset

---

## Tier 1: Overview

**Goal:** Get a quick snapshot of what correlates with high feature activation.

**Time:** ~5 minutes

**Output:** Hypotheses to test in Tier 2

### What the Overview Shows

The overview computes several things:

1. **Activation Statistics**: Mean, median, sparsity
2. **Top Tokens (PageRank)**: Abilities that appear frequently in high-activation examples
3. **Family Breakdown**: Aggregated by ability family
4. **Top Weapons**: Which weapons appear in high-activation examples
5. **Bottom Tokens**: Abilities that are RARE in high-activation examples (suppressors)
6. **Sample Contexts**: Example builds with high activation

### Running the Overview

Claude Code runs:

```bash
poetry run python -m splatnlp.mechinterp.cli.overview_cli \
    --feature-id {FEATURE_ID} \
    --model ultra \
    --top-k 20
```

### ⚠️ CRITICAL: Beware of Flanderization

**Top activations can be misleading!** The highest-activation examples may be "flanderized" - exaggerated, extreme versions of the true concept that don't represent the feature's core meaning.

**The problem:**
- Top 100 activations over-emphasize extreme cases
- These extremes may be niche weapons or rare builds
- The TRUE concept often lives in the **mid-activation range (25th-75th percentile)**

**What to do:**

1. **Don't rely solely on top activations** - Check if the mid-activation slice (25-75%) tells a different story

2. **Compare activation regions:**

```python
import numpy as np

acts = df['activation'].to_numpy()
weapons = df['weapon_id'].to_list()

# Define regions
p25, p75, p90 = np.percentile(acts[acts > 0], [25, 75, 90])

mid_mask = (acts >= p25) & (acts <= p75)
top_mask = acts >= p90

# Compare weapon distributions
from collections import Counter
mid_weapons = Counter([weapons[i] for i, m in enumerate(mid_mask) if m])
top_weapons = Counter([weapons[i] for i, m in enumerate(top_mask) if m])

print("Mid-activation (25-75%) top weapons:")
for wep, count in mid_weapons.most_common(5):
    print(f"  {ctx.id_to_weapon_display_name(wep)}: {count}")

print("\nTop-activation (>90%) top weapons:")
for wep, count in top_weapons.most_common(5):
    print(f"  {ctx.id_to_weapon_display_name(wep)}: {count}")
```

3. **Signs of flanderization:**

| Pattern | What It Means |
|---------|---------------|
| Different weapons in mid vs top | Top is flanderized; true concept in mid |
| Same weapons across all regions | Feature is consistent (no flanderization) |
| Niche weapons only in top | Extreme cases are "super-stimuli" |
| Diverse weapons in mid, concentrated in top | Label the mid-range concept |

**Example (Feature 9971):**
```
Mid (25-75%): Splattershot (115), Wellstring (65), Sploosh (57)  ← DIVERSE
Top (>90%):   Bloblobber (44), Glooga Deco (39), Range Blaster (28)  ← NICHE

Interpretation: True concept = "general offensive investment"
Top activations = "flanderized" SCU spam on special-dependent weapons
Label the general concept, note the super-stimuli pattern.
```

### Interpreting Overview Results

#### Activation Statistics

| Metric | What It Tells You |
|--------|-------------------|
| **Mean** | Average activation across all examples |
| **Sparsity** | % of examples with ~0 activation. High (>90%) = selective feature |
| **Examples** | Sample size for PageRank computation |

#### Top Tokens (PageRank)

These are abilities that **correlate** with high activation.

**⚠️ CRITICAL: Correlation ≠ Causation**

A token appearing here might be:
- **True driver**: Actually causes the feature to activate
- **Spurious correlation**: Just happens to co-occur with the true driver

You CANNOT conclude from overview alone that a token "drives" the feature. That requires experiments in Tier 2.

#### Family Breakdown

| Pattern | What It Suggests |
|---------|------------------|
| One family >40% | Single-family dominated - test that family |
| Top 2-3 families ~20% each | Multi-family - test interactions |
| Many families <15% each | Distributed - look for meta-pattern |

#### Top Weapons

| Pattern | What It Suggests |
|---------|------------------|
| One weapon >50% | Weapon-specific feature |
| One weapon class dominant | Class-specific (e.g., all blasters) |
| Diverse weapons | Not weapon-specific |

#### Bottom Tokens (Suppressors)

**Don't skip this!** Tokens that are RARE in high-activation examples tell you what the feature AVOIDS. This is often more informative than what it detects.

Example: If Quick Respawn, Special Saver, and Comeback are all suppressors, the feature detects "death-averse" builds (players who don't plan to die).

#### Identifying Binary Abilities

Check if any of these appear in the top PageRank:

| Binary Abilities |
|------------------|
| comeback, stealth_jump, last_ditch_effort, haunt |
| ninja_squid, respawn_punisher, object_shredder, drop_roller |
| opening_gambit, tenacity, thermal_ink, ability_doubler |

**If yes:** These need special handling in Tier 2 because they show delta=0 in standard sweeps.

### Classifying the Feature

Based on overview, classify the feature to guide Tier 2:

| Classification | Criteria | Next Steps |
|----------------|----------|------------|
| Single-family dominated | One family >40% | 1D sweep for that family |
| Multi-family | Top 2-3 families balanced | 2D heatmaps for interactions |
| Weapon-specific | One weapon >50% | Kit sweep, weapon analysis |
| Binary-influenced | Binary abilities in top 5 | Special binary handling |
| Distributed/unclear | No clear pattern | May need Tier 3 |

---

## Tier 2: Deep Dive

**Goal:** Find causal drivers through experiments.

**Time:** 15-30 minutes

**Output:** Confident label OR decision to proceed to Tier 3

### Step 2.1: Run 1D Sweeps

A 1D sweep tests how activation changes as you vary one ability family's AP level.

Claude Code runs sweeps for the top families from the overview:

```python
from splatnlp.mechinterp.skill_helpers import load_context
from splatnlp.mechinterp.experiments.family_sweep import Family1DSweepRunner
from splatnlp.mechinterp.schemas.experiment_specs import ExperimentSpec, ExperimentType

ctx = load_context('ultra')

families = ['special_charge_up', 'swim_speed_up', 'ink_saver_main']  # Top families from overview

for family in families:
    spec = ExperimentSpec(
        type=ExperimentType.FAMILY_1D_SWEEP,
        feature_id=FEATURE_ID,
        model_type='ultra',
        variables={'family': family}
    )
    runner = Family1DSweepRunner()
    result = runner.run(spec, ctx)
    print(f"{family}: max_delta={result.aggregates.max_delta:.3f}")
```

### Interpreting 1D Sweep Results

The key metric is **max_delta** - the maximum change in activation across AP levels.

| max_delta | Interpretation |
|-----------|----------------|
| > 0.10 | **Very strong driver** |
| 0.05 - 0.10 | **Strong driver** |
| 0.02 - 0.05 | **Weak driver** |
| < 0.02 | **Not a driver** (or binary ability) |
| Negative | **Suppressor** - higher AP reduces activation |

**Example Results:**
```
special_charge_up: max_delta=0.153  ← Strong driver!
swim_speed_up: max_delta=0.038     ← Weak driver
ink_saver_main: max_delta=0.007    ← Not a driver
comeback: max_delta=0.000          ← Binary ability (see Step 2.2)
```

#### ⚠️ Watch for Error-Correction Features

If 1D sweeps show **small deltas** or effects that **only appear in unusual rung combinations**, the feature may be an **error-correction feature**.

Error-correction features fire when:
- Expected prerequisites are MISSING
- Out-of-distribution (OOD) ability combinations occur
- The model is "confused" and needs to correct

**How to detect:**

Compare activation when low-level evidence is present vs missing:

```python
# Example: Does feature fire MORE when SCU_3 is missing from a high-SCU build?
high_scu_with_low = df.filter(
    pl.col('ability_input_tokens').list.contains(scu_57_id) &
    pl.col('ability_input_tokens').list.contains(scu_3_id)
)
high_scu_without_low = df.filter(
    pl.col('ability_input_tokens').list.contains(scu_57_id) &
    ~pl.col('ability_input_tokens').list.contains(scu_3_id)
)

mean_with = high_scu_with_low['activation'].mean()
mean_without = high_scu_without_low['activation'].mean()

print(f"With SCU_3: {mean_with:.4f}")
print(f"Without SCU_3: {mean_without:.4f}")
# If WITHOUT > WITH, feature fires when prerequisite is missing = error correction
```

**Signs of error-correction:**

| Pattern | Interpretation |
|---------|----------------|
| Higher activation when low rungs MISSING | "Explains away" missing evidence |
| Only fires on weird rung combos (e.g., SCU_57 without SCU_3) | OOD detector |
| Negative interaction in 2D heatmaps | Interference/correction |

**Label style:** "Error-Correction: {FAMILY} Missing Low Rungs" or "OOD Detector: {PATTERN}"

### Step 2.2: Handle Binary Abilities

**Why binary abilities show delta=0:**

Binary abilities don't have AP levels - you either have them (10 AP) or you don't. A 1D sweep has nothing to vary, so delta=0.

**This does NOT mean they're unimportant!**

#### Check 1: Presence Rate Enrichment

Compare how often the binary ability appears in high-activation vs all examples:

```python
# Enrichment = (rate in high activation) / (rate in all examples)
# Enrichment > 1.5x suggests the ability is characteristic
# Enrichment < 0.7x suggests the feature AVOIDS this ability
```

| Enrichment | Interpretation |
|------------|----------------|
| > 2.0x | **Strongly characteristic** |
| 1.5 - 2.0x | **Moderately characteristic** |
| 0.7 - 1.5x | **Neutral** |
| < 0.7x | **Depleted** - feature avoids this ability |

#### Check 2: Mean Activation WITH vs WITHOUT

```python
# Compare mean activation when ability is present vs absent
# Delta > 0.03 suggests meaningful effect
```

#### Check 3: 2D Conditional Analysis (MOST IMPORTANT)

Binary abilities can have **conditional effects** - they matter more at certain scaling ability levels.

Example: Comeback might have no effect on its own, but at ISM_6 it might double the activation.

```python
# For each scaling level (0, 3, 6, 12, 21, 29):
#   - Compare activation WITH vs WITHOUT binary ability
#   - Look for levels where delta is large
```

**What to look for:**

| Pattern | Interpretation |
|---------|----------------|
| Delta varies by level | **Conditional effect** - binary matters at certain levels |
| Delta consistent across levels | **Additive effect** - independent contribution |
| Delta ~0 at all levels | **Spurious correlation** |

#### Check 4: Test Combinations

Binary abilities often work as **packages**. Test multiple together:

```python
# Compare:
# - Neither present
# - Only first present
# - Only second present
# - Both present

# If "both present" >> sum of individual effects, there's synergy
```

### Step 2.3: Check Weapon Patterns

If weapons are concentrated in the overview:

1. **Check if truly weapon-specific** or kit-based (shared sub/special weapon)
2. **Use splatoon3-meta skill** to look up weapon kits
3. **Cross-reference** - do high-activation weapons share a sub or special?

If they share kit components, the feature may encode kit behavior, not weapon behavior.

### Step 2.4: Synthesize Findings

Create a summary table of all evidence:

| Evidence Type | Finding | Interpretation |
|---------------|---------|----------------|
| 1D Sweep: SCU | max_delta=0.153 | Strong driver |
| 1D Sweep: SSU | max_delta=0.038 | Weak driver |
| Binary: Comeback | 2.02x enriched, +0.056 delta | Characteristic |
| 2D: ISM × CB | +0.205 delta at ISM_6 | Conditional effect |
| Weapons | 76% Stamper | Weapon-concentrated |

### Step 2.5: Propose Label

| Finding Pattern | Label Style | Example |
|-----------------|-------------|---------|
| Single strong scaling driver | "{FAMILY} Build" | "SCU Stacker" |
| Scaling + binary conditional | "{Scaling} + {Binary}" | "ISM + Comeback Build" |
| Multiple scaling drivers | "{A} + {B} Combo" | "SSU + RSU Mobility" |
| Weapon-specific | "{Weapon} {Ability}" | "Stamper ISM Build" |
| All effects weak | → Proceed to Tier 3 | - |

### When to Proceed to Tier 3

Go to Tier 3 if:
- All max_deltas < 0.03
- No clear primary driver
- Conflicting signals
- Feature seems important but unclear why

---

## Tier 3: Deeper Dive

**Goal:** Systematic hypothesis testing for thorny features.

**Time:** 30-60 minutes

**Output:** Confident label OR "confirmed weak/unimportant"

### When to Use Tier 3

- Standard deep dive didn't produce clear interpretation
- All scaling effects weak (max_delta < 0.03)
- No clear primary driver identified
- Feature contributes to outputs but activation patterns unclear

### Step 3.1: Check Decoder Weights

**Key insight:** A feature's importance = activation strength × decoder weights

A feature with weak activation effects might still matter if it has high decoder weights!

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

print(f"Decoder weight percentile: {percentile:.1f}%")
```

| Percentile | Interpretation |
|------------|----------------|
| > 75% | **High influence** - feature important despite weak activation |
| 25-75% | **Typical** |
| < 25% | **Low influence** - likely unimportant |

### Step 3.2: Final Classification

| Activation Effects | Decoder Weights | Conclusion |
|-------------------|-----------------|------------|
| Strong (>0.05) | High (>75%) | Important, clear driver |
| Strong (>0.05) | Low (<25%) | Important for activation, weak output influence |
| Weak (<0.03) | High (>75%) | "Fine-tuning" feature - needs more investigation |
| Weak (<0.03) | Low (<25%) | **Confirmed weak/unimportant** |

### Step 3.3: Use Hypothesis State (Optional)

For complex investigations, track hypotheses formally:

```python
from splatnlp.mechinterp.state import ResearchState, Hypothesis

state = ResearchState(feature_id=FEATURE_ID, model_type="ultra")

state.add_hypothesis(Hypothesis(
    id="h1",
    description="Feature encodes weapon-specific pattern",
    status="pending"
))

# After running experiments:
state.add_evidence(
    hypothesis_id="h1",
    experiment_type="weapon_sweep",
    finding="Distributed across weapons, not weapon-specific",
    supports=False
)

state.update_hypothesis("h1", status="rejected")
```

This creates an audit trail for revisiting later.

---

## Advanced: Beam Trace Net Contribution Analysis

When you want to understand how features influence the model's actual token selections during generation, use **beam trace net contribution analysis**. This goes beyond understanding what activates a feature to understanding what the feature *does* to model outputs.

### When to Use This

- Understanding a feature's role in the generation process
- Comparing how Full vs Ultra SAEs influence the same weapon's build
- Finding which features drive specific ability selections
- Identifying features that suppress certain abilities

### Net Contribution Formula

For each token selection, the **net contribution** of a feature measures how much it pushes (or suppresses) that token's logit:

```
contribution_f = activation_f × (d_f · W_out[t])
```

Where:
- `activation_f` = the feature's activation value for this input
- `d_f` = the feature's decoder direction (512-dimensional vector)
- `W_out[t]` = the output layer weights for token t
- `d_f · W_out[t]` = dot product giving the feature's "vote" direction for token t

**Interpretation:**
- **Positive contribution**: Feature pushes toward selecting this token
- **Negative contribution**: Feature suppresses this token
- **Magnitude**: Strength of influence

### Running Beam Trace Analysis

```python
from splatnlp.mechinterp.beam_trace import run_beam_trace
import json

# Run beam trace for a weapon from NULL
trace = run_beam_trace(
    model_type="ultra",  # or "full"
    weapon_id=8000,      # Splatana Stamper
    initial_tokens=[],   # Empty = NULL start
    max_steps=6
)

# Save for analysis
with open("/tmp/trace.json", "w") as f:
    json.dump(trace, f, indent=2)
```

### Computing Net Contributions

```python
import torch
from splatnlp.mechinterp.skill_helpers import load_context

ctx = load_context("ultra")

# Load SAE decoder weights
sae_path = '/mnt/e/dev_spillover/SplatNLP/sae_runs/run_20250704_191557/sae_model_final.pth'
sae_ckpt = torch.load(sae_path, map_location='cpu', weights_only=True)
decoder = sae_ckpt['decoder.weight']  # [512, num_features]

# Load model output layer
model_path = '/mnt/e/dev_spillover/SplatNLP/saved_models/dataset_v0_2_super/clean_slate.pth'
model_ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
output_weight = model_ckpt['output_layer.weight']  # [vocab_size, 512]

def compute_contributions(step, selected_token_id):
    """Compute net contributions for a trace step."""
    contributions = []

    for feat in step['top_features']:
        fid = feat['feature_id']
        act = feat['activation']

        # Get decoder direction for this feature
        d_f = decoder[:, fid]

        # Get output weight for selected token
        w_t = output_weight[selected_token_id]

        # Net contribution = activation × (decoder · output_weight)
        contribution = act * torch.dot(d_f, w_t).item()

        contributions.append({
            'feature_id': fid,
            'activation': act,
            'contribution': contribution,
            'label': get_label(fid)  # from consolidated labels
        })

    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    return contributions
```

### Interpreting Contributions

For each step in the trace, report contributions as `±X` values:

| Contribution | Interpretation |
|-------------|----------------|
| > +0.5 | Strong driver of this selection |
| +0.1 to +0.5 | Moderate driver |
| -0.1 to +0.1 | Minimal influence |
| -0.5 to -0.1 | Moderate suppressor |
| < -0.5 | Strong suppressor |

**Example output:**
```
Step 1: Selected swim_speed_up_12 (prob=0.847)
  Top contributors:
    [+0.82] Feature 1234: "SSU Stacker"
    [+0.31] Feature 5678: "Mobility Build"
  Top suppressors:
    [-0.45] Feature 9012: "Ink Efficiency Focus"
    [-0.22] Feature 3456: "Anchor Playstyle"
```

### Comparing Full vs Ultra Traces

This analysis reveals architectural differences between SAE models:

| Aspect | Full Model | Ultra Model |
|--------|------------|-------------|
| Per-feature strength | 30-50x stronger contributions | Distributed, weaker individual |
| Archetype commitment | Early, decisive | Gradual, consensus-based |
| Feature count | Fewer features active | More features voting |

Use this to understand whether a model commits to an archetype early (Full) or builds consensus across many features (Ultra).

---

## Quick Reference

### Evidence Thresholds

#### Scaling Abilities (1D Sweeps)

| max_delta | Strength |
|-----------|----------|
| > 0.10 | Very strong |
| 0.05 - 0.10 | Strong |
| 0.02 - 0.05 | Weak |
| < 0.02 | Not a driver |

#### Binary Abilities

| Metric | Strong | Moderate | Weak |
|--------|--------|----------|------|
| Enrichment | > 2.0x | 1.5-2.0x | < 1.5x |
| WITH/WITHOUT Delta | > 0.05 | 0.03-0.05 | < 0.03 |
| 2D Conditional Delta | > 0.10 | 0.05-0.10 | < 0.05 |

#### Decoder Weights

| Percentile | Influence |
|------------|-----------|
| > 90% | Very high |
| 75-90% | High |
| 25-75% | Typical |
| 10-25% | Low |
| < 10% | Very low |

### Decision Flowchart

```
START
  │
  ▼
┌─────────────────────────────────┐
│    TIER 0: TRIAGE               │
│    Check decoder weight         │
│    percentile                   │
└─────────────────┬───────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
   < 10%                  ≥ 10%
        │                   │
        ▼                   │
[Quick overview check]      │
        │                   │
   No structure?            │
        │                   │
        Yes                 │
        │                   │
        ▼                   │
┌──────────────────┐        │
│ LABEL:           │        │
│ "Weak/Aux [ID]"  │        │
│ DONE - move on   │        │
└──────────────────┘        │
                            │
        ┌───────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│    TIER 1: OVERVIEW             │
│    [Run Overview]               │
└─────────────────┬───────────────┘
                  │
                  ▼
Is one family >40%? ──Yes──► [1D Sweep that family]
  │                              │
  No                             ▼
  │                         Strong effect? ──Yes──► LABEL IT
  ▼                              │
Binary abilities                 No
in top 5?      ──Yes──► [Binary Analysis]
  │                              │
  No                             ▼
  │                         Conditional effect? ──Yes──► LABEL IT
  ▼                              │
[1D Sweeps for                   No
 top families]                   │
  │                              ▼
  ▼                    ┌────────────────────┐
Any strong             │   All effects weak  │
effects?               │   → TIER 3          │
  │                    └────────────────────┘
  Yes                            │
  │                              ▼
  ▼                    [Check Decoder Weights]
LABEL IT                         │
                                 ▼
                       High weights? ──Yes──► Investigate more
                                 │
                                 No
                                 │
                                 ▼
                       LABEL AS "WEAK FEATURE"
```

---

## Common Pitfalls

### 1. Presenting Overview as Findings

**Wrong:** "The overview shows SCU is the top token, so this feature detects SCU."

**Right:** "The overview shows SCU correlates with high activation. Let me run a 1D sweep to test if it's causal."

### 2. Ignoring Binary Abilities Because delta=0

**Wrong:** "Comeback has delta=0, so it doesn't matter."

**Right:** "Comeback is binary, so delta=0 is expected. Let me check enrichment and run 2D conditional analysis."

### 3. Missing Conditional Effects

Binary abilities might only matter at certain scaling levels. A feature might look like "just ISM" when it's actually "ISM + Comeback at moderate levels."

**Always run 2D analysis for binary abilities in the top PageRank.**

### 4. Ignoring Depletion

If enrichment < 0.7x, the feature actively AVOIDS that ability. This is meaningful!

Example: If QR, SS, and CB are all depleted, the feature detects "death-averse" builds.

### 5. Trusting Small Sample Sizes

High enrichment with few examples is unreliable.

**Wrong:** "Tenacity has 6.87x enrichment!"

**Right:** "Tenacity has 6.87x enrichment, but only 16 examples (0.05% of data). Too sparse to trust."

### 6. Labeling Weak Features Without Decoder Check

Weak activation effects don't mean the feature is unimportant. Check decoder weights first.

### 7. Forgetting Weapon/Kit Patterns

A feature might be weapon-specific or kit-specific, not ability-specific. Always check weapon distribution and consider kit analysis.

---

## Worked Example

### Feature 13352

#### Tier 1: Overview

**Top Tokens (PageRank):**
1. ink_saver_main (ISM)
2. quick_respawn (QR)
3. swim_speed_up (SSU)
4. comeback (binary)
5. stealth_jump (binary)

**Top Weapons:** Stamper (27%), Custom Jr (19%), various aggressive slayers

**Classification:** Binary-influenced (Comeback and Stealth Jump in top 5)

#### Tier 2: Deep Dive

**1D Sweeps:**
| Family | max_delta | Interpretation |
|--------|-----------|----------------|
| ISM | 0.069 | Moderate driver |
| QR | 0.039 | Weak driver |
| SSU | 0.031 | Weak driver |
| Comeback | 0.000 | Binary - need special analysis |
| Stealth Jump | 0.000 | Binary - need special analysis |

**Binary Analysis - Comeback:**
- Enrichment: 2.02x (strongly characteristic)
- WITH/WITHOUT delta: +0.056 (meaningful)

**Binary Analysis - Stealth Jump:**
- Enrichment: 1.71x (moderately characteristic)
- WITH/WITHOUT delta: +0.042 (meaningful)

**2D Conditional - ISM × (Comeback + Stealth Jump):**

| ISM Level | Neither | Both | Delta |
|-----------|---------|------|-------|
| 0 | 0.057 | 0.151 | +0.094 |
| 3 | 0.110 | 0.350 | +0.241 |
| 6 | 0.133 | **0.468** | **+0.335** |
| 12 | 0.088 | 0.235 | +0.147 |

**Key Finding:** The combination at ISM_6 produces +0.335 delta - the conditional effect is 5x stronger than the marginal effect!

**Synthesis:**
- ISM is the scaling driver (0.069 max_delta)
- Comeback + Stealth Jump have strong conditional effects at moderate ISM
- Peak activation when: ISM_6 + Comeback + Stealth Jump present
- Weapons are aggressive slayers

#### Label

**Old:** "Comeback Jump Build"

**New:** "Zombie Slayer (ISM + CB/SJ)"

**Rationale:** ISM is the scaling driver. Comeback + Stealth Jump ("death-accepting" abilities) have strong conditional effects at moderate ISM levels. The feature detects aggressive builds that accept dying but want ink efficiency when alive. Top weapons confirm aggressive playstyle.

---

## Appendix: Claude Code Skills Reference

### Invoking Skills

Claude Code automatically uses skills when relevant. You can also explicitly request them:

```
Use the mechinterp-overview skill to analyze feature 13352.
```

### Available Skills

| Skill | Description |
|-------|-------------|
| `mechinterp-overview` | Quick feature summary with PageRank, family breakdown, weapons |
| `mechinterp-runner` | Execute experiment specs (1D sweeps, 2D heatmaps, etc.) |
| `mechinterp-glossary-and-constraints` | Ability families, AP rungs, domain rules |
| `mechinterp-labeler` | Queue management, label storage, progress tracking |
| `mechinterp-state` | Hypothesis and evidence tracking for complex investigations |
| `mechinterp-investigator` | Full investigation workflow orchestration |
| `mechinterp-validation-suite` | Credibility checks (split-half, shuffle null) |
| `splatoon3-meta` | Game meta knowledge, weapon kits, ability synergies |
| `mechinterp-ability-semantics` | Ability family semantic groupings |

### Skill Locations

Skills are defined in `.claude/skills/` with SKILL.md files containing instructions and reference code.

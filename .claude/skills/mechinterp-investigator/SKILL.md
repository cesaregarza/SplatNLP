---
name: mechinterp-investigator
description: Orchestrate a systematic research program to investigate and meaningfully label SAE features
---

# MechInterp Investigator

This skill guides a systematic investigation of SAE features to arrive at meaningful, non-trivial labels. It orchestrates the other mechinterp skills into a coherent research workflow.

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

### Phase 1.5: Activation Region Analysis (CRITICAL)

**Don't only examine extreme activations!** High activations may be "super-stimuli" - exaggerated versions of the true concept.

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

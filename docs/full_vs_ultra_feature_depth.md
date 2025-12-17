# Feature Depth in SAE Labels: Full vs Ultra

This document compares the conceptual depth of labeled SAE features between the
Full and Ultra checkpoints.

It is intentionally qualitative and label-driven: the goal is to document what
the current labels already make obvious (even before Full labeling is complete),
and to give a shared vocabulary for "depth" when discussing features.

## Context: Full vs Ultra output quality is (currently) similar

On the Sendou Tier-1 reconstruction eval, Full and Ultra are statistically
indistinguishable across masks 1-6 on both `best_accuracy` and
`completion_slot_acc`.

See: `docs/sendou_tier1_eval_report.md`.

This matters because it helps decouple task performance from feature depth: the
features can differ substantially even when the eval metrics do not.

## What "feature depth" means in this doc

When we say a feature is "deeper", we mean it captures higher-level,
multi-factor structure that reads like a coherent concept, rather than a single
token/family detector.

A practical ladder (not a strict taxonomy):

1. **Atomic detectors (shallow)**: fires on one ability family/AP rung with
   little additional structure.
2. **Build archetypes (mid-depth)**: coherent multi-token packages, often
   weapon/kit-skewed (e.g., a common meta build).
3. **Strategy/role concepts (deep)**: cross-weapon patterns about how to play
   (death tolerance, visibility vs stealth, special economy philosophy, etc),
   often characterized by trade-offs (promote X, suppress Y), weak single
   drivers, and sometimes conditional interactions.

## Data sources (labels + model/SAE sizes)

Labels:
- Full: `/mnt/e/mechinterp_runs/labels/consolidated_full.json`
- Ultra: `/mnt/e/mechinterp_runs/labels/consolidated_ultra.json`

SAE sizes used in this repo's tooling:
- Full SAE: **2,048 features** (expansion factor 4.0)
- Ultra SAE: **24,576 features** (expansion factor 48.0)

See: `src/splatnlp/mechinterp/cli/beam_trace_cli.py`.

### SAE selection note (why Ultra is 48x)

Full's SAE was trained first. Ultra's SAE configuration was then selected **ex
post** after a large sweep (~342 Ultra SAE runs across expansion factors and
other hyperparameters), choosing the run that best matched the Full SAE on key
training diagnostics (e.g., L0 sparsity, reconstruction MSE, dead neurons).

So the 48x expansion factor was not an "ex ante" design choice; it was the
outcome of matching sparsity/fidelity criteria.

One subtlety: the "sparsity" numbers reported in these runs are fractions
(e.g., ~0.09 of features active). Matching the fraction across SAEs of
different sizes does not match the absolute number of active features per
example. At ~0.09:

- Full (2,048 feats) uses ~**190** active features/example.
- Ultra (24,576 feats) uses ~**2,200** active features/example.

So even with similar sparsity fractions, the Ultra SAE has much more room to
allocate separate features to sub-patterns.

Also, more room does not guarantee better reconstruction. In practice, changing
expansion factor moves you along a Pareto frontier between reconstruction
fidelity (MSE / logit fidelity), sparsity, and "health" (dead neurons /
stability). It is possible (and common) to see more splitting with slightly
worse reconstruction, especially when sparsity is enforced strongly.

Importantly, in this project Ultra's SAE size was not chosen "just because we
could": the sweep result can be read as evidence that smaller Ultra SAEs were
capacity-limited (forced into polysemy or poor health/fidelity) under the
target regime, and that a larger dictionary was necessary to realize a stable,
high-quality sparse code.

### Snapshot counts (as of the label files on disk)

These numbers change as labels are added, so treat them as a snapshot:

| model | labeled features | mechanical | tactical | strategic |
| :--- | ---: | ---: | ---: | ---: |
| full | 35 | 10 | 24 | 1 |
| ultra | 103 | 2 | 92 | 9 |

To refresh:

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

order = ["mechanical", "tactical", "strategic"]

print("| model | labeled features | mechanical | tactical | strategic |")
print("| :--- | ---: | ---: | ---: | ---: |")

for name in ["full", "ultra"]:
    p = Path(f"/mnt/e/mechinterp_runs/labels/consolidated_{name}.json")
    data = json.loads(p.read_text())
    cats = Counter(v.get("dashboard_category", "<none>") for v in data.values())
    row = [name, str(len(data))] + [str(cats.get(k, 0)) for k in order]
    print("| " + " | ".join(row) + " |")
PY
```

## What the labels already show

### Full: more shallow detectors + broad "packages"

The current Full labels skew toward:

- **Atomic ability detectors / low-AP detectors**, e.g.
  - Feature **842** (Swim Speed Up Detector): clean monosemantic feature for
    swim speed investment.
  - Feature **692** (Low ISM Preference (6AP)): strongly prefers minimal
    `ink_saver_main` investment.
- **Weapon/kit meta archetypes**, often Stamper-heavy, e.g.
  - Feature **959** (Stamper Stealth Jump Meta): 100% Stamper-specific.
  - Feature **2039** (Stamper Burst Bomb Efficiency Kit): explicitly
    weapon/kit-linked.
- **Respawn-cycle packages** that read as broad bundles (QR/Comeback/SJ), e.g.
  feature **540** (Comeback-Centric Respawn Build) and feature **889** (Quick
  Respawn Death-Trading Build).

There are mid-depth playstyle concepts in Full (e.g., feature **1512** (Stealth
Approach - Midrange/Slayer)), and at least one explicitly strategic label
(feature **350** (Low-Investment Diversified Builds)), but the labeled set so
far contains far fewer clear role/strategy features than Ultra that generalize
across many weapons and are defined by crisp trade-offs.

### Ultra: more splitting + more strategy/role concepts

Ultra labels already contain a distinct tier of deep concepts, including
features explicitly labeled strategic, and (importantly) features whose notes
emphasize:

- **Weak single drivers** ("1D sweeps: all weak") -> concept is not "just SSU"
  or "just QR", but a multi-factor region of build space.
- **Promote/suppress structure** -> reads like a policy with trade-offs.
- **Conditional interactions** -> concept defined by combinations more than
  marginals.

Concrete examples from the current Ultra label set:

- Feature **16329** (Special Investment Portfolio - Support/Anchor): diversified
  special investment across multiple families, with an explicit SCU x Tenacity
  interaction ("anti-synergy at high SCU").
- Feature **10938** (Positional Survival - Midrange): "Survival through
  positioning, not stealth or trading," with a coherent set of promotes
  (mobility + resist) and suppressions (QR, stealth, trading).
- Feature **22988** (Balanced Utility - Visible Midrange+): activates on the
  absence of stacking and is framed as a third archetype distinct from stealth
  slayers and zombie slayers.

These are qualitatively different from pure family detectors: they look like
build philosophies and role concepts.

## A concrete "depth" pattern: feature splitting

One way depth shows up in practice is splitting: what looks like a single
blended package in Full becomes multiple separable, more legible concepts in
Ultra.

Example (respawn/stealth packages):

- Full has broad bundles like:
  - **540** "Comeback-Centric Respawn Build" (CB + QR + SSU/SJ synergy)
  - **889** "Quick Respawn Death-Trading Build" (QR + CB + SJ)
- Ultra separates related structure into distinct directions, e.g.:
  - **12099** "QR Stacker (Slayer)" (QR-dominant)
  - **5741** "Stealth Mobility Build" (SSU + stealth jump significance)
  - **13352** "Zombie Slayer (ISM + CB/SJ)" (a conditional "zombie package")

This doesn't mean Full "lacks" the concept; it often means the Full SAE (2K)
cannot disentangle overlapping concepts as cleanly as the Ultra SAE (24K), and
and/or the Ultra model's representation itself is more separable after longer
training.

## Beyond horizontal splitting: vertical hierarchy

The previous section frames Ultra's advantage as horizontal splitting: one Full
feature becomes multiple Ultra features at similar conceptual depth. But the
label category distribution hints at something stronger.

From the snapshot above, Full currently has more mechanical labels (10) than
Ultra (2) despite having far fewer total labels. If Ultra were simply splitting
Full's features into finer mechanical pieces, we'd expect the opposite.

One possible explanation is that Ultra is not just splitting horizontally; it
is also learning vertical structure: higher-level features that organize
mechanical patterns.

Schematically (illustrative, not a literal mapping):

```
Full (capacity-limited):
- Comeback-centric respawn build (tactical bundle): entangles QR + CB + SJ.

Ultra (capacity-rich):
- QR threshold (mechanical)
- Comeback presence (mechanical)
- Stealth Jump presence (mechanical)
- Zombie slayer philosophy (strategic): fires on the combination as a coherent
  playstyle.
```

Under this view, Ultra's strategic features are not leftover variance. Instead,
they are emergent abstractions that capture the "why" behind ability
combinations. The SAE has enough capacity to represent both:

1. Atomic building blocks (mechanical layer)
2. Latent dimensions that unify them (strategic layer)

Full's SAE, under capacity pressure, collapses these into tactical "packages"
that blend both signals. This can be efficient for reconstruction but harder to
interpret as a coherent concept.

### Why this matches how builds are organized

The vertical hierarchy interpretation feels plausible because competitive
builds are often discussed in terms of strategic axes, not just specific
ability bundles. For example:

**Death tolerance** ("zombie" vs "anchor"):
- Zombie builds (Comeback + Quick Respawn + Stealth Jump) weaponize death: they
  accept frequent deaths in exchange for aggressive positioning and
  respawn-window buffs.
- Anchor builds (Respawn Punisher + Object Shredder) punish death: they assume
  low death rates and make each kill count harder.

These are philosophies about how to convert deaths into value. A strategic
feature encoding "zombie slayer" would fire on the combination because it
implies a coherent risk posture, not because Quick Respawn and Comeback happen
to co-occur.

**Visibility trade-off** (stealth vs visible):
- Stealth builds (Ninja Squid + some SSU + Stealth Jump) sacrifice speed and
  slots for approach concealment.
- Visible builds accept being tracked but gain raw stats or utility.

**Investment philosophy** ("omamori"/utility spread vs stacking):
- "Omamori" builds spread small investments across many defensive options (QSJ,
  Sub Resistance, Ink Resistance, Special Saver, etc.) to prevent catastrophic
  outcomes without over-committing.
- Stacking builds concentrate AP into one or two families for breakpoint
  efficiency.

Full's features often detect which abilities are present (mechanical) or which
combination is popular (tactical). Ultra's strategic features appear to detect
which philosophy is at play, organizing multiple mechanical patterns under a
coherent concept.

### Evidence for hierarchy (vs. just more splitting)

Several Ultra strategic features show signatures consistent with vertical
abstraction:

- **Weak single drivers**: 1D family sweeps show small deltas, but the feature
  activates strongly on combinations.
- **Promote/suppress trade-offs**: decoder weights read like a policy ("if you
  want X, avoid Y"), mirroring meta advice like "don't stack QR if you're
  running Respawn Punisher."
- **Cross-weapon generalization**: the concept applies across weapon classes,
  suggesting it captures a playstyle rather than a kit-specific build.

Full's tactical labels, by contrast, often have one or two dominant drivers
(e.g., "QR family delta = 0.86") and stronger weapon/kit specificity. They look
more like common build templates than strategic dimensions.

### Caveat: labeling bias

This interpretation depends on the labels being representative. If Full were
labeled more exhaustively, we might find strategic features hiding in the
unlabeled 2,013 features. The current snapshot (35 Full labels vs 103 Ultra) is
too small to be conclusive.

A sharper test: run the same labeling protocol on a random sample of Full
features and compare the category distribution at scale.

## Case study: Splatana Stamper beam trace comparison

To make the depth difference concrete, we traced both models' predictions for
Splatana Stamper (weapon ID 8000) starting from an empty build (`<NULL>`), and
computed each feature's net contribution to the selected token at each step.

Note: This is a single trace and is meant as an illustration, not a population
statistic.

### What "net contribution" means

For each step where the model selects ability token $t$, we compute:

$$\\text{contribution}_f = \\text{activation}_f \\times (\\mathbf{d}_f \\cdot \\mathbf{W}_{\\text{out}}[t])$$

Where:
- $\\text{activation}_f$ is the SAE feature's activation at that step
- $\\mathbf{d}_f$ is the feature's decoder direction (what it reconstructs)
- $\\mathbf{W}_{\\text{out}}[t]$ is the output layer's weight vector for token
  $t$

This measures how much the feature's activation pushes the logit for the
selected token. Positive values promote the selection; negative values suppress
it (the token was selected despite this feature pushing against it).

### Different builds generated

The two models produce substantially different builds for the same weapon:

| Step | Full Selection | Ultra Selection |
|:----:|:---------------|:----------------|
| 1 | ink_saver_main_3 | ink_saver_main_3 |
| 2 | **comeback** | **comeback** |
| 3 | **quick_respawn_15** | special_power_up_3 |
| 4 | ink_resistance_up_3 | special_power_up_6 |
| 5 | ink_recovery_up_3 | ink_recovery_up_3 |
| 6 | n/a | sub_resistance_up_3 |

Full moves into a zombie-like path (Comeback + QR stacking).

Ultra builds a utility spread: no Quick Respawn, instead adding Special Power
Up, Ink Recovery Up, and Sub Resistance Up (small investments across multiple
families).

### Contribution magnitude difference

| Model | Typical top promoter | Typical top suppressor |
|:------|---------------------:|-----------------------:|
| Full  | +5.0 to +9.4 | -0.3 to -1.0 |
| Ultra | +0.1 to +0.2 | -0.05 to -0.13 |

Full's per-feature contributions are ~30-50x stronger. This reflects:

- **Full**: single features dominate decisions. One tactical bundle can swing
  the logit by +7 or more.
- **Ultra**: many features contribute small amounts. The decision emerges from
  distributed voting across dozens of weak contributors.

### Feature-by-feature breakdown

#### Step 2: Comeback selection (both models agree)

| Model | Top Promoter | Contribution | Interpretation |
|:------|:-------------|-------------:|:---------------|
| Full | f540 "Comeback-Centric Respawn Build" | **+7.52** | Tactical package - implies the whole zombie archetype |
| Ultra | f2164 "Dualie Squelchers Zombie Support" | +0.24 | Weapon-adjacent hint, but weak signal |

Full's f540 fires so strongly that it effectively commits the build to the
zombie path. Ultra has no equivalent "lock-in" feature.

#### Step 3: Divergence point

| Model | Selection | Top Promoter | Contribution |
|:------|:----------|:-------------|-------------:|
| Full | quick_respawn_15 | f889 "Quick Respawn Death-Trading Build" | **+7.65** |
| Ultra | special_power_up_3 | f9252 "ISM + SPU Scaling Build" | +0.18 |

Full continues the zombie package: f889 is another tactical bundle that
promotes QR stacking alongside Comeback (and often Stealth Jump).

Ultra pivots to a different build philosophy. f9252 represents a strategic
concept (scaling ISM and SPU together), not a specific meta build.

#### Step 4: Utility additions

| Model | Selection | Top Promoter | Contribution |
|:------|:----------|:-------------|-------------:|
| Full | ink_resistance_up_3 | f1654 "Low-AP Ink Resistance (Anti-Stacking)" | **+9.38** |
| Ultra | special_power_up_6 | f6291 "Sub/Special Power (Distributed)" | +0.10 |

Full's f1654 is a mechanical detector: it fires on exactly 3 AP of Ink
Resistance (an "omamori"/utility-spread investment pattern).

Ultra's f6291 is a strategic feature: it encodes investment philosophy
(distributed sub/special investment) rather than a specific AP threshold.

### Suppressor patterns

The features pushing against selections also differ qualitatively:

**Full suppressors** are often anti-stacking features:
- f692 "Low ISM Preference (6AP)" suppresses QR_15 at -0.97
- f350 "Low-Investment Diversified Builds" suppresses QR_15 at -0.39
- f65 "Respawn-Cycle Balanced Build (Anti-Stacking)" suppresses QR_15 at -0.35

These represent alternative build philosophies that would prefer spreading AP
rather than committing to the zombie package.

**Ultra suppressors** are weapon-specific builds:
- f5784 "Tenta Brella QR Build" suppresses SPU selections
- f1994 "Stamper SSU/ISM Build" suppresses SPU_6
- f13352 "Zombie Slayer (ISM + CB/SJ)" suppresses SRU_3

These represent alternative Stamper archetypes: the model "knows" multiple
valid Stamper builds, and the suppressors are features preferring a different
one.

### Interpretation

This trace comparison supports the vertical hierarchy hypothesis:

1. **Full has high-magnitude tactical bundles** that commit to an archetype
   early. Once f540 "Comeback-Centric Respawn Build" fires at +7.5 in step 2,
   the zombie path is effectively locked in; f889 naturally follows.

2. **Ultra has distributed strategic voting** with no single dominant feature.
   This allows the model to explore a different region of build space: a broad
   utility spread that Full's features do not cleanly represent.

3. **Full's suppressors are build philosophies**; **Ultra's suppressors are
   build variants**. This suggests Ultra has finer-grained representation of
   the Stamper build space, while Full compresses it into a few competing
   archetypes.

4. **Ultra discovers a "third way"** for Stamper that Full does not represent:
   a visible midrange utility build (no QR, no zombie, just broad defensive
   coverage). This matches the "omamori"/utility-spread philosophy: small
   investments across multiple defensive options to prevent catastrophic
   outcomes without over-committing to any single family.

## Working hypothesis (what to test next)

The observed depth gap plausibly reflects a mix of:

1. **Effective SAE capacity**: Ultra's SAE is larger (selected ex post to hit
   the target "healthy" regime), which reduces the pressure to compress
   multiple correlated concepts into one latent and makes splitting easier.
2. **Backbone representation**: Ultra's longer training may encode more (or
   more separable) high-level structure, making strategic/role axes easier for
   the SAE to isolate.

To disentangle these:

- Compare against smaller Ultra SAE runs from the sweep (lower expansion
  factors) to estimate how much "depth" is purely capacity-driven.
- Train a larger Full SAE (targeting similar sparsity/MSE) to test whether
  strategic/role concepts emerge with more splitting.

## Practical implications (why this matters)

- **Interpretability**: Ultra features are easier to narrate as reasons during
  beam search because they correspond to coherent sub-decisions (role, death
  tolerance, special economy philosophy), not just "more SSU".
- **Steering/ablation**: deeper features are better knobs; they change a build
  along an intelligible axis instead of causing collateral changes from blended
  detectors.

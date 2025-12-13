---
name: mechinterp-labeler
description: Manage feature labeling workflow - queue management, label storage, similar features, progress tracking
---

# MechInterp Labeler

Manage the feature labeling workflow. This skill provides tools for:
- Priority queue management
- Setting and syncing labels
- Finding similar features
- Tracking labeling progress

## Purpose

The labeler skill enables interactive feature labeling sessions:
1. Get the next feature to label from a priority queue
2. Use overview and experiments to understand the feature
3. Save labels with categories and notes
4. Find similar features to label next
5. Track overall progress

## Commands

### Get Next Feature

```bash
cd /root/dev/SplatNLP

# Get next feature from queue
poetry run python -m splatnlp.mechinterp.cli.labeler_cli next --model ultra

# Don't auto-build queue if empty
poetry run python -m splatnlp.mechinterp.cli.labeler_cli next --model ultra --no-build
```

### Set a Label

**IMPORTANT**: Always use `--source` to track label provenance.

**Source Options:**
- `claude code` — Label created through Claude Code CLI investigation
- `codex` — Label created through Codex (OpenAI) agent
- `codex/claude` — Label created through Codex orchestrating Claude
- `manual` — Label created by human manually
- `dashboard` — Label created through dashboard UI (default)

```bash
# Label from Claude Code investigation
poetry run python -m splatnlp.mechinterp.cli.labeler_cli label \
    --feature-id 18712 \
    --name "Special Charge Stacker" \
    --model ultra \
    --source "claude code"

# With category and notes
poetry run python -m splatnlp.mechinterp.cli.labeler_cli label \
    --feature-id 18712 \
    --name "SCU Detector" \
    --category tactical \
    --notes "Responds to Special Charge Up presence, stronger at high AP" \
    --source "claude code"

# Manual labeling by human
poetry run python -m splatnlp.mechinterp.cli.labeler_cli label \
    --feature-id 18712 \
    --name "My Label" \
    --source "manual"
```

**Categories:**
- `mechanical`: Low-level patterns (token presence, combinations)
- `tactical`: Mid-level patterns (build strategies, weapon synergies)
- `strategic`: High-level patterns (playstyle, meta concepts)
- `none`: Uncategorized

## ⚠️ Super-Stimuli Warning

**High activations may be "flanderized" versions of the true concept!**

When labeling features, don't only examine extreme activations. High activation builds can be:
- **Super-stimuli**: Extreme, exaggerated versions of the core concept
- **Weapon-gated**: Only achievable on specific niche weapons
- **Unrepresentative**: Missing the general pattern that applies across weapons

### How to Detect Super-Stimuli

1. **Examine activation regions** (not just top/bottom):
   - Floor (≤0.01), Low (0.01-0.05), Mild (0.05-0.10)
   - Moderate (0.10-0.20), High (0.20-0.35), Very High (>0.35)

2. **Look for weapons that span ALL levels continuously**:
   - If Splattershot appears in every region → feature encodes a general concept
   - If only niche weapons reach Very High → those are "super-stimuli"

3. **Compare low-moderate vs very high**:
   - Low-moderate: diverse weapons, general builds = TRUE CONCEPT
   - Very high: concentrated on 3-4 special-dependent weapons = SUPER-STIMULI

### Example: Feature 9971

```
Initial label (wrong): "Death-Averse SCU Stacker"
- Only looked at high activations (SCU_57 + special-dependent weapons)

Better label: "Offensive Intensity (Death-Averse)"
- Low-moderate region showed diverse weapons (Splattershot family, Sploosh, Hydra)
- Feature tracks general offensive investment, not specifically SCU
- Very high region (Bloblobber, Glooga) are "super-stimuli" not the core concept
```

**Key insight**: The low-moderate region reveals the TRUE feature concept. High activations show what happens when that concept is pushed to extremes.

### Skip a Feature

```bash
# Skip the next feature
poetry run python -m splatnlp.mechinterp.cli.labeler_cli skip --model ultra

# Skip specific feature with reason
poetry run python -m splatnlp.mechinterp.cli.labeler_cli skip \
    --feature-id 18712 \
    --reason "ReLU floor too high, hard to interpret"
```

### Add Features to Queue

```bash
# Add single feature
poetry run python -m splatnlp.mechinterp.cli.labeler_cli add 18712 --model ultra

# Add multiple with priority
poetry run python -m splatnlp.mechinterp.cli.labeler_cli add 18712,18890,19042 \
    --priority 0.8 \
    --reason "SCU-related cluster"
```

### Find Similar Features

```bash
poetry run python -m splatnlp.mechinterp.cli.labeler_cli similar \
    --feature-id 18712 \
    --top-k 5 \
    --model ultra
```

### Check Status

```bash
poetry run python -m splatnlp.mechinterp.cli.labeler_cli status --model ultra
```

Output example:
```
## Labeling Status (ultra)

### Labels
- Total labeled: 45
- From dashboard: 30
- From research: 10
- Merged: 5

### Categories
- tactical: 20
- mechanical: 15
- strategic: 5
- uncategorized: 5

### Queue
- Pending: 25
- Completed: 40
- Skipped: 5
```

### Sync Labels

Pull labels from all sources (dashboard, research states):

```bash
poetry run python -m splatnlp.mechinterp.cli.labeler_cli sync --model ultra
```

### Export Labels

```bash
poetry run python -m splatnlp.mechinterp.cli.labeler_cli export \
    --model ultra \
    --output /mnt/e/mechinterp_runs/labels/export.csv
```

### Build Priority Queue

```bash
# By activation count (features with most data)
poetry run python -m splatnlp.mechinterp.cli.labeler_cli build-queue \
    --model ultra \
    --method activation_count \
    --top-k 50

# From cluster (similar to a seed feature)
poetry run python -m splatnlp.mechinterp.cli.labeler_cli build-queue \
    --model ultra \
    --method cluster \
    --seed 18712 \
    --top-k 10
```

## Typical Labeling Session

```
User: Let's label some features

Claude: [runs: labeler_cli next --model ultra]
        Next feature: 18712 (priority: 0.85)

        [runs: overview_cli --feature-id 18712]
        ## Feature 18712 Overview
        - Top token: special_charge_up (27%)
        - Family: SCU 31%
        ...

        Based on the overview, this feature appears to detect
        Special Charge Up stacking. Want me to run a sweep?

User: Yes, confirm with an SCU sweep

Claude: [runs: runner_cli with family_1d_sweep]
        Results confirm monotonic increase with SCU.

        Proposed label: "SCU Detector - High AP"

User: Call it "Special Charge Stacker"

Claude: [runs: labeler_cli label --feature-id 18712
               --name "Special Charge Stacker" --category tactical
               --source "claude code"]
        Label saved (source: claude code). Finding similar features...

        [runs: labeler_cli similar --feature-id 18712]
        Similar features:
        - 19042 (sim=0.82)
        - 18890 (sim=0.75)

        Want to add these to the queue?
```

## Label Storage

Labels are stored in three places (kept in sync):

1. **Dashboard**: `src/splatnlp/dashboard/feature_labels_{model}.json`
2. **Research State**: `/mnt/e/mechinterp_runs/state/{model}/f{id}.json`
3. **Consolidated**: `/mnt/e/mechinterp_runs/labels/consolidated_{model}.json`

The consolidator merges all sources and resolves conflicts.

## Queue Storage

Queue state is persisted at:
- `/mnt/e/mechinterp_runs/labels/queue_{model}.json`

Contains:
- Pending entries with priorities
- Completed feature IDs
- Skipped feature IDs

## Programmatic Usage

```python
from splatnlp.mechinterp.labeling import (
    LabelConsolidator,
    LabelingQueue,
    QueueBuilder,
    SimilarFinder,
)

# Queue management
queue = LabelingQueue.load("ultra")
entry = queue.get_next()
queue.mark_complete(entry.feature_id, "My Label")

# Set labels
consolidator = LabelConsolidator("ultra")
consolidator.set_label(
    feature_id=18712,
    name="SCU Detector",
    category="tactical",
    notes="Responds to SCU presence",
)

# Find similar
finder = SimilarFinder("ultra")
similar = finder.find_by_top_tokens(18712, top_k=5)

# Build queue
builder = QueueBuilder("ultra")
queue = builder.build_by_activation_count(top_k=50)
```

## See Also

- **mechinterp-overview**: Quick feature overview before labeling
- **mechinterp-runner**: Run experiments to validate hypotheses
- **mechinterp-state**: Track detailed research progress
- **mechinterp-summarizer**: Generate notes from experiments

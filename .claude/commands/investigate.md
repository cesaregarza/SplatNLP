Investigate SAE features: $ARGUMENTS

## Execution Strategy

### Single Feature
Use the **mechinterp-investigator** skill directly to investigate the feature.

### Multiple Features

#### Step 1: Pre-warm Server Cache
**IMPORTANT**: Before spawning parallel subagents, sequentially warm the activation server cache for all features. This prevents parallel agents from hitting simultaneous 40s DB loads.

```bash
# Check server is running
curl -s http://127.0.0.1:8765/health | jq .

# Pre-warm cache for each feature (run sequentially, ~40s each on cache miss)
for fid in {feature_ids}; do
  echo "Warming cache for feature $fid..."
  curl -s "http://127.0.0.1:8765/activations/$fid/arrow" -o /dev/null \
    -w "Feature $fid: %{time_total}s\n"
done
```

If server is not running, start it first:
```bash
poetry run python -m splatnlp.mechinterp.server.activation_server --port 8765 &
```

#### Step 2: Spawn Parallel Subagents
Once cache is warm, spawn parallel subagents:

```
For each feature ID in the request:
  Task(
    subagent_type="general-purpose",
    prompt="Use the mechinterp-investigator skill to investigate feature {ID} with the ultra model. Follow the full protocol in .claude/skills/mechinterp-investigator/SKILL.md.",
    run_in_background=true
  )
```

Wait for all subagents to complete, then consolidate findings.

## CLI Tools (No JSON Required)

Subagents should use the direct subcommand interface:

```bash
# Overview with extended analyses
poetry run python -m splatnlp.mechinterp.cli.overview_cli \
    --feature-id {ID} --model ultra --all

# 1D family sweep
poetry run python -m splatnlp.mechinterp.cli.runner_cli family-sweep \
    --feature-id {ID} --family {FAMILY} --model ultra

# 2D heatmap
poetry run python -m splatnlp.mechinterp.cli.runner_cli heatmap \
    --feature-id {ID} --family-x {X} --family-y {Y} --model ultra

# Binary ability analysis
poetry run python -m splatnlp.mechinterp.cli.runner_cli binary \
    --feature-id {ID} --model ultra

# Weapon/kit sweeps
poetry run python -m splatnlp.mechinterp.cli.runner_cli weapon-sweep \
    --feature-id {ID} --model ultra --top-k 20

poetry run python -m splatnlp.mechinterp.cli.runner_cli kit-sweep \
    --feature-id {ID} --model ultra --analyze-combinations

# Decoder output
poetry run python -m splatnlp.mechinterp.cli.decoder_cli output-influence \
    --feature-id {ID} --model ultra --top-k 15
```

## Required Protocol Per Feature

| Phase | Test | CLI Command |
|-------|------|-------------|
| 0 | Decoder triage | `decoder_cli weight-percentile` |
| 1 | Overview + extended | `overview_cli --all` |
| 2 | 1D sweeps | `runner_cli family-sweep` (top 3-5 families) |
| 2 | Binary enrichment | `runner_cli binary` |
| 3 | 2D heatmaps | `runner_cli heatmap` (required for interactions) |
| 4 | Decoder output | `decoder_cli output-influence` |
| 5 | Weapon vibes | Read `splatoon3-meta/references/weapon-vibes.md` |

## Output Format

Each subagent should produce:
```
## Feature XXXXX

**Label:** [proposed label]
**Category:** mechanical | tactical | strategic
**Confidence:** 0.0-1.0

### Evidence
- 1D sweeps: [family]: [delta] (causal/not causal)
- Binary enrichment: [ability]: [X.XXx] (enriched/depleted/normal)
- 2D interactions: [X × Y]: [finding]
- Decoder promotes: [tokens]
- Decoder suppresses: [tokens]

### Weapon Profile
- Core (25-75%): [weapons with %]
- Flanderization (90%+): [if different]
- Role: [from weapon-vibes]

### Label Justification
[Why this label captures the concept]
```

## Label Storage

After consolidating results, update `/mnt/e/mechinterp_runs/labels/consolidated_ultra.json`.

## Examples
- `/investigate feature 6235` - Single feature investigation
- `/investigate features 1819, 552, 14964` - Parallel subagent investigation
- `/investigate validate labels in docs/stamper_null_feature_hits.md`

# Preprocessing Pipeline

The preprocessing pipeline transforms raw Splatoon 3 match data from stat.ink into tokenized training data. It's designed to handle the quirks of gear data while creating a balanced dataset that biases toward optimal builds.

## Pipeline Overview

```
Raw stat.ink data
       ↓
Extract individual player records
       ↓
Add weapon IDs and build hashes
       ↓
Sample and balance the data
       ↓
Convert abilities to tokens
       ↓
Clean up and partition
```

## Step by Step

### 1. Data Acquisition

Downloads match data from stat.ink's API. Each match has 8 players (4v4), with full gear loadouts and match outcomes.

**Code**: `splatnlp.preprocessing.pull`

### 2. Player Record Extraction

The raw data has matches as rows with columns for each player position (A1-A4, B1-B4). This step pivots that into individual player rows.

Each row becomes: one player's gear loadout + whether their team won.

**Code**: `splatnlp.preprocessing.create`

### 3. Column Generation

Adds derived columns:
- **weapon_id**: Numeric ID for each weapon (from name mapping)
- **ability_hash**: xxHash128 of the ability JSON string. This uniquely identifies each gear configuration.
- **win**: Boolean for team victory

The ability_hash is important for the next step.

**Code**: `splatnlp.preprocessing.create.generate_maps()`

### 4. Stratified Sampling

This is where the magic happens for dataset balance.

**Max entries per build**: Caps each unique gear configuration (by ability_hash) to about 100 entries. Without this, popular meta builds would dominate the dataset and the model would just memorize them.

**Winrate balancing**: Adjusts the win/loss ratio to a target (default 60% win). The idea is to bias the training data toward "good" builds. If you only trained on 50/50 data, the model learns what people use. Training on 60% wins teaches it what works.

How it works:
- If actual win rate < target: undersample losses
- If actual win rate > target: undersample wins

**Fraction sampling**: Takes a random fraction (default 10%) of the final data. Keeps training tractable.

**Code**: `splatnlp.preprocessing.sample`

### 5. Ability Tokenization

This is the critical step that converts continuous ability point (AP) values into discrete tokens.

**AP Normalization**: Raw AP values get multiplied by 10. In Splatoon 3:
- 1 main ability = 10 AP
- 1 sub ability = 3 AP
- Max AP per ability = 57 (3 mains + 9 subs of same ability)
- Total AP per player = 57 (3 mains × 10 + 9 subs × 3 = 57)

**Filtering**: Only keeps rows where total AP = 57. Incomplete gear sets are dropped.

**Bucketing**: Converts AP values to tokens using thresholds.

For main-only abilities (Ninja Squid, Stealth Jump, etc.):
- Binary: either present or not
- Token: `"ability_name"` (e.g., `"stealth_jump"`)

For standard abilities (stackable ones):
- Bucketed by AP thresholds: [3, 6, 12, 15, 21, 29, 38, 51, 57]
- Token: `"ability_name_threshold"` (e.g., `"swim_speed_up_12"`)

A player with 12 AP of Swim Speed Up gets the token `swim_speed_up_12`. One with 15 AP gets `swim_speed_up_15`. And so on.

The thresholds aren't arbitrary. They correspond to meaningful breakpoints in the game mechanics:
- 3 AP = 1 sub
- 6 AP = 2 subs
- 12 AP = 1 main + partial subs
- etc.

**Output**: The `ability_tags` column contains space-separated tokens like:
```
"stealth_jump swim_speed_up_12 ink_saver_main_6 quick_respawn_3"
```

**Code**: `splatnlp.preprocessing.process`

### 6. Column Cleanup

Drops intermediate columns (player stats, raw JSON, etc.). Keeps:
- ability_tags (tokenized abilities)
- weapon_id
- Game metadata (patch version, timestamp, etc.)

**Code**: `splatnlp.preprocessing.remove`

### 7. Storage

Final output is partitioned by weapon and saved as compressed Parquet files. Snappy compression keeps memory usage reasonable.

The main CLI outputs a single `weapon_partitioned.csv` with all the tokenized data.

## Why Bucket Instead of Raw AP?

You might wonder why we don't just feed raw AP values to the model. A few reasons:

1. **Tokenization enables embedding**: The model learns separate embeddings for different AP levels. "12 AP of Swim Speed" means something different than "3 AP of Swim Speed" strategically.

2. **Discretization matches the game**: Players don't think in continuous AP. They think "I have 1 main and 1 sub of Swim Speed" (13 AP). Buckets align with how players reason about builds.

3. **Reduces vocabulary size**: Instead of predicting exact AP (0-57 for each ability), the model predicts which threshold buckets are reached. Much smaller output space.

4. **Enables set completion**: The model can predict "add swim_speed_up_12" rather than trying to predict exact numerical values.

## Why Bias Toward Wins?

Training on balanced 50/50 win/loss data teaches the model what gear people actually use. But people aren't always optimal. By oversampling winning builds (60%), we bias the model toward configurations that empirically perform better.

This is intentional. The goal isn't to model player behavior, it's to model optimal builds.

## Memory Efficiency

The pipeline uses PyArrow for partitioning, which keeps memory usage bounded even with millions of records. Each weapon gets its own partition file, so processing happens incrementally rather than all at once.

## Running the Pipeline

```bash
python -m splatnlp.preprocessing.pipeline --base_path data/ --persist
```

This downloads data, runs all preprocessing steps, and saves the final tokenized CSV.

## Code Location

Main pipeline: `src/splatnlp/preprocessing/pipeline.py`

Key modules:
- `pull.py`: Data acquisition
- `create.py`: Record extraction and column generation
- `sample.py`: Stratified sampling
- `process.py`: Ability tokenization
- `remove.py`: Column cleanup
- `transform.py`: Pipeline orchestration

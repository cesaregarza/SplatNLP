# TF-IDF in the SAE Dashboard

This document explains how TF-IDF (Term Frequency-Inverse Document Frequency) is computed and used in the SplatNLP dashboard to identify characteristic ability tokens for each SAE feature.

## Overview

TF-IDF helps identify which ability tokens are **uniquely characteristic** of a feature's top activations, rather than just frequently occurring across all builds. A token that appears in many builds globally will have a lower IDF, while a token that's rare globally but common in the feature's top activations will have a high TF-IDF score.

## Mathematical Definition

### IDF (Inverse Document Frequency)

Computed once across the entire dataset:

```
IDF(token) = log(N + 1) - log(count(token) + 1)
```

Where:
- `N` = total number of builds in the dataset
- `count(token)` = number of builds containing that token

This is computed in `src/splatnlp/dashboard/utils/tfidf.py`:

```python
def compute_idf(df: pl.DataFrame) -> pl.DataFrame:
    LOG_N = np.log(len(df) + 1)
    return (
        df.explode("ability_input_tokens")
        .group_by("ability_input_tokens")
        .agg(pl.col("index").count().alias("count"))
        .with_columns(
            (pl.lit(LOG_N).sub(pl.col("count").add(1).log())).alias("idf")
        )
        .select(["ability_input_tokens", "idf"])
    )
```

### TF (Term Frequency)

Computed per-feature on the top activating examples:

```
TF(token) = count of token in feature's top activations
```

### TF-IDF Score

```
TF-IDF(token) = TF(token) × IDF(token)
```

Computed in `src/splatnlp/dashboard/utils/tfidf.py`:

```python
def compute_tf_idf(idf: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.explode("ability_input_tokens")
        .sort(["index", "ability_input_tokens"])
        .group_by("ability_input_tokens")
        .agg(pl.col("index").count().alias("tf"))
        .join(idf, on="ability_input_tokens", how="left")
        .with_columns(pl.col("tf").mul(pl.col("idf")).alias("tf_idf"))
        .sort("tf_idf", descending=True)
    )
```

## Data Flow

### 1. IDF Precomputation

For the efficient database (`EfficientFSDatabase`), IDF is precomputed once:

```bash
python -m splatnlp.dashboard.commands.precompute.idf \
    --data-dir /path/to/activations \
    --output-path data/precomputed/idf.parquet
```

This processes all batches and computes global IDF values, stored as a Parquet file.

For the standard database (`FSDatabase`), IDF is computed on initialization from `analysis_df`.

### 2. Per-Feature TF-IDF Analysis

When a feature is selected in the dashboard:

1. **Load top activations**: Get the top N activating builds for the feature
2. **Compute TF**: Count token occurrences in these top activations
3. **Join with IDF**: Multiply TF by precomputed IDF
4. **Rank by TF-IDF**: Sort tokens by score to find characteristic tokens

This happens in `IntervalsGridRenderer.analyze()`:

```python
def analyze(self, activations_df, feature_labels_manager, selected_feature_id):
    # Compute TF-IDF
    tf_idf = compute_tf_idf(self.idf, activations_df)

    # Get top tokens with their scores
    top_tokens = (
        tf_idf.sort("tf_idf", descending=True)
        .head(MAX_TFIDF_FEATURES)  # Default: 10
        ...
    )
```

## Data Structures

### Input: `analysis_df`

The base dataframe containing all builds:

| Column | Type | Description |
|--------|------|-------------|
| `index` | int | Row index (unique build identifier) |
| `ability_input_tokens` | list[int] | List of ability token IDs in the build |
| `weapon_id_token` | int | Weapon token ID |

### Input: `activations_df`

Top activations for a specific feature (subset of `analysis_df` with activation values):

| Column | Type | Description |
|--------|------|-------------|
| `index` | int | Build index |
| `ability_input_tokens` | list[int] | Ability tokens |
| `weapon_id_token` | int | Weapon token |
| `activation` | float | SAE feature activation value |

### Output: TF-IDF DataFrame

| Column | Type | Description |
|--------|------|-------------|
| `ability_input_tokens` | int | Token ID |
| `tf` | int | Term frequency in top activations |
| `idf` | float | Inverse document frequency |
| `tf_idf` | float | Final TF-IDF score |

## Dashboard Display

The TF-IDF results are displayed in the **Feature Stats** panel:

- **Top 10 TF-IDF tokens** shown with their scores
- Tokens are converted from IDs to human-readable names (e.g., `swim_speed_up_21`)
- High TF-IDF tokens are highlighted in the intervals grid examples

### Example Output

```
Feature: Special Charge Up Enthusiast

Top TF-IDF Abilities:
1. special_charge_up_29  (0.847)
2. special_charge_up_21  (0.723)
3. special_power_up_15   (0.412)
...
```

## Interpretation

- **High TF-IDF**: Token is characteristic of this feature (frequent in top activations, rare globally)
- **Low TF-IDF**: Token is either rare in top activations OR common across all builds

### Example Interpretation

If `special_charge_up_29` has high TF-IDF for Feature #42:
- It appears frequently in Feature #42's top activations
- It doesn't appear as frequently in the global dataset
- Feature #42 likely "detects" or "responds to" high levels of Special Charge Up

## Configuration

Key constants in `intervals_grid_component.py`:

```python
MAX_TFIDF_FEATURES = 10      # Number of top tokens to display
TOP_BINS_FOR_ANALYSIS = 8    # Number of activation bins to analyze
```

## Files

| File | Purpose |
|------|---------|
| `src/splatnlp/dashboard/utils/tfidf.py` | Core TF-IDF computation functions |
| `src/splatnlp/dashboard/fs_database.py` | IDF computation on database init |
| `src/splatnlp/dashboard/efficient_fs_database.py` | Loads precomputed IDF |
| `src/splatnlp/dashboard/components/intervals_grid_component.py` | TF-IDF analysis and display |
| `src/splatnlp/dashboard/commands/precompute/idf.py` | IDF precomputation CLI |

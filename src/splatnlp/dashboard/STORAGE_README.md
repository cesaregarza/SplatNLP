# SAE Activation Storage & Mechanistic Interpretability Pipeline

End-to-end guide for training SAEs, generating activations, and setting up the mechinterp infrastructure.

## Prerequisites

Before starting, ensure you have:

```bash
# 1. Trained base model checkpoint
saved_models/your_model/model.pth
saved_models/your_model/vocab.json
saved_models/your_model/weapon_vocab.json

# 2. Tokenized dataset
data/tokenized_data.csv

# 3. Sufficient disk space (~20-50GB per model depending on dataset size)
# 4. Poetry environment with dependencies installed
poetry install --with dev
```

---

## Full Pipeline Overview

```
1. Train SAE on model activations
   ↓
2. Generate activations for entire dataset
   ↓
3. Convert to efficient storage format (Zarr/Parquet)
   ↓
4. [Optional] Create fast-access format (transposed or sparse)
   ↓
5. Precompute analysis data (histograms, IDF, influences)
   ↓
6. Start activation server for mechinterp queries
   ↓
7. Launch interactive dashboard (Plotly Dash)
   ↓
8. Investigate features with CLI tools
```

---

## Step 1: Train SAE

Train a sparse autoencoder on the model's internal representations.

```bash
poetry run python -m splatnlp.monosemantic_sae.cli \
    --model-checkpoint saved_models/your_model/model.pth \
    --data-csv data/tokenized_data.csv \
    --vocab-path saved_models/your_model/vocab.json \
    --weapon-vocab-path saved_models/your_model/weapon_vocab.json \
    --save-dir sae_runs/my_run \
    --hook-target masked_mean \
    --expansion-factor 48 \
    --l1-coeff 3e-4 \
    --epochs 5 \
    --sae-batch-size 4096
```

**Key parameters:**
- `--hook-target`: Where to capture activations (`masked_mean` for pooled repr)
- `--expansion-factor`: SAE hidden size = input_dim × expansion_factor
- `--l1-coeff`: Sparsity penalty (higher = sparser features)

**Output:** `sae_runs/my_run/sae_model_final.pth`

---

## Step 2: Generate Activations

Run the trained SAE over the full dataset to collect feature activations.

```bash
poetry run python -m splatnlp.dashboard.commands.generate_activations_cmd \
    --model-checkpoint saved_models/your_model/model.pth \
    --sae-checkpoint sae_runs/my_run/sae_model_final.pth \
    --data-csv data/tokenized_data.csv \
    --vocab-path saved_models/your_model/vocab.json \
    --weapon-vocab-path saved_models/your_model/weapon_vocab.json \
    --output-dir /mnt/e/activations_raw \
    --batch-size 1024
```

**Output:** Raw activation files in joblib/numpy format

---

## Step 3: Convert to Efficient Format

Convert raw activations to optimized Zarr + Parquet storage.

```bash
poetry run python -m splatnlp.dashboard.commands.convert_to_efficient_cmd \
    --input-dir /mnt/e/activations_raw \
    --output-dir /mnt/e/activations_efficient \
    --batch-size 500000
```

**Output structure:**
```
activations_efficient/
├── activations/
│   ├── batch_0000.zarr    # [samples, features] per batch
│   ├── batch_0001.zarr
│   └── ...
├── metadata/
│   ├── batch_0000.parquet # sample_id, weapon_id, abilities, global_index
│   └── ...
├── embeddings/
│   └── ...
└── conversion_metadata.json
```

---

## Step 4: Create Fast-Access Format

Choose ONE of these formats for O(1) feature access:

### Option A: Transposed Zarr (recommended)

Reorganizes data from `[samples, features]` to `[features, samples]`.

```bash
poetry run python -m splatnlp.dashboard.commands.precompute.transpose \
    --source-dir /mnt/e/activations_efficient \
    --target-dir /mnt/e/activations_transposed
```

**Output:** Single `transposed_activations.zarr` with shape `[n_features, n_samples]`

### Option B: Sparse Per-Feature Files

Pre-extract non-zero activations per feature.

```bash
poetry run python -m splatnlp.dashboard.commands.precompute.sparse_extract \
    --source-dir /mnt/e/activations_efficient \
    --output-dir /mnt/e/neuron_acts
```

**Output structure:**
```
neuron_acts/
├── neuron_0000/
│   ├── acts.npy    # float32 non-zero values
│   └── idxs.npy    # int64 sample indices
├── neuron_0001/
└── ...
```

---

## Step 5: Precompute Analysis Data

### 5a. Histograms & Stats

```bash
poetry run python -m splatnlp.dashboard.commands.precompute.histograms_fast \
    --data-dir /mnt/e/activations_efficient \
    --output-dir /mnt/e/activations_efficient/precomputed \
    --method vectorized
```

### 5b. IDF Scores

```bash
poetry run python -m splatnlp.dashboard.commands.precompute.idf \
    --data-dir /mnt/e/activations_efficient \
    --output-path /mnt/e/activations_efficient/precomputed/idf.parquet
```

### 5c. Feature Influences (decoder analysis)

```bash
poetry run python -m splatnlp.dashboard.commands.precompute.influences \
    --model-checkpoint saved_models/your_model/model.pth \
    --sae-checkpoint sae_runs/my_run/sae_model_final.pth \
    --output-path /mnt/e/activations_efficient/precomputed/influences.parquet \
    --top-k 100
```

**Precomputed output:**
```
activations_efficient/precomputed/
├── histograms.pkl
├── feature_stats.pkl
├── idf.parquet
└── influences.parquet
```

---

## Step 6: Start Activation Server

```bash
ACTIVATION_DATA_DIR=/mnt/e/activations_efficient \
ACTIVATION_TRANSPOSED_DIR=/mnt/e/activations_transposed \
poetry run uvicorn splatnlp.mechinterp.server.activation_server:app \
    --host 127.0.0.1 --port 8765
```

**Endpoints:**
- `GET /health` - Server status
- `GET /activations/{feature_id}` - Feature activations (JSON)
- `GET /activations/{feature_id}/arrow` - Feature activations (Arrow, faster)
- `GET /context/vocab` - Token vocabulary

---

## Step 7: Launch Interactive Dashboard (Plotly Dash)

```bash
poetry run python -m splatnlp.dashboard.cli \
    --use-efficient \
    --data-dir /mnt/e/activations_efficient \
    --examples-dir /mnt/e/activations_efficient/examples
```

Opens at `http://127.0.0.1:8050` with:
- Feature activation histograms
- Top activating examples
- Token/weapon breakdowns
- Interactive exploration

---

## Step 8: Investigate Features

### Option A: Claude Code Skills (Recommended)

Use the built-in mechinterp skills for AI-assisted investigation:

```bash
# In Claude Code, use the /investigate slash command
/investigate 845 full

# Or invoke skills directly:
# - mechinterp-overview: Quick feature summary
# - mechinterp-runner: Run structured experiments
# - mechinterp-investigator: Full investigation workflow
# - mechinterp-labeler: Manage feature labels
# - mechinterp-decoder: Analyze decoder weights
```

The skills automatically connect to the activation server and provide interpreted results.

### Option B: CLI Tools

For scripting or manual analysis, use the CLI tools:

### Feature Overview

Get a quick summary of what a feature responds to:

```bash
poetry run python -m splatnlp.mechinterp.cli.overview_cli \
    --feature-id 845 --model full --max-examples 2000

# With extended analyses
poetry run python -m splatnlp.mechinterp.cli.overview_cli \
    --feature-id 845 --model full --all
```

### Run Experiments

Execute structured experiments (family sweeps, heatmaps, weapon analysis):

```bash
# Family sweep - test how an ability family affects activation
poetry run python -m splatnlp.mechinterp.cli.runner_cli family-sweep \
    --feature-id 845 --family swim_speed_up --model full

# Interaction heatmap between two ability families
poetry run python -m splatnlp.mechinterp.cli.runner_cli heatmap \
    --feature-id 845 --family-x quick_respawn --family-y comeback

# Top weapons analysis
poetry run python -m splatnlp.mechinterp.cli.runner_cli weapon-sweep \
    --feature-id 845 --model full --top-k 20
```

### Decoder Analysis

Analyze what output tokens a feature influences:

```bash
poetry run python -m splatnlp.mechinterp.cli.decoder_cli \
    --feature-id 845 --model full --top-k 20
```

### Label Management

Save/load feature labels:

```bash
# Labels are stored in JSON format
# Default location: /mnt/e/mechinterp_runs/labels/consolidated_{model}.json

poetry run python -m splatnlp.mechinterp.cli.labeler_cli \
    --feature-id 845 --model full --action show
```

---

## Verification Steps

After each pipeline step, verify it worked:

```bash
# After Step 3 (convert to efficient)
ls -la /mnt/e/activations_efficient/activations/  # Should show batch_*.zarr
cat /mnt/e/activations_efficient/conversion_metadata.json

# After Step 4 (transpose)
poetry run python -c "
import zarr
z = zarr.open_array('/mnt/e/activations_transposed/transposed_activations.zarr', 'r')
print(f'Shape: {z.shape}')  # Should be [n_features, n_samples]
"

# After Step 5 (precompute)
ls -la /mnt/e/activations_efficient/precomputed/  # Should show .pkl and .parquet files

# After Step 6 (server)
curl http://127.0.0.1:8765/health  # Should return healthy status
```

---

## Storage Formats Reference

| Format | Layout | Access | Use Case |
|--------|--------|--------|----------|
| **Batch Zarr** | `[samples, features]` × N batches | O(n_batches) | Base storage |
| **Transposed Zarr** | `[features, samples]` | O(1) | Feature queries |
| **Neuron Acts** | Sparse `.npy` per feature | O(1) | Pre-filtered sparse |

### Performance

| Backend | ~100K activations | ~500K activations |
|---------|-------------------|-------------------|
| Transposed Zarr | ~0.5s | ~0.8s |
| Neuron Acts | ~0.6s | ~0.8s |
| Batch Scan | 2+ minutes | 2+ minutes |

---

## Using StorageConfig

```python
from splatnlp.dashboard.efficient_fs_database import EfficientFSDatabase
from splatnlp.dashboard.storage_config import StorageConfig

# Explicit configuration
config = StorageConfig(
    model_type="my_model",
    data_dir=Path("/mnt/e/activations_efficient"),
    examples_dir=Path("/mnt/e/activations_efficient/examples"),
    transposed_dir=Path("/mnt/e/activations_transposed"),
    neuron_acts_dir=None,  # or Path to sparse format
)
db = EfficientFSDatabase(config=config)

# Or with individual parameters
db = EfficientFSDatabase(
    data_dir="/mnt/e/activations_efficient",
    transposed_dir="/mnt/e/activations_transposed",
)
```

---

## Adding New Storage Backends

1. Implement `ActivationLoader` protocol in `storage_backends.py`:

```python
class MyLoader:
    @property
    def is_available(self) -> bool:
        """Return True if this backend can be used."""
        ...

    def load(self, feature_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (indices, activations) or None if unavailable."""
        ...
```

2. Initialize in `EfficientFSDatabase._init_storage_backends()`
3. Add fast path in `get_all_feature_activations_for_pagerank()`

---

## Troubleshooting

**Slow feature queries:** Ensure transposed or neuron_acts format exists

**Out of memory during transpose:** Reduce chunk size or process in stages

**Missing precomputed data:** Check `{data_dir}/precomputed/` exists

**Server can't find data:** Verify env vars point to correct directories

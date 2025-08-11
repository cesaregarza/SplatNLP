# Precompute Module

This module contains utilities for precomputing various data needed by the SplatNLP dashboard to optimize performance and enable interactive exploration of SAE features.

## Overview

The precompute module is designed to handle expensive computations offline, allowing the dashboard to load and display results instantly. It supports both the **Full** model (2,048 features) and **Ultra** model (24,576 features) with different storage backends.

## Main Components

### 1. `unified.py` - Orchestrator
**Purpose**: Main entry point that coordinates all precomputation tasks.

**Key Features**:
- Automatically detects model type (Full vs Ultra)
- Manages different data formats (filesystem vs Zarr)
- Orchestrates influence computation and histogram generation
- Provides a single interface for all precomputation needs

**Usage**:
```bash
python precompute_all.py --model-type ultra --top-k 100
```

### 2. `influences.py` - Feature Influence Analysis
**Purpose**: Computes how each SAE feature influences output token predictions.

**What it does**:
- Calculates influence matrix: V × F (vocab size × features)
- Identifies top positive/negative token associations for each feature
- Helps understand what each feature "means" in terms of output

**Output**: Parquet file with top-k influenced tokens per feature

**Usage**:
```bash
python precompute_influences.py \
    --model-checkpoint saved_models/dataset_v0_2_super/clean_slate.pth \
    --sae-checkpoint sae_runs/run_20250704_191557/sae_model_final.pth \
    --top-k 100
```

### 3. `histograms_parallel.py` - Parallel Histogram Computation
**Purpose**: Computes activation histograms using multiprocessing for speed.

**Key Features**:
- Processes features in parallel chunks
- Two methods: parallel (multiprocessing) or vectorized
- Handles sparse activations efficiently
- Computes statistics (min, max, mean, std, sparsity)

**Best for**: Ultra model with many features (24,576)

**Usage**:
```bash
python precompute_histograms.py \
    --data-dir /mnt/e/activations_ultra_efficient \
    --method parallel \
    --n-workers 16
```

### 4. `histograms_fast.py` - Single-Pass Histogram Computation
**Purpose**: Ultra-fast histogram computation using a single pass through data.

**Key Features**:
- Reads each batch file ONCE
- Updates all histograms simultaneously in memory
- Fully vectorized operations
- Memory-efficient streaming approach

**Best for**: When you have enough RAM to hold histogram accumulators

**Usage**:
```bash
python -m splatnlp.dashboard.commands.precompute.histograms_fast \
    --data-dir /mnt/e/activations_ultra_efficient \
    --method vectorized
```

### 5. `transpose.py` - Data Transposition for Feature Access
**Purpose**: Reorganizes activation data from (samples × features) to (features × samples).

**Why it's needed**:
- Original format optimized for sample-wise access
- Dashboard needs feature-wise access for histograms
- Transposition enables direct feature slicing

**Process**:
1. Creates transposed Zarr array
2. Enables O(1) feature access
3. Makes histogram computation 100x faster

**Usage**:
```bash
python transpose_activations.py \
    --source-dir /mnt/e/activations_ultra_efficient \
    --target-dir /mnt/e/activations_transposed
```

### 6. `idf.py` - Inverse Document Frequency
**Purpose**: Computes IDF scores for ability tokens across the dataset.

**What it does**:
- Measures token rarity/importance
- Helps identify distinctive vs common abilities
- Used for TF-IDF analysis in dashboard

**Output**: Parquet file with IDF scores per token

**Usage**:
```bash
python precompute_idf.py \
    --data-dir /mnt/e/activations_ultra_efficient \
    --output-path data/precomputed_ultra/idf.parquet
```

## Data Flow

```
1. Generate Activations (generate_activations_cmd.py)
   ↓
2. Convert to Efficient Format (convert_to_efficient_cmd.py) 
   ↓
3. Precompute Data:
   ├─ Compute Influences (influences.py)
   ├─ Compute Histograms (histograms_*.py)
   ├─ Compute IDF (idf.py)
   └─ [Optional] Transpose for faster access (transpose.py)
   ↓
4. Dashboard loads precomputed data
```

## Model Configurations

### Full Model
- **Features**: 2,048
- **Storage**: Filesystem (.npy files)
- **Location**: `/mnt/e/activations2/outputs/neuron_acts/`
- **Best for**: Detailed analysis, smaller datasets

### Ultra Model
- **Features**: 24,576
- **Storage**: Zarr arrays (compressed)
- **Location**: `/mnt/e/activations_ultra_efficient/`
- **Best for**: Large-scale analysis, production use

## Performance Tips

1. **Use parallel processing**: The parallel histogram computation can use multiple CPU cores
2. **Transpose first**: For repeated histogram computations, transpose the data once
3. **Precompute everything**: Run the unified script before launching dashboard
4. **Cache results**: Precomputed data is cached in `data/precomputed_{model_type}/`

## Output Files

After running precomputation, you'll find:

```
data/precomputed_ultra/
├── influences.parquet      # Feature → Token influences
├── histograms.pkl          # Activation histograms
├── feature_stats.pkl       # Statistical summaries
└── idf.parquet            # Token IDF scores
```

## Quick Start

For Ultra model:
```bash
# Precompute everything
python precompute_all.py --model-type ultra

# Launch dashboard with precomputed data
./run_dashboard.sh ultra
```

For Full model:
```bash
# Precompute everything
python precompute_all.py --model-type full

# Launch dashboard
./run_dashboard.sh full
```

## Troubleshooting

**Out of Memory**: 
- Use `histograms_parallel.py` with smaller chunk sizes
- Process fewer features at once with `--test` flag

**Slow Computation**:
- Increase workers: `--n-workers 32`
- Use vectorized method: `--method vectorized`
- Consider transposing data first

**Missing Data**:
- Ensure activations are generated first
- Check paths in MODEL_CONFIGS dictionary
- Verify Zarr/Parquet files are not corrupted

## Development

To add a new precomputation:

1. Create new module in this directory
2. Follow naming convention: `{task}.py`
3. Add to `__init__.py` exports
4. Integrate with `unified.py` if needed
5. Update this README

## Dependencies

- `numpy`: Numerical operations
- `torch`: Model loading and tensor operations
- `zarr`: Efficient array storage
- `pandas`/`polars`: Data manipulation
- `tqdm`: Progress bars
- `multiprocessing`: Parallel computation
# Experiment Tracking

The project uses Weights & Biases (WandB) for experiment tracking, with support for Bayesian hyperparameter optimization via sweeps.

## WandB Integration

### Basic Usage

Enable WandB logging with CLI flags:
```bash
python -m splatnlp.monosemantic_sae.cli \
    --wandb-log \
    --wandb-project splatnlp-sae \
    --wandb-entity your-team \
    ...
```

### Initialization

```python
if args.wandb_log:
    run_name = os.environ.get("WANDB_RUN_ID", f"local_{generate_id()}")
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=run_name,
    )
```

All CLI arguments are automatically logged to the run config. This makes experiment comparison straightforward.

### What Gets Tracked

**Training Metrics** (logged every N steps):
- `train/loss`: Total loss
- `train/mse_loss`: Reconstruction error
- `train/l1_loss`: Sparsity penalty
- `train/kl_loss`: Usage regularization
- `train/mean_sparsity`: Average L0 sparsity
- `train/mean_magnitude`: Average activation magnitude
- `train/dead_neurons`: Count of inactive features
- `train/dead_neuron_pct`: Percentage dead
- `train/lr`: Current learning rate
- `train/buffer_fill_pct`: Activation buffer utilization

**Validation Metrics**:
- `val/miracle_distance`: Custom composite metric
- `val/mse_loss`: Reconstruction quality
- `val/mean_sparsity`: Feature sparsity
- `val/dead_neurons`: Dead neuron count

**Model Tracking**:
```python
wandb.watch(sae_model, log="all", log_freq=100)
```

Automatically tracks gradients and weights. Useful for diagnosing training issues (vanishing gradients, weight collapse, etc.).

## Hyperparameter Sweeps

The project uses Bayesian optimization for hyperparameter tuning via WandB sweeps.

### Sweep Configuration

See `configs/sae-sweep.yaml`:
```yaml
program: scripts/train_sae.sh
method: bayes
metric:
  name: val_miracle_distance
  goal: minimize
early_terminate:
  type: hyperband
  min_iter: 3
  eta: 3
  s: 2
```

**Method**: Bayesian optimization (not grid search). Uses a Gaussian Process to model the objective function and intelligently select next hyperparameters. More sample-efficient than random search.

**Metric**: Minimizes validation "miracle distance" (see Evaluation section).

**Early Termination**: Hyperband algorithm stops poorly performing runs early. Saves compute by focusing resources on promising configurations.

### Parameters Being Swept

```yaml
parameters:
  expansion-factor:
    distribution: uniform
    min: 1.0
    max: 12.0
  l1-coeff:
    distribution: log_uniform_values
    min: 5e-7
    max: 1e-3
  usage-coeff:
    distribution: uniform
    min: 0.1
    max: 3.0
  lr:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-4
  # ... more parameters
```

Log-uniform distributions for learning rates and coefficients (spans orders of magnitude). Uniform for bounded continuous values.

### Running a Sweep

1. Create the sweep:
```bash
wandb sweep configs/sae-sweep.yaml
```

2. Launch agents:
```bash
wandb agent your-entity/splatnlp-sae/sweep_id
```

Each agent pulls hyperparameter configurations from the sweep controller, runs training, and reports metrics back. Multiple agents can run in parallel across machines.

### Sweep Output Organization

Each run saves artifacts to a unique directory:
```bash
save_dir=sae_runs/${wandb.run.id}
```

This prevents checkpoint collisions between parallel runs. The WandB run ID ties artifacts back to the experiment dashboard.

## Miracle Distance Metric

A custom composite metric for SAE quality that balances multiple objectives:

```python
def compute_miracle_distance(mse_loss, mean_sparsity, dead_pct, config):
    mse_ratio = mse_loss / config.target_mse
    sparsity_ratio = mean_sparsity / config.target_sparsity
    dead_ratio = dead_pct / config.target_dead_pct

    weighted_sum = (
        config.mse_weight * log(mse_ratio) ** 2
        + config.sparsity_weight * log(sparsity_ratio) ** 2
        + config.dead_weight * log(dead_ratio) ** 2
    )
    return sqrt(weighted_sum)
```

**Why this metric?**
- SAE training has competing objectives: low reconstruction error vs high sparsity
- Dead neurons are bad, but some are expected
- Log ratios normalize across scales (MSE might be 0.01, sparsity might be 10.0)
- Squared terms penalize deviations from targets symmetrically

Targets are configurable in `SAEConfig`:
```python
target_mse: float = 0.015
target_sparsity: float = 15.0
target_dead_pct: float = 5.0
```

The optimizer minimizes distance to this "ideal" operating point.

## Reproducibility

### Configuration Logging

All training runs save their full configuration:
```python
with open(save_dir / "training_args.json", "wb") as f:
    orjson.dumps(vars(args), f, option=orjson.OPT_INDENT_2)
```

To reproduce a run, load the JSON and pass the same arguments.

### Random Seed Handling

Tests set seeds explicitly:
```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
```

For production training, seeds aren't fixed by default (for exploring different initializations), but can be added via CLI flags.

### Distributed Training Reproducibility

In DDP mode, each epoch re-seeds the sampler:
```python
if isinstance(dataloader.sampler, DistributedSampler):
    dataloader.sampler.set_epoch(epoch)
```

This ensures each GPU sees different data each epoch, but the overall data distribution is consistent across runs with the same seed.

## Local Logging

Even without WandB, all metrics are logged locally:

```python
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(save_dir / "training.log"),
    ]
)
```

Progress bars via `tqdm` show real-time metrics:
```
Epoch 5: 100%|██████| 1000/1000 [05:23<00:00, loss=0.0124, sparsity=12.3]
```

## Analyzing Experiments

WandB provides:
- **Run comparison**: Side-by-side metric plots
- **Hyperparameter importance**: Which parameters matter most
- **Parallel coordinates**: Visualize parameter interactions
- **Artifact versioning**: Track model checkpoints

The sweep dashboard shows:
- Best performing configurations
- Parameter correlations with performance
- Early termination decisions

This infrastructure makes it easy to systematically explore the hyperparameter space and identify optimal configurations.

## Code Location

WandB integration: `src/splatnlp/monosemantic_sae/cli.py`
Sweep config: `configs/sae-sweep.yaml`
Metric computation: `src/splatnlp/monosemantic_sae/sae_training/evaluate.py`
Logging setup: `src/splatnlp/monosemantic_sae/cli.py` (`_setup_logging`)

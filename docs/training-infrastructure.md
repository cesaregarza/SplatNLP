# Training Infrastructure

This project implements production-grade training infrastructure with distributed training, mixed precision, and proper memory management.

## Distributed Training (DDP)

The model supports multi-GPU training via PyTorch's DistributedDataParallel (DDP).

### How It Works

DDP replicates the model on each GPU and synchronizes gradients after each backward pass. Each process handles a different subset of the data.

**Initialization** (`src/splatnlp/model/cli.py`):
```python
torch.distributed.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
```

NCCL backend is used for efficient GPU-to-GPU communication on NVIDIA hardware.

**Model Wrapping** (`src/splatnlp/model/training_loop.py`):
```python
model = DistributedDataParallel(model, device_ids=[local_rank])
```

**Data Distribution**:
- `DistributedSampler` splits the dataset across processes
- Each epoch calls `sampler.set_epoch(epoch)` to re-shuffle per process
- Prevents data overlap between GPUs

**Metric Synchronization**:
Metrics are computed locally then reduced across processes:
```python
torch.distributed.barrier()  # Synchronize
torch.distributed.all_reduce(metric_tensor)  # Sum across ranks
metric_tensor /= world_size  # Average
```

This ensures consistent evaluation across all GPUs.

### Running DDP Training

```bash
torchrun --nproc_per_node=4 -m splatnlp.model.cli \
    --data_path data/tokenized.csv \
    --output_dir ./checkpoints \
    ...
```

Non-zero ranks have stdout suppressed to avoid duplicate logs.

## Mixed Precision Training

Uses automatic mixed precision (AMP) with bfloat16 to reduce memory usage and speed up training.

**Implementation** (`src/splatnlp/model/training_loop.py`):
```python
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    logits = model(abilities, weapons, key_padding_mask)
    loss = criterion(logits, targets)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
scaler.step(optimizer)
scaler.update()
```

**Why bfloat16?**
- Same dynamic range as float32 (8-bit exponent)
- Lower precision (7-bit mantissa vs 23-bit)
- Native support on modern GPUs (A100, H100)
- No need for loss scaling in most cases

The `GradScaler` automatically handles underflow issues that can occur with reduced precision gradients.

## Memory Optimization

Several techniques reduce memory footprint:

**Efficient Data Loading**:
```python
DataLoader(
    dataset,
    batch_size=batch_size,
    pin_memory=True,          # Faster CPU to GPU transfer
    persistent_workers=True,   # Keep workers alive between epochs
    non_blocking=True,        # Async data transfers
    num_workers=16            # Parallel data loading
)
```

`pin_memory=True` allocates data in page-locked memory, enabling faster DMA transfers to GPU.

**Gradient Clipping**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Prevents exploding gradients and stabilizes training. Applied after `scaler.unscale_()` in mixed precision mode.

**SAE Activation Buffer**:
The SAE training uses a circular buffer to store activations efficiently:
```python
class ActivationBuffer:
    def __init__(self, size, dim, device="cuda"):
        self.buffer = torch.zeros(size, dim, device=device)
        self.position = 0
```

Instead of storing all activations (memory explosion), it maintains a fixed-size window. New activations overwrite old ones in a ring buffer pattern.

## Learning Rate Scheduling

Two schedulers are implemented:

**ReduceLROnPlateau** (main model):
```python
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
scheduler.step(val_loss)
```
Reduces LR by half when validation loss plateaus for 5 epochs. Adaptive to training dynamics.

**CosineAnnealingLR** (SAE):
```python
T_max = epochs * steps_per_epoch
scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
```
Smoothly decays LR following a cosine curve. Good for finding flat minima.

## Early Stopping

Prevents overfitting by monitoring validation loss:

```python
@dataclass
class EarlyStopping:
    patience: int = 7
    min_delta: float = 0.0
    best_loss: float = float("inf")
    counter: int = 0

    def check(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
```

Training stops if validation loss doesn't improve for `patience` epochs. Saves compute and prevents overfitting.

## Checkpoint Management

**Saving**:
```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "best_val_loss": best_loss,
}, checkpoint_path)
```

**DDP State Dict Conversion**:
DDP wraps models in a module, adding a `module.` prefix to parameter names. The `convert_ddp_state` utility strips this:
```python
def convert_ddp_state(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}
```

This allows loading DDP-trained models in single-GPU inference.

## Configuration via CLI

All hyperparameters are exposed via argparse with sensible defaults:

```bash
python -m splatnlp.model.cli \
    --embedding_dim 32 \
    --hidden_dim 512 \
    --num_layers 3 \
    --learning_rate 1e-4 \
    --batch_size 1024 \
    --use_mixed_precision True \
    --num_workers 16 \
    --dropout 0.3 \
    --num_epochs 20
```

Each parameter has:
- Type validation (int, float, bool)
- Default value
- Help text explaining usage

Boolean flags use `argparse.BooleanOptionalAction` for clean `--flag` / `--no-flag` syntax.

## Structured Logging

Training logs to both stdout and file:
```python
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ]
)
```

JSON serialization of all arguments provides an audit trail:
```python
with open(save_dir / "training_args.json", "w") as f:
    json.dump(vars(args), f, indent=2)
```

Reproducing a run is straightforward: copy the args JSON.

## Code Location

Main training logic: `src/splatnlp/model/training_loop.py`
CLI interface: `src/splatnlp/model/cli.py`
Configuration: `src/splatnlp/model/config.py`
SAE training: `src/splatnlp/monosemantic_sae/sae_training/`

Key functions:
- `train_epoch()`: Single epoch training loop with optional DDP
- `validate()`: Validation pass with metric computation
- `fit()`: Full training loop with early stopping and scheduling
- `convert_ddp_state()`: DDP checkpoint conversion

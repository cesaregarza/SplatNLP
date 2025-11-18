# Data Augmentation

The model uses a masking-based data augmentation strategy to create multiple training instances from each gear set. This increases data diversity and helps the model learn to complete partial builds.

## The Problem

Each player's gear set in the raw data is complete (57 AP total). But at inference time, users provide partial builds and ask the model to suggest completions. If we only trained on complete sets, the model wouldn't know how to handle partial inputs.

## The Solution: Masked Subsets

For each complete gear set in the training data, we generate multiple masked instances by randomly removing abilities. This teaches the model to predict missing abilities from partial context.

## Implementation

### Dataset Structure

The `SetDataset` class generates N instances per gear set:
```python
num_instances_per_set = 5  # Default
uses_null_token = True  # Optional: include empty set baseline
total_instances = num_instances_per_set + 1  # If using null token
```

So a dataset with 10,000 gear sets becomes 60,000 training samples (5 masked + 1 null per set).

### Instance Generation

For each original gear set, the dataset generates:

**Instance 0 (if null token enabled):**
- Input: `[<NULL>]`
- Target: Full original set
- Teaches baseline builds with no context

**Instances 1-N:**
- Input: Random subset of original abilities
- Target: Full original set
- Number of removals chosen probabilistically

### Removal Distribution

The number of abilities to remove is sampled from a skewed triangular distribution:

```python
def weighted_random_removals(max_removals):
    x = np.arange(1, max_removals + 1)
    distribution = x / x.sum()  # Triangular: [1/sum, 2/sum, ..., n/sum]
    distribution = distribution ** skew_factor  # Skew toward higher removals
    distribution /= distribution.sum()  # Renormalize
    return np.random.choice(x, p=distribution)
```

**Why skewed?**
- `skew_factor = 1.2` (default) increases probability of more removals
- We want to see partial builds, not just "remove one ability"
- Higher removals = harder completion task = better generalization

**Example with 10 abilities:**
- Triangular base: [0.018, 0.036, ..., 0.182]
- After skew: [0.009, 0.022, ..., 0.215]
- More likely to remove 7-9 abilities than 1-3

### Masking Process

1. Count abilities in original set: `set_length`
2. Sample number of removals: `num_removals` (1 to set_length-1)
3. Randomly select which abilities to remove
4. Create input tensor from remaining abilities
5. Target tensor is always the full original set

Example:
```
Original: [swim_speed_12, ink_saver_6, ninja_squid]
Removal count: 2
Removed indices: [0, 2]
Input: [ink_saver_6]
Target: [swim_speed_12, ink_saver_6, ninja_squid]
```

The model sees the partial input and learns to predict the full target.

## Why This Works

**Trains on realistic scenarios:**
- Users don't provide random subsets at inference
- But they provide *some* subset
- Seeing many different partial builds teaches the model to generalize

**Increases effective dataset size:**
- 10k gear sets → 60k training samples
- Each sample has different masked context
- Reduces overfitting

**Forces set completion behavior:**
- Input: partial build
- Output: probabilities for all abilities
- Natural fit for multi-label classification

**Handles variable-length inputs:**
- Masked inputs have different lengths
- Padding and attention masks handle this cleanly
- Model learns that sequence length doesn't matter (it's a set)

## Comparison to Alternatives

**Alternative 1: Train on full sets only**
- Problem: Model never sees partial inputs
- At inference: Distribution shift, poor completion quality

**Alternative 2: Random masking à la BERT**
- Problem: BERT-style masking replaces tokens with [MASK]
- Here, we remove tokens entirely (sparse sets)
- Better matches the use case

**Alternative 3: Fixed masking ratios**
- Problem: Always masks 50%, or 30%, etc.
- Our approach: Variable masking (1 to N-1 removals)
- More diverse training signal

## Implementation Details

### Collation

The collate function handles variable-length sequences:
```python
def collate_fn(batch):
    inputs, weapons, targets = zip(*batch)
    # Pad to max length in batch
    padded_inputs = pad_sequence(inputs, padding_value=PAD_ID)
    # Create attention masks
    attention_masks = (padded_inputs != PAD_ID)
    return padded_inputs, weapons, targets, attention_masks
```

Padding is necessary because PyTorch requires fixed-size tensors. The attention mask tells the model which positions are real data vs padding.

### Efficiency Considerations

**Distribution caching:**
```python
if max_removals in self.distribution_cache:
    distribution = self.distribution_cache[max_removals]
```
Computing the skewed triangular distribution is expensive. We cache distributions for each `max_removals` value. Typical gear sets have 3-8 abilities, so the cache stays small.

**Random sampling:**
```python
removal_indices = random.sample(range(set_length), num_removals)
```
`random.sample` is O(num_removals) without replacement. Fast for small sets.

**Determinism:**
- If you set `random_state` in dataset generation, splits are reproducible
- Within an epoch, masking is random (different each epoch)
- This is intentional: more data augmentation

## Tuning Parameters

**num_instances_per_set:**
- Default: 5
- Higher values: more augmentation, slower training
- Lower values: less overfitting prevention
- Sweet spot: 5-10 for most datasets

**skew_factor:**
- Default: 1.2
- Higher values: bias toward heavy masking
- Lower values: more uniform removal distribution
- 1.0 = pure triangular, 2.0 = very aggressive

**null_token:**
- Enable to include "baseline build" samples
- Adds 1 instance per set with no context
- Useful for cold-start recommendations

## Training Implications

**Batch composition:**
- Each batch contains masks from different original sets
- Different mask amounts per sample
- Variable sequence lengths handled by padding

**Epoch diversity:**
- Each epoch generates different masks
- The model sees 5 different partial builds per set per epoch
- Effective dataset size = actual_size × epochs × instances_per_set

**Convergence:**
- More augmentation means more epochs needed
- But better generalization at the end
- Early stopping prevents overfitting

## Code Location

Dataset implementation: `src/splatnlp/preprocessing/datasets/dataset.py`
- `SetDataset.__getitem__()`: Instance generation
- `weighted_random_removals()`: Skewed sampling

Dataloader generation: `src/splatnlp/preprocessing/datasets/generate_datasets.py`
- `generate_dataloaders()`: Creates train/val/test loaders with masking

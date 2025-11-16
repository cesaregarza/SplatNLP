# Evaluation Metrics

The project uses multiple metrics tailored to the multi-label classification problem. Different metrics capture different aspects of model quality.

## Primary Model Metrics

The SetCompletionModel predicts multiple ability tokens simultaneously. Each prediction is a binary decision: does this ability belong in the build or not?

### F1 Score

The harmonic mean of precision and recall:
```python
f1 = 2 * precision * recall / (precision + recall)
```

**Why F1?** Balances false positives (suggesting wrong abilities) and false negatives (missing good abilities). A model that predicts everything has high recall but low precision. A model that predicts nothing has undefined precision and zero recall. F1 penalizes both extremes.

The implementation uses scikit-learn's `f1_score` with `average='samples'`, meaning F1 is computed per-sample then averaged. This handles variable numbers of true abilities per sample.

### Precision

What fraction of predicted abilities were correct:
```python
precision = true_positives / (true_positives + false_positives)
```

High precision means when the model suggests an ability, it's usually a good suggestion. Low precision means lots of noise in the recommendations.

### Recall

What fraction of true abilities were predicted:
```python
recall = true_positives / (true_positives + false_negatives)
```

High recall means the model captures most of the optimal abilities. Low recall means it misses important ones.

### Hamming Distance

Per-token error rate:
```python
hamming = mean(predictions != targets)
```

Unlike F1/precision/recall which care about the set of predictions, Hamming distance treats each ability independently. A Hamming distance of 0.05 means 5% of all ability predictions are wrong (either false positive or false negative).

Useful for understanding raw accuracy, but doesn't distinguish between types of errors.

### Loss (BCEWithLogitsLoss)

Binary cross-entropy with logits:
```python
loss = -mean(targets * log(sigmoid(logits)) + (1-targets) * log(1-sigmoid(logits)))
```

The training objective. Lower is better. Measures how well the model's probability estimates match the true binary labels. A model that outputs 0.5 for everything has high loss. A model that outputs 1.0 for true abilities and 0.0 for false ones has near-zero loss.

## Metric Computation

**Training** (`src/splatnlp/model/training_loop.py`):
```python
predictions = (torch.sigmoid(logits) > 0.5).cpu().numpy()
targets = targets.cpu().numpy()

metrics = {
    "f1": f1_score(targets, predictions, average="samples", zero_division=0),
    "precision": precision_score(targets, predictions, average="samples", zero_division=0),
    "recall": recall_score(targets, predictions, average="samples", zero_division=0),
    "hamming": hamming_loss(targets, predictions),
}
```

The threshold of 0.5 converts logits to binary predictions. This is the training-time decision boundary.

**Validation/Test**:
Same metrics, but computed on held-out data. Validation metrics guide early stopping and learning rate scheduling. Test metrics report final performance.

**Distributed Aggregation**:
In DDP mode, metrics are computed locally then averaged across ranks:
```python
torch.distributed.all_reduce(metric_tensor, op=ReduceOp.SUM)
metric_tensor /= world_size
```

This gives globally consistent metrics even when data is distributed.

## SAE-Specific Metrics

The Sparse Autoencoder has different objectives than the primary model.

### Reconstruction Error (MSE)

How well does the SAE reconstruct the original activation?
```python
mse = mean((reconstruction - original) ** 2)
```

Lower is better, but too low means the SAE might not be sparse enough (overfitting to exact reconstruction).

### L1 Sparsity

Average absolute activation across all SAE neurons:
```python
l1 = mean(sum(|activations|))
```

Measures how "active" the SAE is. Lower means sparser. But too low means dead neurons.

### L0 Sparsity

Number of active neurons per sample:
```python
l0 = mean(sum(activations > 0))
```

More interpretable than L1. If L0 = 15, on average 15 SAE neurons fire per input. Target is typically 10-20 for interpretability.

### Dead Neuron Percentage

Fraction of SAE neurons that never activate:
```python
dead_pct = sum(usage_ema < threshold) / num_neurons * 100
```

Dead neurons waste capacity. Some are expected (5-10%), but too many (>30%) indicates training issues.

### Feature Magnitude

Average activation strength of active neurons:
```python
magnitude = mean(activations[activations > 0])
```

Measures signal strength. Very low magnitude with high sparsity might mean the SAE is barely representing anything.

### KL Divergence (Usage Regularization)

How far is actual feature usage from target usage?
```python
kl = target * log(target / actual) + (1-target) * log((1-target) / (1-actual))
```

Pushes each neuron toward a target firing rate (e.g., 5%). Prevents both dead neurons (0% usage) and overused neurons (100% usage).

### Miracle Distance

Composite metric balancing all SAE objectives:
```python
distance = sqrt(
    w_mse * log(mse / target_mse) ** 2
    + w_sparsity * log(sparsity / target_sparsity) ** 2
    + w_dead * log(dead_pct / target_dead_pct) ** 2
)
```

Used as the optimization target for hyperparameter sweeps. A single number that captures "overall SAE quality." Lower is better.

**Default targets**:
- MSE: 0.015 (low reconstruction error)
- Sparsity: 15.0 (about 15 features active per input)
- Dead %: 5.0 (some dead neurons are okay)

## Evaluation Functions

**Primary Model** (`src/splatnlp/model/evaluation.py`):
```python
def test_model(model, criterion, dataloader, device):
    """Evaluate on test set, return all metrics."""
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            logits = model(...)
            loss = criterion(logits, targets)
            # Accumulate predictions and targets
    # Compute F1, precision, recall, hamming
    return metrics
```

No gradient computation, uses full dataset.

**SAE** (`src/splatnlp/monosemantic_sae/sae_training/evaluate.py`):
```python
def evaluate_sae_model(sae_model, buffer, device, config):
    """Evaluate SAE on activation buffer."""
    sae_model.eval()
    with torch.no_grad():
        for batch in buffer:
            reconstruction = sae_model(batch)
            mse = F.mse_loss(reconstruction, batch)
            activations = sae_model.get_activations(batch)
            # Compute sparsity, magnitude, etc.
    return SAEMetrics(mse, sparsity, dead_pct, ...)
```

Uses the activation buffer, not raw data.

## Ablation Analysis

The dashboard provides interactive ablation:
```python
def ablate_feature(model, sae, input, feature_id, new_value):
    """Modify one SAE feature and observe prediction changes."""
    activations = sae.encode(model.get_hidden(input))
    activations[feature_id] = new_value
    modified = sae.decode(activations)
    return model.predict_from_hidden(modified)
```

This answers: "What happens to predictions if I change feature X?"

Used for:
- Understanding feature importance
- Validating interpretability claims
- Causal analysis of model behavior

## Metric Interpretation

**Good primary model**:
- F1 > 0.7 (most predictions correct, most true abilities found)
- Precision ~ Recall (balanced trade-off)
- Hamming < 0.1 (less than 10% token error rate)

**Good SAE**:
- MSE < 0.02 (faithful reconstruction)
- L0 sparsity 10-20 (interpretable number of features)
- Dead neurons < 10% (efficient use of capacity)
- Miracle distance < 1.0 (near target operating point)

**Warning signs**:
- High precision, low recall: Model is too conservative
- High recall, low precision: Model is too aggressive
- High MSE: SAE losing information
- High dead %: SAE capacity wasted
- Very low sparsity: SAE not separating concepts

## Code Location

Primary model evaluation: `src/splatnlp/model/evaluation.py`
Training metrics: `src/splatnlp/model/training_loop.py`
SAE evaluation: `src/splatnlp/monosemantic_sae/sae_training/evaluate.py`
Ablation analysis: `src/splatnlp/dashboard/components/ablation_component.py`

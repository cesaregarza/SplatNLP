# Sparse Autoencoder for Interpretability

The Sparse Autoencoder (SAE) is a secondary model trained on the activations of the main SetCompletionModel. Its goal is to find interpretable features that explain what the model has learned. This follows Anthropic's work on monosemanticity.

## The Problem: Superposition

Neural networks pack lots of information into their hidden dimensions. A single neuron might respond to multiple unrelated concepts. This is called superposition, and it makes the model hard to interpret.

In the SetCompletionModel, the 512-dimensional hidden vector after pooling encodes everything the model knows about a gear set. But we can't just look at individual dimensions and understand what they mean.

## The Solution: Sparse Overcomplete Dictionary

The SAE learns a sparse, overcomplete representation of those activations.

**Overcomplete**: The SAE has more neurons than the input dimensionality. With an expansion factor of 4, we go from 512 dimensions to 2048 SAE neurons.

**Sparse**: Only a small fraction of SAE neurons activate for any given input. We enforce this with L1 regularization.

The idea: if we force the model to use fewer neurons (sparsity) but give it more neurons to choose from (overcompleteness), each neuron should specialize in one specific concept. This is monosemanticity.

## Architecture

```
Input: 512-d activation from SetCompletionModel
           ↓
     Subtract learned bias
           ↓
     Linear encoder (512 → 2048)
           ↓
     ReLU + clip(0, 6)
           ↓
     Normalized linear decoder (2048 → 512)
           ↓
     Add learned bias
           ↓
Output: Reconstructed 512-d activation
```

The decoder weights are normalized (unit norm). This prevents the model from cheating by scaling up weights to reduce the L1 penalty.

## Loss Function

Three components:

**Reconstruction Loss (MSE)**: The SAE should faithfully reconstruct the original activation.
```
L_recon = mean((reconstruction - input)²)
```

**L1 Sparsity**: Penalizes the total activation of SAE neurons. Pushes most neurons toward zero.
```
L_sparse = mean(sum(|activations|))
```

**Usage Regularization (KL Divergence)**: Ensures neurons are used at a target frequency. Prevents dead neurons (never activate) and overused neurons (fire on everything).
```
L_usage = KL(target_usage || actual_usage)
```

Target usage is typically 5%, meaning each neuron should fire on about 5% of inputs.

**Total Loss**:
```
L = L_recon + l1_coeff × L_sparse + usage_coeff × L_usage
```

The coefficients are scheduled. L1 starts at 0 and ramps up over 6000 steps. Usage coefficient follows a cosine schedule with warmup.

## Training Process

The SAE trains on activations from the primary model, not on raw data.

1. **Primary model forward pass** (inference mode): Feed a batch of gear sets through SetCompletionModel
2. **Capture activations**: Hook intercepts the 512-d vector after masked mean pooling
3. **Buffer activations**: Store in a circular buffer (819k activations, CUDA memory)
4. **SAE training**: Every N primary steps, train the SAE on buffered activations

This decouples SAE training from primary model training. The primary model is frozen; only the SAE learns.

## Dead Neuron Resampling

A common problem: some SAE neurons never activate. They're "dead" and contribute nothing to the representation.

The training loop tracks neuron usage with an exponential moving average. At predefined steps (7k, 14k, 28k), it:

1. Identifies dead neurons (usage < threshold)
2. Finds inputs with high reconstruction error
3. Resamples dead neurons to match those high-error inputs
4. Resets Adam optimizer moments for those neurons

This gives dead neurons a second chance to learn useful features.

## Model Steering

The SAE enables causal interventions. If you want to know what a feature does, you can:

1. Hook into the SetCompletionModel
2. Run an input through the SAE encoder
3. Modify one neuron's activation
4. Reconstruct via the decoder
5. Continue the forward pass with the modified activation

The `SetCompletionHook` class does this. It has three modes:
- **BYPASS**: No SAE, just pass activations through
- **NO-CHANGE**: Reconstruct via SAE without modification (useful for measuring reconstruction error)
- **EDIT**: Modify a specific neuron's value before reconstruction

Example: "Feature 47 activates strongly when the build has high Swim Speed. What if I artificially set feature 47 to 0?" You can see how the model's predictions change.

## Feature Analysis

Once trained, we analyze what each SAE neuron represents.

**Top Activating Examples**: For each neuron, find the gear sets that activate it most. Common patterns reveal what the neuron encodes.

**Sparsity Ranking**: Score neurons by magnitude/sqrt(sparsity). Neurons that are selective (sparse) but confident (high magnitude) are most interpretable.

**Filtering**: Keep neurons with 1-30% activation frequency. Too sparse means rare edge cases. Too dense means generic patterns.

The dashboard (`src/splatnlp/dashboard/`) visualizes these analyses interactively.

## Interpreting Features

In practice, SAE features often correspond to:
- Specific ability combinations (e.g., "builds with both Swim Speed and Ninja Squid")
- Weapon class patterns (e.g., "shooter-style builds")
- Strategic archetypes (e.g., "aggressive frontline builds")
- Meta patterns (e.g., "current competitive meta")

The monosemanticity goal is that each feature corresponds to one concept. Not always perfect, but much better than raw neurons.

## Limitations

**Reconstruction error**: The SAE doesn't perfectly reconstruct activations. There's always some information loss. The better the sparsity, typically the worse the reconstruction.

**Subjective labeling**: Interpreting what a feature "means" requires human judgment. The model doesn't know it learned "aggressive builds", you infer that from the examples.

**Training instability**: Balancing L1 sparsity, usage regularization, and reconstruction quality is tricky. Dead neurons can cascade. Hyperparameter tuning is required.

**Computational cost**: Training the SAE requires running the primary model on lots of data to collect activations. With 819k buffer size and multiple epochs, it's not cheap.

## Code Location

Main SAE code: `src/splatnlp/monosemantic_sae/`

Key modules:
- `models.py`: SparseAutoencoder architecture
- `hooks.py`: SetCompletionHook for model steering
- `sae_training/`: Training loop and utilities
- `feature_analysis.py`: Analysis tools
- `cli.py`: Command-line training interface

Dashboard: `src/splatnlp/dashboard/`

## References

Based on Anthropic's work: [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html)

The core insight: sparse overcomplete dictionaries can extract interpretable features from superposed representations. Applied here to understand what the gear recommendation model has learned about optimal builds.

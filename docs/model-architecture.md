# Model Architecture

The SetCompletionModel (nicknamed SplatGPT) is the core model for predicting gear abilities. It takes a partial gear set and weapon as input and predicts which additional abilities would complement the build.

## Why a Custom Architecture?

Standard transformer models process sequences where order matters. But gear sets are fundamentally unordered. The abilities you have equipped don't care what order you picked them in. This is a set completion problem, not a sequence prediction problem.

The model combines two ideas:
- Set Transformers for handling unordered inputs
- GPT-2 style prediction for generating completions

## Architecture Overview

```
Input: [ability_tokens] + weapon_id
           ↓
     Embeddings (add weapon context to ability embeddings)
           ↓
     SetTransformerLayer × N (with residuals)
           ↓
     Masked Mean Pooling
           ↓
     Output Layer (vocab_size logits)
```

The model runs about 83M parameters with default settings (embedding_dim=32, hidden_dim=512, 3 layers, 8 heads).

## Component Breakdown

### Input Embeddings

Two separate embeddings:
- **ability_embedding**: Maps ability tokens to vectors
- **weapon_embedding**: Maps weapon IDs to vectors

These get combined by addition. The weapon embedding gets broadcast across all ability tokens, so every ability representation is "contextualized" by the weapon. A Swim Speed Up token for a Splattershot plays differently than for a Slosher.

### SetTransformerLayer

Each layer has three main parts:

1. **SetTransformer Block**
   - Uses "inducing points" as a bottleneck. Instead of computing attention between all pairs of tokens (O(n²)), it routes attention through a fixed set of learned points (O(n)).
   - Encoder: Two InducedSetAttentionBlocks compress the variable-length set into a fixed representation
   - Decoder: One PoolingMultiheadAttention block + two self-attention blocks decode it back

2. **Cross-Attention**
   - Projects the compressed representation back to the original sequence length
   - Needed for the residual connection

3. **Feedforward Network**
   - Standard feedforward with 4x expansion
   - Adds non-linearity after attention

Residual connections wrap each sub-component. Layer normalization is optional but typically enabled during training.

### Attention Mechanisms

**InducedSetAttentionBlock**: The key innovation from Set Transformers. Learned inducing points act as a communication bottleneck. Tokens attend to inducing points, which aggregate information, then inducing points attend back to tokens. This gives global context with linear complexity.

**PoolingMultiheadAttention**: Uses learned seed vectors to pool information from the input set. Think of it as "learned queries" that extract specific aspects of the set.

**SelfAttentionBlock**: Standard self-attention wrapped in a residual block.

### Masked Mean Pooling

After the transformer layers, we need to reduce the sequence to a single vector. Simple mean pooling won't work because of padding tokens. The masked mean:
1. Masks out padding positions
2. Sums the valid token representations
3. Divides by the count of valid tokens

This handles variable-length inputs correctly.

### Output Layer

A linear projection from hidden_dim to vocab_size. Each output logit corresponds to one possible ability token. The model predicts all tokens simultaneously (multi-label classification), not one at a time like autoregressive models.

## Training Details

**Loss Function**: BCEWithLogitsLoss for multi-label classification. Each ability token is a binary prediction.

**Metrics**:
- F1 Score (harmonic mean of precision and recall)
- Precision (what fraction of predictions were correct)
- Recall (what fraction of true abilities were predicted)
- Hamming Loss (per-token error rate)

**Optimization**:
- Mixed precision training (bfloat16) with gradient scaling
- ReduceLROnPlateau scheduler
- Early stopping based on validation loss

## Key Design Decisions

**Weapon as additive context**: Adding weapon embeddings to ability embeddings (rather than concatenating or using cross-attention) keeps the dimensionality constant and lets the model learn weapon-ability interactions directly.

**Multi-hot targets**: The model predicts the entire vocabulary at once rather than generating one ability at a time. This makes training faster and avoids exposure bias, but means the model can't capture inter-ability dependencies during generation.

**Inducing points for efficiency**: With potentially hundreds of ability tokens, O(n²) attention would be expensive. Inducing points make the model scale linearly while maintaining the benefits of global attention.

**Permutation invariance**: The set transformer architecture doesn't depend on input order. You can shuffle the input abilities and get the same prediction (modulo numerical precision).

## Code Location

Main model definition: `src/splatnlp/model/models.py`

Key classes:
- `SetCompletionModel`: The full model
- `SetTransformerLayer`: Individual transformer layer
- `SetTransformer`: The set transformer sub-block
- `InducedSetAttentionBlock`: Attention with inducing points
- `PoolingMultiheadAttention`: Learned pooling mechanism

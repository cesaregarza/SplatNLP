# SplatNLP Documentation

Technical documentation for the SplatNLP project.

## Model & Data

Core architecture and data processing:

- [Model Architecture](model-architecture.md) - SetCompletionModel internals, attention mechanisms, why set-based
- [Preprocessing Pipeline](preprocessing-pipeline.md) - Data flow from stat.ink to tokenized training data
- [Sparse Autoencoder](sparse-autoencoder.md) - Interpretability via monosemantic feature learning

## Infrastructure

Training and deployment:

- [Training Infrastructure](training-infrastructure.md) - Distributed training, mixed precision, memory optimization
- [Experiment Tracking](experiment-tracking.md) - WandB integration, hyperparameter sweeps, reproducibility
- [Evaluation Metrics](evaluation-metrics.md) - F1, precision, recall, SAE-specific metrics
- [Testing](testing.md) - Test suite, CI/CD, code quality
- [API Serving](api-serving.md) - FastAPI deployment, request/response format, production considerations

## Quick Start

For usage instructions and CLI examples, see the main [README](../README.md).

These docs focus on the technical internals: why the architecture is designed the way it is, what the preprocessing steps accomplish, and how the ML infrastructure supports production-grade development.

## Highlights

**Model**: 83M parameter set transformer with weapon context integration. Handles variable-length unordered inputs. Multi-label classification for gear recommendations.

**Training**: Distributed training via DDP, mixed precision (bfloat16), gradient clipping, early stopping, learning rate scheduling. Full experiment tracking with WandB sweeps.

**Interpretability**: Sparse autoencoder trained on model activations. Monosemantic features with usage regularization. Interactive dashboard for feature exploration.

**Testing**: 14+ test files covering model, data loading, training loops, hooks, and dashboard components. CI/CD via GitHub Actions runs tests on every PR.

**Deployment**: FastAPI server with Pydantic validation. Model artifacts loaded from remote URLs. ~50ms inference latency.

## Code Layout

```
src/splatnlp/
├── model/              # SetCompletionModel, training, evaluation
├── preprocessing/      # Data pipeline, tokenization, sampling
├── monosemantic_sae/   # Sparse autoencoder, hooks, analysis
├── dashboard/          # Interactive SAE feature explorer
├── embeddings/         # Doc2Vec experiments, clustering
├── serve/              # FastAPI inference server
└── viz/                # Visualization utilities
```

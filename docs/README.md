# SplatNLP Documentation

Technical documentation for the SplatNLP project.

## Contents

- [Model Architecture](model-architecture.md) - How the SetCompletionModel (SplatGPT) works internally
- [Preprocessing Pipeline](preprocessing-pipeline.md) - Data processing from raw stat.ink data to tokenized training sets
- [Sparse Autoencoder](sparse-autoencoder.md) - Interpretability via monosemantic feature learning

## Quick Start

For usage instructions and CLI examples, see the main [README](../README.md).

These docs focus on the technical internals: why the architecture is designed the way it is, what the preprocessing steps accomplish, and how the interpretability system works.

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

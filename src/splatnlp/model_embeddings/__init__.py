"""Model-derived embedding utilities for SetCompletionModel."""

from .extract import extract_embeddings, extract_embeddings_from_dataloader
from .harness import build_embedding_dataloader, extract_training_embeddings
from .trajectory import build_predictors
from .verify import verify_embeddings_vs_logits

__all__ = [
    "extract_embeddings",
    "extract_embeddings_from_dataloader",
    "build_embedding_dataloader",
    "extract_training_embeddings",
    "build_predictors",
    "verify_embeddings_vs_logits",
]

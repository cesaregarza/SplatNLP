from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from gensim.models import Doc2Vec
    from gensim.models.doc2vec import TaggedDocument


_MISSING_GENSIM_MSG = (
    "Gensim is required for the embeddings/Doc2Vec workflow. "
    "Install it via `poetry install --with embeddings` "
    "(or `poetry install --with dev,embeddings`)."
)


def require_doc2vec():
    try:
        from gensim.models import Doc2Vec
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_MISSING_GENSIM_MSG) from exc
    return Doc2Vec


def require_doc2vec_and_tagged_document():
    try:
        from gensim.models import Doc2Vec
        from gensim.models.doc2vec import TaggedDocument
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_MISSING_GENSIM_MSG) from exc
    return Doc2Vec, TaggedDocument

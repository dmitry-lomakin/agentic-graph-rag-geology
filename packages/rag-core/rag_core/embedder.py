"""Batch embedding via local SentenceTransformer (multilingual-e5-large).

Replaces OpenAI embeddings with local model for bilingual RU/EN geology corpus.
Uses e5 prefixes: "passage: " for documents, "query: " for queries.
"""

from __future__ import annotations

import logging

from rag_core.config import get_settings, make_embedding_model
from rag_core.models import Chunk

logger = logging.getLogger(__name__)


def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Batch embed chunks using local SentenceTransformer.

    Uses enriched_content (context + content) if available.
    Applies "passage: " prefix per e5 convention.
    Sets chunk.embedding for each chunk.
    """
    if not chunks:
        return chunks

    cfg = get_settings()
    model = make_embedding_model(cfg)

    texts = [f"{cfg.embedding.prefix_passage}{chunk.enriched_content}" for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=cfg.embedding.batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    for i, chunk in enumerate(chunks):
        chunk.embedding = embeddings[i].tolist()

    logger.info("Embedded %d chunks (%s)", len(chunks), cfg.embedding.model_name)
    return chunks


def embed_query(query: str) -> list[float]:
    """Embed a single query string using local SentenceTransformer.

    Applies "query: " prefix per e5 convention.
    Returns a list of floats (1024-dim for multilingual-e5-large).
    """
    cfg = get_settings()
    model = make_embedding_model(cfg)

    prefixed = f"{cfg.embedding.prefix_query}{query}"
    embedding = model.encode(
        [prefixed],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embedding[0].tolist()


def embed_texts(texts: list[str], prefix: str | None = None) -> list[list[float]]:
    """Embed multiple texts with optional custom prefix.

    Args:
        texts: Raw text strings to embed.
        prefix: Override prefix (defaults to passage prefix).

    Returns:
        List of float vectors, one per input text.
    """
    if not texts:
        return []

    cfg = get_settings()
    model = make_embedding_model(cfg)

    if prefix is None:
        prefix = cfg.embedding.prefix_passage

    prefixed = [f"{prefix}{t}" for t in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=cfg.embedding.batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return [emb.tolist() for emb in embeddings]

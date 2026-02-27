"""Agentic Graph RAG configuration via Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Neo4jSettings(BaseSettings):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "neo4j"

    model_config = {"env_prefix": "NEO4J_"}


class OpenAISettings(BaseSettings):
    api_key: str = ""
    base_url: str = ""  # LiteLLM proxy: e.g. "http://localhost:4000/v1"
    llm_model: str = "gpt-4o"
    llm_model_mini: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    model_config = {"env_prefix": "OPENAI_"}


class EmbeddingSettings(BaseSettings):
    model_name: str = "intfloat/multilingual-e5-large"
    dimensions: int = 1024
    prefix_passage: str = "passage: "
    prefix_query: str = "query: "
    batch_size: int = 32
    device: str = ""  # auto-detect if empty

    model_config = {"env_prefix": "EMBEDDING_"}


class IndexingSettings(BaseSettings):
    chunk_size: int = 768
    chunk_overlap: int = 100
    skeleton_beta: float = 0.25
    knn_k: int = 10
    pagerank_damping: float = 0.85

    model_config = {"env_prefix": "INDEXING_"}


class RetrievalSettings(BaseSettings):
    top_k_vector: int = 10
    top_k_final: int = 10
    vector_threshold: float = 0.5
    max_hops: int = 3
    ppr_alpha: float = 0.15

    model_config = {"env_prefix": "RETRIEVAL_"}


class AgentSettings(BaseSettings):
    max_retries: int = 2
    relevance_threshold: float = 2.0

    model_config = {"env_prefix": "AGENT_"}


class Settings(BaseSettings):
    neo4j: Neo4jSettings = Neo4jSettings()
    openai: OpenAISettings = OpenAISettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    indexing: IndexingSettings = IndexingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    agent: AgentSettings = AgentSettings()

    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Create and cache settings instance loading from environment."""
    return Settings()


_embedding_model = None


def make_embedding_model(settings: Settings | None = None):
    """Create and cache SentenceTransformer embedding model.

    Returns a SentenceTransformer instance loaded with the configured model.
    Cached globally so the model is loaded only once.
    """
    global _embedding_model  # noqa: PLW0603
    if _embedding_model is not None:
        return _embedding_model

    from sentence_transformers import SentenceTransformer

    cfg = settings or get_settings()
    device = cfg.embedding.device or None  # None = auto-detect
    _embedding_model = SentenceTransformer(cfg.embedding.model_name, device=device)
    return _embedding_model


def make_openai_client(settings: Settings | None = None):
    """Create OpenAI client with optional LiteLLM proxy support.

    If OPENAI_BASE_URL is set, uses it as base_url (e.g. LiteLLM proxy).
    If api_key is empty and base_url is set, uses "none" as placeholder.
    Raises ValueError if neither api_key nor base_url is configured.
    """
    from openai import OpenAI

    cfg = settings or get_settings()
    if not cfg.openai.api_key and not cfg.openai.base_url:
        raise ValueError(
            "OPENAI_API_KEY or OPENAI_BASE_URL must be set. "
            "Set OPENAI_API_KEY for direct OpenAI access, or "
            "OPENAI_BASE_URL for a LiteLLM proxy."
        )
    kwargs: dict[str, str] = {}
    if cfg.openai.api_key:
        kwargs["api_key"] = cfg.openai.api_key
    elif cfg.openai.base_url:
        kwargs["api_key"] = "none"  # LiteLLM proxy doesn't need real key
    if cfg.openai.base_url:
        kwargs["base_url"] = cfg.openai.base_url
    return OpenAI(**kwargs)

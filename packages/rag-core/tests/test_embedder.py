"""Tests for rag_core.embedder (SentenceTransformer-based)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rag_core.embedder import embed_chunks, embed_query, embed_texts
from rag_core.models import Chunk


class TestEmbedChunks:
    def test_empty_list(self):
        assert embed_chunks([]) == []

    @patch("rag_core.embedder.make_embedding_model")
    @patch("rag_core.embedder.get_settings")
    def test_embeds_chunks(self, mock_settings, mock_make_model):
        cfg = MagicMock()
        cfg.embedding.prefix_passage = "passage: "
        cfg.embedding.batch_size = 32
        cfg.embedding.model_name = "test-model"
        mock_settings.return_value = cfg

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_make_model.return_value = mock_model

        chunks = [Chunk(content="hello"), Chunk(content="world")]
        result = embed_chunks(chunks)

        assert len(result) == 2
        assert result[0].embedding == [0.1, 0.2, 0.3]
        assert result[1].embedding == [0.4, 0.5, 0.6]

        # Verify called with passage prefix
        call_args = mock_model.encode.call_args
        texts = call_args[0][0]
        assert texts[0] == "passage: hello"
        assert texts[1] == "passage: world"

    @patch("rag_core.embedder.make_embedding_model")
    @patch("rag_core.embedder.get_settings")
    def test_uses_enriched_content(self, mock_settings, mock_make_model):
        cfg = MagicMock()
        cfg.embedding.prefix_passage = "passage: "
        cfg.embedding.batch_size = 32
        cfg.embedding.model_name = "test-model"
        mock_settings.return_value = cfg

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1]])
        mock_make_model.return_value = mock_model

        chunk = Chunk(content="text", context="prefix")
        embed_chunks([chunk])

        call_args = mock_model.encode.call_args
        texts = call_args[0][0]
        assert texts == ["passage: prefix\n\ntext"]  # enriched_content

    @patch("rag_core.embedder.make_embedding_model")
    @patch("rag_core.embedder.get_settings")
    def test_raises_on_model_error(self, mock_settings, mock_make_model):
        cfg = MagicMock()
        cfg.embedding.prefix_passage = "passage: "
        cfg.embedding.batch_size = 32
        cfg.embedding.model_name = "test-model"
        mock_settings.return_value = cfg

        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Model error")
        mock_make_model.return_value = mock_model

        with pytest.raises(Exception, match="Model error"):
            embed_chunks([Chunk(content="test")])


class TestEmbedQuery:
    @patch("rag_core.embedder.make_embedding_model")
    @patch("rag_core.embedder.get_settings")
    def test_embed_query(self, mock_settings, mock_make_model):
        cfg = MagicMock()
        cfg.embedding.prefix_query = "query: "
        cfg.embedding.batch_size = 32
        mock_settings.return_value = cfg

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.7, 0.8, 0.9]])
        mock_make_model.return_value = mock_model

        result = embed_query("test query")
        assert result == [0.7, 0.8, 0.9]

        call_args = mock_model.encode.call_args
        assert call_args[0][0] == ["query: test query"]


class TestEmbedTexts:
    @patch("rag_core.embedder.make_embedding_model")
    @patch("rag_core.embedder.get_settings")
    def test_embed_texts(self, mock_settings, mock_make_model):
        cfg = MagicMock()
        cfg.embedding.prefix_passage = "passage: "
        cfg.embedding.batch_size = 32
        mock_settings.return_value = cfg

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_make_model.return_value = mock_model

        result = embed_texts(["text1", "text2"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]

    def test_empty_list(self):
        assert embed_texts([]) == []

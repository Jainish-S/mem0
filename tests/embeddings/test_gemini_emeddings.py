from unittest.mock import patch

import pytest

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.gemini import GoogleGenAIEmbedding


@pytest.fixture
def mock_genai():
    with patch("mem0.embeddings.gemini.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_embed_response = type(
            "EmbedResponse", (), {"embeddings": [type("Embedding", (), {"values": [0.1, 0.2, 0.3, 0.4]})()]}
        )()
        mock_client.models.embed_content.return_value = mock_embed_response
        yield mock_client


@pytest.fixture
def config():
    return BaseEmbedderConfig(api_key="dummy_api_key", model="test_model", embedding_dims=786)


def test_embed_query(mock_genai, config):
    embedder = GoogleGenAIEmbedding(config)

    text = "Hello, world!"
    embedding = embedder.embed(text)

    assert embedding == [0.1, 0.2, 0.3, 0.4]
    from google.genai import types

    mock_genai.models.embed_content.assert_called_once_with(
        model="test_model", contents="Hello, world!", config=types.EmbedContentConfig(output_dimensionality=786)
    )


def test_embed_returns_empty_list_if_none(mock_genai, config):
    mock_genai.models.embed_content.return_value = None

    embedder = GoogleGenAIEmbedding(config)

    with pytest.raises(AttributeError):
        embedder.embed("test")


def test_embed_raises_on_error(mock_genai, config):
    mock_genai.models.embed_content.side_effect = RuntimeError("Embedding failed")

    embedder = GoogleGenAIEmbedding(config)

    with pytest.raises(RuntimeError, match="Embedding failed"):
        embedder.embed("some input")


def test_config_initialization(config):
    embedder = GoogleGenAIEmbedding(config)

    assert embedder.config.api_key == "dummy_api_key"
    assert embedder.config.model == "test_model"
    assert embedder.config.embedding_dims == 786

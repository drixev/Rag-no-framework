from types import SimpleNamespace
from unittest.mock import MagicMock

from rag.embeddings import openai_embeddings


def _fake_response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])


def test_embed_texts_returns_empty_list_without_calling_api(monkeypatch):
    fake_create = MagicMock()
    monkeypatch.setattr(openai_embeddings._client.embeddings, "create", fake_create)

    result = openai_embeddings.embed_texts([])

    assert result == []
    fake_create.assert_not_called()


def test_embed_texts_splits_into_batches(monkeypatch):
    calls = []

    def fake_create(model, input):
        calls.append(list(input))
        return _fake_response([[float(len(t))] for t in input])

    monkeypatch.setattr(openai_embeddings._client.embeddings, "create", fake_create)

    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    result = openai_embeddings.embed_texts(texts, batch_size=2)

    assert calls == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert result == [[1.0], [2.0], [3.0], [4.0], [5.0]]

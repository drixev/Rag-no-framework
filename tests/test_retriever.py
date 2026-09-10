from rag.config import settings
from rag.retrieval import retriever


def test_retrieve_embeds_query_and_delegates_to_index_store(monkeypatch):
    captured = {}

    def fake_embed_texts(texts):
        captured["texts"] = texts
        return [[1.0, 0.0, 0.0]]

    def fake_query_hybrid(query_vector, query_text, k, conninfo):
        captured["query_vector"] = query_vector
        captured["query_text"] = query_text
        captured["k"] = k
        captured["conninfo"] = conninfo
        return [{"id": "producto:1", "text": "hola", "score": 0.9}]

    monkeypatch.setattr(retriever, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retriever, "query_hybrid", fake_query_hybrid)

    results = retriever.retrieve("hola mundo", k=3)

    assert captured["texts"] == ["hola mundo"]
    assert captured["query_vector"] == [1.0, 0.0, 0.0]
    assert captured["query_text"] == "hola mundo"
    assert captured["k"] == 3
    assert results == [{"id": "producto:1", "text": "hola", "score": 0.9}]


def test_retrieve_uses_default_top_k(monkeypatch):
    captured = {}
    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(
        retriever, "query_hybrid", lambda qv, qt, k, conninfo: captured.setdefault("k", k) or []
    )

    retriever.retrieve("query")

    assert captured["k"] == settings.retrieval_top_k

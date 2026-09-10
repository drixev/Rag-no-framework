from fastapi.testclient import TestClient

import api.main as main_module
from api.main import app
from rag.sql.text_to_sql import UnsafeSQLError

client = TestClient(app)


def test_query_endpoint(monkeypatch):
    monkeypatch.setattr(main_module, "rag_pipeline", lambda query: "respuesta de prueba")

    response = client.post("/query", json={"query": "hola"})

    assert response.status_code == 200
    assert response.json() == {"answer": "respuesta de prueba"}


def test_query_stream_endpoint(monkeypatch):
    def fake_stream(query):
        yield "sources", [{"id": "doc1", "text": "hola", "score": 0.5}]
        yield "chunk", "hola"
        yield "chunk", " mundo"

    monkeypatch.setattr(main_module, "rag_pipeline_stream", fake_stream)

    with client.stream("POST", "/query/stream", json={"query": "hola"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "event: sources" in body
    assert "event: chunk" in body
    assert "hola" in body


def test_query_sql_endpoint_success(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "run_text_to_sql",
        lambda query: {"sql": "SELECT * FROM productos", "rows": [{"id": 1}]},
    )

    response = client.post("/query/sql", json={"query": "dame todos los productos"})

    assert response.status_code == 200
    assert response.json() == {"sql": "SELECT * FROM productos", "rows": [{"id": 1}]}


def test_query_sql_endpoint_rejects_unsafe_sql(monkeypatch):
    def fake_run(query):
        raise UnsafeSQLError("no permitido")

    monkeypatch.setattr(main_module, "run_text_to_sql", fake_run)

    response = client.post("/query/sql", json={"query": "borra todo"})

    assert response.status_code == 422

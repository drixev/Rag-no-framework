from unittest.mock import MagicMock

from rag.ingestion import loaders
from rag.ingestion.loaders import StructuredSource, load_structured_source, load_text_documents


def test_load_text_documents_reads_txt_and_md_only(tmp_path):
    (tmp_path / "a.txt").write_text("contenido a", encoding="utf-8")
    (tmp_path / "b.md").write_text("contenido b", encoding="utf-8")
    (tmp_path / "c.json").write_text("{}", encoding="utf-8")

    docs = load_text_documents(str(tmp_path))

    assert {d.source_id for d in docs} == {"a.txt", "b.md"}
    assert all(d.source_type == "documento" for d in docs)
    assert {d.text for d in docs} == {"contenido a", "contenido b"}


def test_load_structured_source_runs_query_and_serializes_rows(monkeypatch):
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [{"id": 1, "nombre": "Mouse"}, {"id": 2, "nombre": "Teclado"}]
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    monkeypatch.setattr(loaders.psycopg, "connect", lambda conninfo, row_factory=None: mock_conn)

    source = StructuredSource(
        source_type="producto",
        query="SELECT id, nombre FROM productos",
        id_field="id",
        serializer=lambda row: f"Producto: {row['nombre']}",
    )

    documents = load_structured_source(source, "postgresql://fake")

    assert [(d.source_type, d.source_id, d.text) for d in documents] == [
        ("producto", "1", "Producto: Mouse"),
        ("producto", "2", "Producto: Teclado"),
    ]
    mock_cursor.execute.assert_called_once_with(source.query)

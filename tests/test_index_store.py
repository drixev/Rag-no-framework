from unittest.mock import MagicMock

from rag.retrieval import index_store


def _mock_connection(mock_cursor: MagicMock) -> MagicMock:
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    return mock_conn


def test_upsert_chunks_executes_upsert_for_each_row(monkeypatch):
    mock_cursor = MagicMock()
    mock_conn = _mock_connection(mock_cursor)

    monkeypatch.setattr(index_store.psycopg, "connect", lambda conninfo: mock_conn)
    monkeypatch.setattr(index_store, "register_vector", lambda conn: None)

    chunk_rows = [("producto", "1", 0, "texto uno")]
    vectors = [[0.1, 0.2]]

    index_store.upsert_chunks(chunk_rows, vectors, "postgresql://fake")

    mock_cursor.executemany.assert_called_once()
    _, values = mock_cursor.executemany.call_args.args
    assert values == [("producto", "1", 0, "texto uno", [0.1, 0.2])]


def test_upsert_chunks_skips_when_no_rows(monkeypatch):
    fake_connect = MagicMock()
    monkeypatch.setattr(index_store.psycopg, "connect", fake_connect)

    index_store.upsert_chunks([], [], "postgresql://fake")

    fake_connect.assert_not_called()


def test_delete_stale_chunks_deletes_missing_sources_and_excess_indices(monkeypatch):
    mock_cursor = MagicMock()
    mock_conn = _mock_connection(mock_cursor)

    monkeypatch.setattr(index_store.psycopg, "connect", lambda conninfo: mock_conn)
    monkeypatch.setattr(index_store, "register_vector", lambda conn: None)

    index_store.delete_stale_chunks("producto", {"1", "2"}, {"1": 0}, "postgresql://fake")

    delete_call, executemany_call = mock_cursor.execute.call_args, mock_cursor.executemany.call_args
    query, params = delete_call.args
    assert "NOT (source_id = ANY(%s))" in query
    assert params[0] == "producto"
    assert set(params[1]) == {"1", "2"}

    _, values = executemany_call.args
    assert values == [("producto", "1", 0)]


def test_delete_stale_chunks_skips_executemany_when_no_excess_indices(monkeypatch):
    mock_cursor = MagicMock()
    mock_conn = _mock_connection(mock_cursor)

    monkeypatch.setattr(index_store.psycopg, "connect", lambda conninfo: mock_conn)
    monkeypatch.setattr(index_store, "register_vector", lambda conn: None)

    index_store.delete_stale_chunks("producto", {"1"}, {}, "postgresql://fake")

    mock_cursor.execute.assert_called_once()
    mock_cursor.executemany.assert_not_called()


def test_get_indexed_hashes_returns_source_id_to_hash_mapping(monkeypatch):
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [("1", "hash-a"), ("2", "hash-b")]
    mock_conn = _mock_connection(mock_cursor)

    monkeypatch.setattr(index_store.psycopg, "connect", lambda conninfo: mock_conn)
    monkeypatch.setattr(index_store, "register_vector", lambda conn: None)

    result = index_store.get_indexed_hashes("producto", "postgresql://fake")

    assert result == {"1": "hash-a", "2": "hash-b"}
    query, params = mock_cursor.execute.call_args.args
    assert params == ("producto",)


def test_upsert_indexed_sources_executes_upsert_for_each_row(monkeypatch):
    mock_cursor = MagicMock()
    mock_conn = _mock_connection(mock_cursor)

    monkeypatch.setattr(index_store.psycopg, "connect", lambda conninfo: mock_conn)
    monkeypatch.setattr(index_store, "register_vector", lambda conn: None)

    rows = [("producto", "1", "hash-a", 3)]
    index_store.upsert_indexed_sources(rows, "postgresql://fake")

    mock_cursor.executemany.assert_called_once()
    _, values = mock_cursor.executemany.call_args.args
    assert values == rows


def test_upsert_indexed_sources_skips_when_no_rows(monkeypatch):
    fake_connect = MagicMock()
    monkeypatch.setattr(index_store.psycopg, "connect", fake_connect)

    index_store.upsert_indexed_sources([], "postgresql://fake")

    fake_connect.assert_not_called()


def test_delete_stale_indexed_sources_deletes_missing_sources(monkeypatch):
    mock_cursor = MagicMock()
    mock_conn = _mock_connection(mock_cursor)

    monkeypatch.setattr(index_store.psycopg, "connect", lambda conninfo: mock_conn)
    monkeypatch.setattr(index_store, "register_vector", lambda conn: None)

    index_store.delete_stale_indexed_sources("producto", {"1", "2"}, "postgresql://fake")

    query, params = mock_cursor.execute.call_args.args
    assert "NOT (source_id = ANY(%s))" in query
    assert params[0] == "producto"
    assert set(params[1]) == {"1", "2"}


def test_query_hybrid_maps_rows_to_dicts(monkeypatch):
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [("producto", "1", "texto", 0.87)]
    mock_conn = _mock_connection(mock_cursor)

    monkeypatch.setattr(index_store.psycopg, "connect", lambda conninfo: mock_conn)
    monkeypatch.setattr(index_store, "register_vector", lambda conn: None)

    results = index_store.query_hybrid([0.1, 0.2], "webcam barata", 5, "postgresql://fake")

    assert results == [{"id": "producto:1", "text": "texto", "score": 0.87}]

    mock_cursor.execute.assert_called_once()
    query, params = mock_cursor.execute.call_args.args
    assert "bm25" in query.lower()
    assert "@@@" in query
    assert params["query_vector"] == [0.1, 0.2]
    assert params["query_text"] == "webcam barata"
    assert params["k"] == 5
    assert params["candidate_k"] == 25

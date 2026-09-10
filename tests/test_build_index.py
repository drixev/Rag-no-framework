from rag.ingestion import build_index as build_index_module
from rag.ingestion.loaders import SourceDocument, StructuredSource
from rag.ingestion.serialization import ProductRow


def test_structured_documents_loads_every_registered_source(monkeypatch):
    source_a = StructuredSource(source_type="producto", query="SELECT 1", id_field="id", serializer=lambda r: "a")
    source_b = StructuredSource(source_type="pedido", query="SELECT 2", id_field="id", serializer=lambda r: "b")
    monkeypatch.setattr(build_index_module, "STRUCTURED_SOURCES", [source_a, source_b])

    def fake_load(source, conninfo):
        return [SourceDocument(source_type=source.source_type, source_id="1", text=source.source_type)]

    monkeypatch.setattr(build_index_module, "load_structured_source", fake_load)

    documents = build_index_module._structured_documents("conninfo")

    assert {d.source_type for d in documents} == {"producto", "pedido"}


def test_productos_source_uses_canonical_text():
    row = {"id": 1, "nombre": "Mouse", "categoria": None, "precio": 10.0, "descripcion": None, "stock": 1}
    productos_source = next(s for s in build_index_module.STRUCTURED_SOURCES if s.source_type == "producto")

    assert productos_source.serializer(row) == build_index_module.row_to_text(ProductRow(**row))


def _stub_no_previous_hashes(monkeypatch):
    monkeypatch.setattr(build_index_module, "get_indexed_hashes", lambda source_type, conninfo: {})


def test_build_index_from_documents_chunks_embeds_and_upserts(monkeypatch):
    documents = [SourceDocument(source_type="documento", source_id="a.txt", text="hola mundo")]

    _stub_no_previous_hashes(monkeypatch)
    monkeypatch.setattr(build_index_module, "chunk_fixed", lambda text: [text])
    monkeypatch.setattr(build_index_module, "embed_texts", lambda texts: [[0.1, 0.2]])

    captured = {}

    def fake_upsert(chunk_rows, vectors, conninfo):
        captured["chunk_rows"] = chunk_rows
        captured["vectors"] = vectors
        captured["conninfo"] = conninfo

    monkeypatch.setattr(build_index_module, "upsert_chunks", fake_upsert)
    monkeypatch.setattr(build_index_module, "upsert_indexed_sources", lambda *a, **k: None)
    monkeypatch.setattr(build_index_module, "delete_stale_chunks", lambda *a, **k: None)
    monkeypatch.setattr(build_index_module, "delete_stale_indexed_sources", lambda *a, **k: None)

    result = build_index_module.build_index_from_documents(documents, "conninfo")

    assert result == {"count": 1, "sources_total": 1, "sources_reindexed": 1, "sources_skipped": 0}
    assert captured["chunk_rows"] == [("documento", "a.txt", 0, "hola mundo")]
    assert captured["vectors"] == [[0.1, 0.2]]
    assert captured["conninfo"] == "conninfo"


def test_build_index_from_documents_skips_embed_when_no_chunks(monkeypatch):
    _stub_no_previous_hashes(monkeypatch)
    called = {"embed": False}

    def fake_embed(texts):
        called["embed"] = True
        return []

    monkeypatch.setattr(build_index_module, "embed_texts", fake_embed)
    monkeypatch.setattr(build_index_module, "upsert_chunks", lambda *a, **k: None)
    monkeypatch.setattr(build_index_module, "upsert_indexed_sources", lambda *a, **k: None)

    result = build_index_module.build_index_from_documents([], "conninfo")

    assert result == {"count": 0, "sources_total": 0, "sources_reindexed": 0, "sources_skipped": 0}
    assert called["embed"] is False


def test_build_index_from_documents_skips_unchanged_source(monkeypatch):
    unchanged = SourceDocument(source_type="producto", source_id="1", text="sin cambios")
    changed = SourceDocument(source_type="producto", source_id="2", text="con cambios")

    unchanged_hash = build_index_module._hash_text(unchanged.text)
    monkeypatch.setattr(
        build_index_module, "get_indexed_hashes", lambda source_type, conninfo: {"1": unchanged_hash, "2": "stale-hash"}
    )
    monkeypatch.setattr(build_index_module, "chunk_fixed", lambda text: [text])

    embedded_texts = []

    def fake_embed(texts):
        embedded_texts.extend(texts)
        return [[0.0]] * len(texts)

    monkeypatch.setattr(build_index_module, "embed_texts", fake_embed)
    monkeypatch.setattr(build_index_module, "upsert_chunks", lambda *a, **k: None)
    monkeypatch.setattr(build_index_module, "upsert_indexed_sources", lambda *a, **k: None)
    monkeypatch.setattr(build_index_module, "delete_stale_chunks", lambda *a, **k: None)
    monkeypatch.setattr(build_index_module, "delete_stale_indexed_sources", lambda *a, **k: None)

    result = build_index_module.build_index_from_documents([unchanged, changed], "conninfo")

    assert embedded_texts == ["con cambios"]
    assert result == {"count": 1, "sources_total": 2, "sources_reindexed": 1, "sources_skipped": 1}


def test_build_index_from_documents_cleans_orphans_per_source_type(monkeypatch):
    documents = [
        SourceDocument(source_type="producto", source_id="1", text="uno dos"),
        SourceDocument(source_type="documento", source_id="a.txt", text="hola"),
    ]

    _stub_no_previous_hashes(monkeypatch)
    monkeypatch.setattr(build_index_module, "chunk_fixed", lambda text: text.split())
    monkeypatch.setattr(build_index_module, "embed_texts", lambda texts: [[0.0]] * len(texts))
    monkeypatch.setattr(build_index_module, "upsert_chunks", lambda *a, **k: None)
    monkeypatch.setattr(build_index_module, "upsert_indexed_sources", lambda *a, **k: None)

    cleaned = []
    monkeypatch.setattr(
        build_index_module,
        "delete_stale_chunks",
        lambda source_type, valid_ids, max_chunk_index, conninfo: cleaned.append(
            (source_type, valid_ids, max_chunk_index, conninfo)
        ),
    )
    cleaned_hashes = []
    monkeypatch.setattr(
        build_index_module,
        "delete_stale_indexed_sources",
        lambda source_type, valid_ids, conninfo: cleaned_hashes.append((source_type, valid_ids, conninfo)),
    )

    build_index_module.build_index_from_documents(documents, "conninfo")

    cleaned_by_type = {entry[0]: entry for entry in cleaned}
    assert cleaned_by_type["producto"] == ("producto", {"1"}, {"1": 1}, "conninfo")
    assert cleaned_by_type["documento"] == ("documento", {"a.txt"}, {"a.txt": 0}, "conninfo")

    cleaned_hashes_by_type = {entry[0]: entry for entry in cleaned_hashes}
    assert cleaned_hashes_by_type["producto"] == ("producto", {"1"}, "conninfo")
    assert cleaned_hashes_by_type["documento"] == ("documento", {"a.txt"}, "conninfo")


def test_build_index_from_documents_respects_explicit_source_types_to_clean(monkeypatch):
    documents = [SourceDocument(source_type="producto", source_id="1", text="hola")]

    _stub_no_previous_hashes(monkeypatch)
    monkeypatch.setattr(build_index_module, "chunk_fixed", lambda text: [text])
    monkeypatch.setattr(build_index_module, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(build_index_module, "upsert_chunks", lambda *a, **k: None)
    monkeypatch.setattr(build_index_module, "upsert_indexed_sources", lambda *a, **k: None)

    cleaned = []
    monkeypatch.setattr(
        build_index_module,
        "delete_stale_chunks",
        lambda source_type, valid_ids, max_chunk_index, conninfo: cleaned.append(source_type),
    )
    monkeypatch.setattr(build_index_module, "delete_stale_indexed_sources", lambda *a, **k: None)

    build_index_module.build_index_from_documents(documents, "conninfo", source_types_to_clean=set())

    assert cleaned == []


def test_build_index_only_cleans_documento_when_documents_dir_given(monkeypatch):
    monkeypatch.setattr(build_index_module, "_structured_documents", lambda conninfo: [])

    captured = {}

    def fake_build_index_from_documents(documents, conninfo, source_types_to_clean=None):
        captured["source_types_to_clean"] = source_types_to_clean
        return {"count": 0}

    monkeypatch.setattr(build_index_module, "build_index_from_documents", fake_build_index_from_documents)

    build_index_module.build_index("conninfo")
    assert captured["source_types_to_clean"] == {"producto"}

    monkeypatch.setattr(build_index_module, "load_text_documents", lambda directory: [])
    build_index_module.build_index("conninfo", documents_dir="docs/rag-sources")
    assert captured["source_types_to_clean"] == {"producto", "documento"}

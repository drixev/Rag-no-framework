"""Temas 2-4-10: indexado offline — carga fuentes, salta las que no cambiaron, trocea, genera embeddings y guarda en pgvector."""
import hashlib

from rag.config import settings
from rag.embeddings.openai_embeddings import embed_texts
from rag.ingestion.chunking import chunk_fixed
from rag.ingestion.loaders import (
    SourceDocument,
    StructuredSource,
    load_structured_source,
    load_text_documents,
)
from rag.ingestion.serialization import ProductRow, row_to_text
from rag.retrieval.index_store import (
    delete_stale_chunks,
    delete_stale_indexed_sources,
    get_indexed_hashes,
    upsert_chunks,
    upsert_indexed_sources,
)

# Tema 7 / mejora de escalado: registro declarativo de tablas indexables como
# RAG estructurado. Sumar una tabla nueva (ej. `pedidos`) es agregar una
# entrada acá — el resto del pipeline (hashing, chunking, embeddings, upsert)
# es genérico y no requiere cambios.
STRUCTURED_SOURCES: list[StructuredSource] = [
    StructuredSource(
        source_type="producto",
        query="SELECT id, nombre, categoria, precio, descripcion, stock FROM productos",
        id_field="id",
        serializer=lambda row: row_to_text(ProductRow(**row)),
    ),
]


def _hash_text(text: str) -> str:
    """
    Función: hashea el texto canónico de una fuente para detectar si cambió
    desde la última indexación.
    Llamada desde: build_index_from_documents()
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _structured_documents(conninfo: str) -> list[SourceDocument]:
    """
    Función: carga los documentos de todas las fuentes estructuradas
    registradas en STRUCTURED_SOURCES (Tema 7: RAG sobre SQL, ahora
    soportando varias tablas en vez de una sola hardcodeada).
    Llamada desde: build_index()
    """
    documents: list[SourceDocument] = []
    for source in STRUCTURED_SOURCES:
        documents += load_structured_source(source, conninfo)
    return documents


def build_index_from_documents(
    documents: list[SourceDocument],
    conninfo: str,
    source_types_to_clean: set[str] | None = None,
) -> dict:
    """
    Función: reindexa `documents` de forma incremental — compara el hash del
    texto de cada fuente contra lo ya indexado (`indexed_sources`) y solo
    trocea + embebe las que cambiaron o son nuevas, lo que evita re-embeber
    corpus enteros sin cambios en cada corrida. Al final borra los chunks y
    el bookkeeping huérfanos de cada source_type reindexado (fuentes
    eliminadas, o que ahora generan menos chunks que antes) — pipeline
    compartido por fuentes estructuradas y no estructuradas (Tema 7).
    Llamada desde: build_index()
    """
    source_types = {doc.source_type for doc in documents}

    previous_hashes: dict[tuple[str, str], str] = {}
    for source_type in source_types:
        previous_hashes.update(
            {
                (source_type, source_id): content_hash
                for source_id, content_hash in get_indexed_hashes(source_type, conninfo).items()
            }
        )

    doc_hashes = {(doc.source_type, doc.source_id): _hash_text(doc.text) for doc in documents}
    changed_documents = [
        doc for doc in documents if previous_hashes.get((doc.source_type, doc.source_id)) != doc_hashes[(doc.source_type, doc.source_id)]
    ]

    chunk_rows: list[tuple[str, str, int, str]] = [
        (doc.source_type, doc.source_id, i, chunk_text)
        for doc in changed_documents
        for i, chunk_text in enumerate(chunk_fixed(doc.text))
    ]

    texts = [row[3] for row in chunk_rows]
    vectors = embed_texts(texts) if texts else []

    upsert_chunks(chunk_rows, vectors, conninfo)

    chunk_counts: dict[tuple[str, str], int] = {}
    for source_type, source_id, chunk_index, _ in chunk_rows:
        key = (source_type, source_id)
        chunk_counts[key] = max(chunk_counts.get(key, 0), chunk_index + 1)

    upsert_indexed_sources(
        [
            (doc.source_type, doc.source_id, doc_hashes[(doc.source_type, doc.source_id)], chunk_counts.get((doc.source_type, doc.source_id), 0))
            for doc in changed_documents
        ],
        conninfo,
    )

    types_to_clean = source_types_to_clean if source_types_to_clean is not None else source_types
    for source_type in types_to_clean:
        valid_ids = {doc.source_id for doc in documents if doc.source_type == source_type}
        max_chunk_index = {
            source_id: count - 1 for (st, source_id), count in chunk_counts.items() if st == source_type
        }
        delete_stale_chunks(source_type, valid_ids, max_chunk_index, conninfo)
        delete_stale_indexed_sources(source_type, valid_ids, conninfo)

    return {
        "count": len(chunk_rows),
        "sources_total": len(documents),
        "sources_reindexed": len(changed_documents),
        "sources_skipped": len(documents) - len(changed_documents),
    }


def build_index(conninfo: str, documents_dir: str | None = None) -> dict:
    """
    Función: punto de entrada de la (re)indexación offline — siempre reindexa
    las fuentes estructuradas registradas en STRUCTURED_SOURCES y, si se pasa
    `documents_dir`, también los documentos de texto de esa carpeta. La
    reindexación es incremental (ver build_index_from_documents): solo
    re-embebe fuentes cuyo contenido cambió desde la corrida anterior. Solo
    limpia chunks huérfanos de los source_types efectivamente reindexados en
    esta corrida, para no borrar documentos indexados si no se pasó
    `documents_dir`.
    Llamada desde: scripts.ingest (CLI offline)
    """
    documents = _structured_documents(conninfo)
    source_types_to_clean = {source.source_type for source in STRUCTURED_SOURCES}
    if documents_dir:
        documents += load_text_documents(documents_dir)
        source_types_to_clean.add("documento")

    return build_index_from_documents(documents, conninfo, source_types_to_clean)


if __name__ == "__main__":
    result = build_index(settings.database_url)
    print(f"Chunks indexados en pgvector: {result['count']}")

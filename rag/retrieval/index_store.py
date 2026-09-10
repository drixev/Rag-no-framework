"""Tema 4: adaptador de vector store — pgvector sobre la misma Postgres del proyecto."""
from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector


@contextmanager
def _connect(conninfo: str) -> Iterator[psycopg.Connection]:
    """
    Función: abre una conexión a Postgres con el tipo `vector` de pgvector
    registrado, para poder pasar/leer embeddings como listas de floats.
    Llamada desde: upsert_chunks(), query_similar()
    """
    with psycopg.connect(conninfo) as conn:
        register_vector(conn)
        yield conn


def upsert_chunks(
    chunk_rows: list[tuple[str, str, int, str]],
    vectors: list[list[float]],
    conninfo: str,
) -> None:
    """
    Función: guarda (o actualiza) en `chunks` el texto y el embedding de cada
    fragmento indexado, sea de un producto o de un documento.
    Llamada desde: ingestion.build_index.build_index_from_documents()
    """
    if not chunk_rows:
        return

    with _connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (source_type, source_id, chunk_index, chunk_text, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source_type, source_id, chunk_index)
                DO UPDATE SET chunk_text = EXCLUDED.chunk_text, embedding = EXCLUDED.embedding
                """,
                [
                    (source_type, source_id, chunk_index, chunk_text, vector)
                    for (source_type, source_id, chunk_index, chunk_text), vector in zip(chunk_rows, vectors)
                ],
            )


def delete_stale_chunks(
    source_type: str,
    valid_source_ids: set[str],
    max_chunk_index: dict[str, int],
    conninfo: str,
) -> None:
    """
    Función: borra de `chunks` los fragmentos huérfanos de `source_type` — de
    fuentes que ya no existen en esta reindexación, o de fuentes que ahora
    generan menos chunks que en una indexación previa.
    Llamada desde: ingestion.build_index.build_index_from_documents()
    """
    with _connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE source_type = %s AND NOT (source_id = ANY(%s))",
                (source_type, list(valid_source_ids)),
            )
            if max_chunk_index:
                cur.executemany(
                    "DELETE FROM chunks WHERE source_type = %s AND source_id = %s AND chunk_index > %s",
                    [(source_type, source_id, idx) for source_id, idx in max_chunk_index.items()],
                )


def get_indexed_hashes(source_type: str, conninfo: str) -> dict[str, str]:
    """
    Función: devuelve {source_id: content_hash} de lo ya indexado para
    `source_type`, para que build_index_from_documents() detecte qué fuentes
    cambiaron desde la corrida anterior y salte las que no (Tema 10: reindex
    incremental).
    Llamada desde: ingestion.build_index.build_index_from_documents()
    """
    with _connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT source_id, content_hash FROM indexed_sources WHERE source_type = %s",
                (source_type,),
            )
            rows = cur.fetchall()

    return dict(rows)


def upsert_indexed_sources(rows: list[tuple[str, str, str, int]], conninfo: str) -> None:
    """
    Función: guarda (o actualiza) el hash de contenido y la cantidad de
    chunks de cada fuente reindexada, usado en la próxima corrida para saltar
    las fuentes sin cambios (Tema 10: reindex incremental).
    Llamada desde: ingestion.build_index.build_index_from_documents()
    """
    if not rows:
        return

    with _connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO indexed_sources (source_type, source_id, content_hash, chunk_count)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_type, source_id)
                DO UPDATE SET content_hash = EXCLUDED.content_hash, chunk_count = EXCLUDED.chunk_count
                """,
                rows,
            )


def delete_stale_indexed_sources(source_type: str, valid_source_ids: set[str], conninfo: str) -> None:
    """
    Función: borra de `indexed_sources` el bookkeeping de fuentes de
    `source_type` que ya no existen en esta reindexación, en paralelo a
    delete_stale_chunks() (Tema 10: reindex incremental).
    Llamada desde: ingestion.build_index.build_index_from_documents()
    """
    with _connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM indexed_sources WHERE source_type = %s AND NOT (source_id = ANY(%s))",
                (source_type, list(valid_source_ids)),
            )


def query_hybrid(query_vector: list[float], query_text: str, k: int, conninfo: str) -> list[dict]:
    """
    Función: recupera los k chunks (de cualquier fuente) más relevantes,
    fusionando el ranking denso (pgvector, distancia coseno) y el léxico BM25
    (pg_search) con Reciprocal Rank Fusion (RRF), calculado en una sola query SQL.
    Llamada desde: retrieval.retriever.retrieve()
    """
    candidate_k = max(k * 5, 20)

    with _connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH semantic_ranked AS (
                    SELECT id, RANK() OVER (ORDER BY embedding <=> %(query_vector)s::vector) AS rank
                    FROM chunks
                    ORDER BY embedding <=> %(query_vector)s::vector
                    LIMIT %(candidate_k)s
                ),
                bm25_ranked AS (
                    SELECT id, RANK() OVER (ORDER BY paradedb.score(id) DESC) AS rank
                    FROM chunks
                    WHERE chunk_text @@@ %(query_text)s
                    LIMIT %(candidate_k)s
                ),
                fused AS (
                    SELECT
                        COALESCE(s.id, b.id) AS id,
                        COALESCE(1.0 / (60 + s.rank), 0.0) + COALESCE(1.0 / (60 + b.rank), 0.0) AS score
                    FROM semantic_ranked s
                    FULL OUTER JOIN bm25_ranked b ON s.id = b.id
                )
                SELECT c.source_type, c.source_id, c.chunk_text, f.score
                FROM fused f
                JOIN chunks c ON c.id = f.id
                ORDER BY f.score DESC
                LIMIT %(k)s
                """,
                {
                    "query_vector": query_vector,
                    "query_text": query_text,
                    "candidate_k": candidate_k,
                    "k": k,
                },
            )
            rows = cur.fetchall()

    return [
        {"id": f"{source_type}:{source_id}", "text": chunk_text, "score": float(score)}
        for source_type, source_id, chunk_text, score in rows
    ]

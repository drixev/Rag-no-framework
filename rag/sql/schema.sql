CREATE TABLE productos (
    id SERIAL PRIMARY KEY,
    nombre TEXT NOT NULL,
    categoria TEXT,               -- puede ser NULL
    precio NUMERIC(10, 2) NOT NULL,
    descripcion TEXT,             -- puede ser NULL
    stock INTEGER NOT NULL DEFAULT 0
);

-- Tema 4: vector store (pgvector) — un chunk por fila, de cualquier fuente
-- (producto o documento). Dimensión 1536 = text-embedding-3-small (ver
-- docs/stack-decisions.md); si cambia el modelo de embeddings, cambia esta columna.
CREATE EXTENSION IF NOT EXISTS vector;

-- Tema 5: búsqueda léxica BM25 (pg_search / ParadeDB) para retrieval híbrido
-- junto a pgvector. Requiere la imagen paradedb/paradedb (ver docker-compose.yml).
CREATE EXTENSION IF NOT EXISTS pg_search;

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,     -- 'producto' | 'documento'
    source_id TEXT NOT NULL,       -- id del producto, o nombre de archivo del documento
    chunk_index INTEGER NOT NULL DEFAULT 0,
    chunk_text TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    UNIQUE (source_type, source_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Solo puede existir un índice BM25 por tabla (limitación de pg_search).
CREATE INDEX IF NOT EXISTS chunks_bm25_idx
    ON chunks USING bm25 (id, chunk_text)
    WITH (key_field = 'id');

-- Tema 10: bookkeeping para reindex incremental — guarda el hash del texto
-- canónico de cada fuente ya indexada, para que build_index_from_documents()
-- salte las fuentes cuyo contenido no cambió desde la corrida anterior.
CREATE TABLE IF NOT EXISTS indexed_sources (
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    PRIMARY KEY (source_type, source_id)
);
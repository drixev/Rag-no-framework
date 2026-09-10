# rag-project

Backend de RAG (proyecto de aprendizaje): Python + FastAPI + uv. Cada módulo
bajo `rag/` corresponde a un tema del temario. Las decisiones de stack
(vector DB, chunking, orquestación, re-ranking) están en
`docs/stack-decisions.md`.

## Setup

1. Instala [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Levanta Postgres con Docker Compose — usa la imagen `paradedb/paradedb`,
   que trae `pgvector` (retrieval denso) y `pg_search` (BM25) instalados;
   con tu propio Postgres necesitás ambas extensiones (ver
   `docs/stack-decisions.md`).
3. Copia `.env.example` a `.env` y completa tus API keys y `DATABASE_URL`.
4. Instala dependencias, crea el esquema, indexa los productos y corre la API.

```bash
docker-compose up -d

cp .env.example .env  # completa tus API keys; DATABASE_URL por defecto: postgresql://rag:rag@localhost:5432/rag

uv sync

psql "$DATABASE_URL" -f rag/sql/schema.sql -f rag/sql/seed.sql

uv run python scripts/ingest.py
# agrega --documents-dir ruta/a/docs para indexar también .txt/.md
# la reindexación es incremental: salta las fuentes sin cambios desde la corrida anterior

uv run uvicorn api.main:app --reload

uv run pytest
```

### Endpoints

- `POST /query`: RAG no estructurado, respuesta completa.
- `POST /query/stream`: igual que `/query`, con streaming SSE (evento `sources` seguido de eventos `chunk`).
- `POST /query/sql`: RAG estructurado (text-to-SQL) sobre `productos`.

```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "¿tienen webcams disponibles?"}'

curl -N -X POST localhost:8000/query/stream -H "Content-Type: application/json" \
  -d '{"query": "¿tienen webcams disponibles?"}'

curl -X POST localhost:8000/query/sql -H "Content-Type: application/json" \
  -d '{"query": "¿cuántos productos hay agotados?"}'
```

## Arquitectura de retrieval

`rag/pipeline.py` recupera contexto vía `retrieval/retriever.py`, que delega
en `retrieval/index_store.py` (pgvector + pg_search). La indexación offline
(`scripts/ingest.py` → `ingestion/build_index.py`) trocea cada fuente
(`ingestion/chunking.py`, fijo con overlap), genera embeddings y los guarda
en `chunks` sin distinguir origen: productos y documentos son intercambiables
para el retrieval (Tema 7).

La indexación escala de tres formas:

1. **Incremental** — hashea el texto de cada fuente y solo re-embebe lo que
   cambió (bookkeeping en `indexed_sources`).
2. **Batching** — `embeddings/openai_embeddings.py` manda los textos a
   OpenAI en lotes, no en un solo request.
3. **Fuentes configurables** — `ingestion/build_index.py::STRUCTURED_SOURCES`
   es una lista declarativa de tablas (query + serializador); sumar una
   tabla nueva no toca el pipeline.

`retrieval/index_store.query_hybrid` fusiona ranking denso (pgvector,
distancia coseno) y BM25 (pg_search) con Reciprocal Rank Fusion, en una sola
query. Sin re-ranking adicional por ahora (ver `docs/stack-decisions.md`).

## Estado del proyecto

- [x] Tema 1 — Fundamentos: embeddings (OpenAI) + generación (Claude)
- [x] Tema 2 — Ingesta desde DB y documentos de texto (`ingestion/loaders.py`, `ingestion/build_index.py`)
- [x] Tema 3 — Chunking fijo con overlap (`ingestion/chunking.py`)
- [x] Tema 4 — Vector DB: pgvector (`retrieval/index_store.py`)
- [x] Tema 5 — Búsqueda híbrida: pgvector + BM25, fusión RRF (`retrieval/index_store.py`); sin re-ranking (ver `docs/stack-decisions.md`)
- [x] Tema 6 — Text-to-SQL con validación de solo-lectura (`rag/sql/text_to_sql.py`)
- [x] Tema 7 — RAG estructurado (`/query/sql`) y no estructurado (`/query`), índice de chunks compartido
- [x] Tema 8 — Streaming SSE en `/query/stream` (orquestación custom, sin framework)
- [ ] Tema 10 — Optimización y producción: reindexación incremental, batching y fuentes configurables ya hechos (`ingestion/build_index.py`, `embeddings/openai_embeddings.py`); falta caching, manejo de errores y seguridad ante prompt injection

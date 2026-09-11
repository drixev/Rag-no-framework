| Componente | Elección |
|---|---|
| Lenguaje backend | Python |
| Modelo de embeddings | OpenAI `text-embedding-3-small` |
| Vector DB | pgvector (extensión sobre la misma Postgres del proyecto) |
| Framework de orquestación | Custom (sin framework) — funciones Python explícitas en `rag/pipeline.py` |
| Motor de re-ranking | Ninguno por ahora |
| Estrategia de chunking | Fijo con overlap |
| Motor de búsqueda léxica (BM25) / híbrida | ParadeDB `pg_search` (BM25 real) sobre la misma Postgres, fusionado con pgvector vía RRF en SQL |

| Componente | Elección | Sesión en que se decidió |
|---|---|---|
| Lenguaje backend | Python | — |
| Modelo de embeddings | OpenAI `text-embedding-3-small` | — (inferido del código existente: `rag/config.py`, `.env.example`; confirmar en tutoría) |
| Vector DB | pgvector (extensión sobre la misma Postgres del proyecto) | Confirmado en Claude Code — 2026-09-04 |
| Framework de orquestación | Custom (sin framework) — funciones Python explícitas en `rag/pipeline.py` | Confirmado en Claude Code — 2026-09-04 |
| Motor de re-ranking | Ninguno por ahora | Confirmado en Claude Code — 2026-09-04 |
| Estrategia de chunking | Fijo con overlap | Confirmado en Claude Code — 2026-09-04 |
| Motor de búsqueda léxica (BM25) / híbrida | ParadeDB `pg_search` (BM25 real) sobre la misma Postgres, fusionado con pgvector vía RRF en SQL | Confirmado en Claude Code — 2026-09-07 |

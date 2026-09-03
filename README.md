# rag-project

Backend de RAG (proyecto de aprendizaje) — Python + FastAPI + uv.
Cada módulo bajo `rag/` corresponde a un tema del temario; los que aún no se
han cubierto son placeholders con un docstring indicando el tema pendiente.

## Setup

1. Instala uv: https://docs.astral.sh/uv/getting-started/installation/
2. Copia `.env.example` a `.env` y completa tus API keys reales.
3. Instala dependencias: `uv sync`
4. Corre la API: `uv run uvicorn api.main:app --reload`
5. Prueba el endpoint: `curl -X POST localhost:8000/query -H "Content-Type: application/json" -d '{"query": "¿Cuánto tiempo tengo para devolver un producto?"}'`

## Estado del proyecto

- [x] Tema 1 — Fundamentos: embeddings (OpenAI) + generación (Claude) + retrieval naive en memoria
- [ ] Tema 2 — Embeddings y chunking a fondo (reemplaza el corpus hardcodeado de `retriever.py`)
- [ ] Tema 3 — Estrategias de chunking
- [ ] Tema 4 — Vector DB real (reemplaza el numpy en memoria)
- [ ] Tema 5 — Búsqueda híbrida y re-ranking
- [ ] Tema 6 — Text-to-SQL
- [ ] Tema 7 — RAG estructurado vs. no estructurado
- [ ] Tema 8 — Orquestación completa + streaming SSE real en `api/main.py`
- [ ] Tema 9 — Evaluación (RAGAS)
- [ ] Tema 10 — Optimización y producción
- [ ] Tema 11 — Patrones avanzados

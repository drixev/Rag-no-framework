"""
Retrieval híbrido (Tema 4/5): fusiona búsqueda densa (pgvector, coseno) y
léxica BM25 (pg_search) directamente en SQL vía RRF. Sin re-ranking por ahora
(ver docs/stack-decisions.md).
"""
from rag.config import settings
from rag.embeddings.openai_embeddings import embed_texts
from rag.retrieval.index_store import query_hybrid


def retrieve(query: str, k: int | None = None) -> list[dict]:
    """
    Función: recupera los top_k chunks más relevantes para `query` combinando
    búsqueda densa y léxica (BM25) sobre los chunks indexados.
    Llamada desde: rag.pipeline._build_user_message()
    """
    k = k or settings.retrieval_top_k
    query_vector = embed_texts([query])[0]
    return query_hybrid(query_vector, query, k, settings.database_url)

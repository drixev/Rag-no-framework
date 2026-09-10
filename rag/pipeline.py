"""Orquestación end-to-end del pipeline — Temas 1 (síncrono) y 8 (streaming)."""
from collections.abc import Iterator

from rag.generation.claude_client import generate, generate_stream
from rag.generation.prompts import GENERATION_SYSTEM_PROMPT
from rag.retrieval.retriever import retrieve


def _build_user_message(query: str) -> tuple[str, list[dict]]:
    """
    Función: recupera el contexto relevante y arma el mensaje de usuario aumentado,
    compartido por la variante síncrona y la variante en streaming del pipeline.
    Llamada desde: rag_pipeline(), rag_pipeline_stream()
    """
    retrieved = retrieve(query)
    context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in retrieved)
    user_message = f"CONTEXTO:\n{context}\n\nPREGUNTA:\n{query}"
    return user_message, retrieved


def rag_pipeline(query: str) -> str:
    """
    Función: orquesta el flujo RAG completo — recupera contexto, arma el prompt
    aumentado y genera la respuesta final de una sola vez.
    Llamada desde: api.main.query()
    """
    user_message, _ = _build_user_message(query)
    return generate(GENERATION_SYSTEM_PROMPT, user_message)


def rag_pipeline_stream(query: str) -> Iterator[tuple[str, object]]:
    """
    Función: variante en streaming del pipeline RAG — primero entrega las fuentes
    recuperadas (evento "sources") y luego va emitiendo fragmentos de la
    respuesta a medida que el modelo la genera (eventos "chunk").
    Llamada desde: api.main.query_stream()
    """
    user_message, retrieved = _build_user_message(query)
    yield "sources", retrieved
    for chunk in generate_stream(GENERATION_SYSTEM_PROMPT, user_message):
        yield "chunk", chunk

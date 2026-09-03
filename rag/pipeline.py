"""Orquestación end-to-end del pipeline — versión básica del Tema 1."""
from rag.generation.claude_client import generate
from rag.generation.prompts import GENERATION_SYSTEM_PROMPT
from rag.retrieval.retriever import retrieve


def rag_pipeline(query: str) -> str:
    retrieved = retrieve(query)
    context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in retrieved)
    user_message = f"CONTEXTO:\n{context}\n\nPREGUNTA:\n{query}"
    return generate(GENERATION_SYSTEM_PROMPT, user_message)

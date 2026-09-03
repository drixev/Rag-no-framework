"""
Retrieval NAIVE: matriz en memoria + similitud coseno (Tema 1).
Sirve para ver el mecanismo end-to-end sin infraestructura.
Se reemplaza por un vector DB real en el Tema 4 (pgvector/Pinecone/Chroma/FAISS)
y el corpus hardcodeado se reemplaza por ingesta real en el Tema 2.
"""
import numpy as np

from rag.config import settings
from rag.embeddings.openai_embeddings import embed_texts

DOCUMENTS = [
    {"id": "doc1", "text": "La política de devoluciones permite cambios dentro de los 30 días posteriores a la compra, siempre que el producto esté sin usar."},
    {"id": "doc2", "text": "El horario de atención al cliente es de lunes a viernes, de 9:00 a 18:00 hora local."},
    {"id": "doc3", "text": "Los envíos internacionales tardan entre 7 y 15 días hábiles dependiendo del país de destino."},
    {"id": "doc4", "text": "Para cancelar una suscripción, el usuario debe ingresar a Configuración > Facturación > Cancelar plan."},
    {"id": "doc5", "text": "Aceptamos pagos con tarjeta de crédito, débito y PayPal. No aceptamos transferencias bancarias directas."},
]

_doc_embeddings = embed_texts([d["text"] for d in DOCUMENTS])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a_norm @ b_norm.T


def retrieve(query: str, k: int | None = None) -> list[dict]:
    k = k or settings.retrieval_top_k
    query_embedding = embed_texts([query])
    similarities = _cosine_similarity(query_embedding, _doc_embeddings)[0]
    top_k_idx = np.argsort(similarities)[::-1][:k]
    return [{**DOCUMENTS[i], "score": float(similarities[i])} for i in top_k_idx]

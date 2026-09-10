from openai import OpenAI
from rag.config import settings

_client = OpenAI(api_key=settings.openai_api_key)

# OpenAI acepta hasta 2048 inputs por request, pero un lote más chico evita
# requests gigantes (y sus timeouts) cuando el corpus tiene miles de chunks.
DEFAULT_EMBEDDING_BATCH_SIZE = 200


def embed_texts(texts: list[str], batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE) -> list[list[float]]:
    """
    Función: genera embeddings para `texts`, partiéndolos en lotes de
    `batch_size` para no exceder los límites de tamaño de request de la API
    de OpenAI en corpus grandes (Tema 10: escalado del pipeline de ingesta).
    Llamada desde: ingestion.build_index.build_index_from_documents() y
    retrieval.retriever.retrieve()
    """
    if not texts:
        return []

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = _client.embeddings.create(model=settings.embedding_model, input=batch)
        vectors.extend(item.embedding for item in response.data)

    return vectors

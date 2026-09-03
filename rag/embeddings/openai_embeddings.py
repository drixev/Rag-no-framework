"""Wrapper del proveedor de embeddings (OpenAI) — implementado en el Tema 1."""
import numpy as np
from openai import OpenAI
from typing import List

from rag.config import settings

_client = OpenAI(api_key=settings.openai_api_key)

def embed_texts(texts: list[str]) -> np.ndarray:
    """Convierte una lista de textos en vectores usando el modelo configurado."""
    response = _client.embeddings.create(model=settings.embedding_model, input=texts)
    return np.array([item.embedding for item in response.data])



"""Wrapper del proveedor de generación (Claude) — implementado en los Temas 1 y 8."""
from collections.abc import Iterator

from anthropic import Anthropic

from rag.config import settings

_client = Anthropic(api_key=settings.anthropic_api_key)


def generate(system_prompt: str, user_message: str, max_tokens: int = 300) -> str:
    """
    Función: llama al modelo de generación configurado con un system prompt y un
    mensaje de usuario, devolviendo la respuesta completa (sin streaming).
    Llamada desde: rag.pipeline.rag_pipeline(), rag.sql.text_to_sql.generate_sql()
    """
    response = _client.messages.create(
        model=settings.generation_model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def generate_stream(system_prompt: str, user_message: str, max_tokens: int = 300) -> Iterator[str]:
    """
    Función: llama al modelo de generación en modo streaming, produciendo el texto
    de la respuesta incrementalmente a medida que el modelo la genera.
    Llamada desde: rag.pipeline.rag_pipeline_stream()
    """
    with _client.messages.stream(
        model=settings.generation_model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        yield from stream.text_stream

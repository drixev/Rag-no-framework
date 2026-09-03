"""Wrapper del proveedor de generación (Claude) — implementado en el Tema 1."""
from anthropic import Anthropic

from rag.config import settings

_client = Anthropic(api_key=settings.anthropic_api_key)


def generate(system_prompt: str, user_message: str, max_tokens: int = 300) -> str:
    """Llama al modelo de generación configurado con un system prompt y un mensaje de usuario."""
    response = _client.messages.create(
        model=settings.generation_model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text

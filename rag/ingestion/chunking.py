"""Tema 3: chunking fijo con overlap — estrategia decidida por el usuario (ver docs/stack-decisions.md)."""


def chunk_fixed(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Función: trocea `text` en fragmentos de hasta `chunk_size` caracteres, con
    `overlap` caracteres de solapamiento entre fragmentos consecutivos. Si el
    texto completo cabe en un solo chunk, lo devuelve sin trocear.
    Llamada desde: ingestion.build_index.build_index_from_documents()
    """
    text = text.strip()
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap debe ser menor que chunk_size")

    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks

"""Tema 2/7: carga de fuentes heterogéneas — filas de BD (estructurado) y documentos de texto (no estructurado)."""
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.rows import dict_row


@dataclass
class SourceDocument:
    """Unidad de contenido fuente, sin trocear, lista para chunking + embeddings."""

    source_type: str  # "producto" | "documento"
    source_id: str
    text: str


@dataclass
class StructuredSource:
    """
    Declara una tabla de BD indexable como RAG estructurado: qué query
    correrle, qué columna identifica cada fila, y cómo serializar la fila a
    texto canónico. Sumar una tabla nueva es agregar una entrada en
    ingestion.build_index.STRUCTURED_SOURCES, sin tocar el resto del pipeline
    de indexación (Tema 7: interfaz común de fuentes).
    """

    source_type: str
    query: str
    id_field: str
    serializer: Callable[[dict], str]


def load_structured_source(source: StructuredSource, conninfo: str) -> list[SourceDocument]:
    """
    Función: ejecuta la query de `source` sobre Postgres y serializa cada
    fila resultante a un SourceDocument — genérico para cualquier tabla
    estructurada registrada, no solo `productos`.
    Llamada desde: ingestion.build_index._structured_documents()
    """
    with psycopg.connect(conninfo, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(source.query)
            rows = cur.fetchall()

    return [
        SourceDocument(
            source_type=source.source_type,
            source_id=str(row[source.id_field]),
            text=source.serializer(row),
        )
        for row in rows
    ]


def load_text_documents(directory: str) -> list[SourceDocument]:
    """
    Función: carga como documentos fuente (RAG no estructurado) todos los
    archivos .txt/.md de `directory`, uno por archivo.
    Llamada desde: ingestion.build_index.build_index()
    """
    base = Path(directory)
    paths = sorted(base.glob("*.txt")) + sorted(base.glob("*.md"))

    return [
        SourceDocument(source_type="documento", source_id=path.name, text=path.read_text(encoding="utf-8"))
        for path in paths
    ]

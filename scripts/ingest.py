"""CLI de indexación offline (Temas 2-4): reindexa productos y, opcionalmente, documentos de texto."""
import argparse

from rag.config import settings
from rag.ingestion.build_index import build_index


def main() -> None:
    """
    Función: parsea los argumentos de línea de comandos y dispara la
    reindexación completa hacia pgvector.
    Llamada desde: invocación directa (`uv run python scripts/ingest.py`)
    """
    parser = argparse.ArgumentParser(description="Reindexar fuentes RAG en pgvector.")
    parser.add_argument(
        "--documents-dir",
        default=None,
        help="Carpeta con documentos .txt/.md a indexar además de los productos",
    )
    args = parser.parse_args()

    result = build_index(settings.database_url, documents_dir=args.documents_dir)
    print(
        f"Chunks indexados en pgvector: {result['count']} "
        f"(fuentes reindexadas: {result['sources_reindexed']}, "
        f"sin cambios: {result['sources_skipped']} de {result['sources_total']})"
    )


if __name__ == "__main__":
    main()

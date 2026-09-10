from dataclasses import dataclass


@dataclass
class ProductRow:
    id: int
    nombre: str
    categoria: str | None
    precio: float
    descripcion: str | None
    stock: int

def row_to_text(row: ProductRow) -> str:
    """
    Función: serializa una fila de producto a un texto canónico, citable, que
    normaliza campos NULL (categoría, descripción) a valores legibles.
    Llamada desde: ingestion.build_index.build_index()
    """
    categoria = row.categoria or "sin categoría"
    descripcion = row.descripcion.strip().rstrip(".") if row.descripcion else "sin descripción disponible"
    disponibilidad = "disponible" if row.stock > 0 else "agotado"


    return (
        f"Producto: {row.nombre}. "
        f"Categoría: {categoria}. "
        f"Precio: ${row.precio:.2f}. "
        f"Descripción: {descripcion}. "
        f"Estado de stock: {disponibilidad}."
    )
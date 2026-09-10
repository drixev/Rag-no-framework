from rag.ingestion.serialization import ProductRow, row_to_text


def test_row_to_text_with_all_fields():
    row = ProductRow(
        id=1, nombre="Mouse", categoria="Periféricos", precio=19.99,
        descripcion="Un mouse.", stock=5,
    )

    text = row_to_text(row)

    assert "Mouse" in text
    assert "Periféricos" in text
    assert "$19.99" in text
    assert "disponible" in text


def test_row_to_text_with_missing_fields():
    row = ProductRow(id=2, nombre="Silla", categoria=None, precio=100.0, descripcion=None, stock=0)

    text = row_to_text(row)

    assert "sin categoría" in text
    assert "sin descripción disponible" in text
    assert "agotado" in text

import pytest

from rag.sql import text_to_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM productos",
        "select nombre, precio from productos where stock > 0",
        "SELECT * FROM productos;",
    ],
)
def test_validate_sql_accepts_safe_select(sql):
    assert text_to_sql.validate_sql(sql).lower().startswith("select")


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE productos",
        "SELECT * FROM productos; DROP TABLE productos",
        "UPDATE productos SET precio = 0",
        "SELECT * FROM usuarios",
        "SELECT * FROM productos -- comentario",
    ],
)
def test_validate_sql_rejects_unsafe_queries(sql):
    with pytest.raises(text_to_sql.UnsafeSQLError):
        text_to_sql.validate_sql(sql)


def test_generate_sql_returns_none_for_no_query(monkeypatch):
    monkeypatch.setattr(
        text_to_sql, "generate", lambda system_prompt, question, max_tokens=200: "NO_QUERY"
    )

    assert text_to_sql.generate_sql("¿cuál es la capital de Francia?") is None


def test_generate_sql_strips_single_backticks(monkeypatch):
    monkeypatch.setattr(
        text_to_sql,
        "generate",
        lambda system_prompt, question, max_tokens=200: "`SELECT * FROM productos`",
    )

    sql = text_to_sql.generate_sql("dame todos los productos")

    assert sql == "SELECT * FROM productos"

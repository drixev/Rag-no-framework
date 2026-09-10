"""Tema 6: text-to-SQL — genera y valida consultas NL→SQL de solo lectura sobre `productos`."""
import re
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from rag.config import settings
from rag.generation.claude_client import generate

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
_PROMPT_TEMPLATE = (_PROMPTS_DIR / "text_to_sql.txt").read_text(encoding="utf-8")
_SCHEMA = (Path(__file__).resolve().parent / "schema.sql").read_text(encoding="utf-8")

_ALLOWED_TABLE = "productos"
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|merge|call|copy|exec|execute)\b",
    re.IGNORECASE,
)


class UnsafeSQLError(ValueError):
    """Se lanza cuando el SQL generado no pasa la validación de seguridad."""


def generate_sql(question: str) -> str | None:
    """
    Función: traduce una pregunta en lenguaje natural a una consulta SQL de solo
    lectura sobre el esquema de `productos`, usando el modelo de generación
    configurado. Devuelve None si el modelo determina que no puede responderse.
    Llamada desde: run_text_to_sql()
    """
    system_prompt = _PROMPT_TEMPLATE.format(schema=_SCHEMA)
    raw_sql = generate(system_prompt, question, max_tokens=200).strip()
    raw_sql = raw_sql.strip("`").strip()

    if raw_sql.upper() == "NO_QUERY":
        return None
    return raw_sql


def validate_sql(sql: str) -> str:
    """
    Función: valida que el SQL generado sea una única sentencia SELECT de solo
    lectura que referencie la tabla permitida, antes de ejecutarlo. Lanza
    UnsafeSQLError si la consulta no pasa alguna de las validaciones.
    Llamada desde: run_text_to_sql()
    """
    stripped = sql.strip().rstrip(";").strip()

    if "--" in stripped or "/*" in stripped:
        raise UnsafeSQLError("No se permiten comentarios SQL en la consulta.")

    if ";" in stripped:
        raise UnsafeSQLError("No se permite más de una sentencia SQL.")

    if not re.match(r"^\s*select\b", stripped, re.IGNORECASE):
        raise UnsafeSQLError("Solo se permiten sentencias SELECT.")

    if _FORBIDDEN_KEYWORDS.search(stripped):
        raise UnsafeSQLError("La consulta contiene una palabra clave no permitida.")

    if _ALLOWED_TABLE.lower() not in stripped.lower():
        raise UnsafeSQLError(f"La consulta debe referenciar únicamente la tabla '{_ALLOWED_TABLE}'.")

    return stripped


def run_text_to_sql(question: str, conninfo: str | None = None) -> dict:
    """
    Función: orquesta el flujo completo de RAG estructurado — genera el SQL a
    partir de la pregunta, lo valida y lo ejecuta contra la base de datos,
    devolviendo tanto la consulta ejecutada como las filas resultantes.
    Llamada desde: api.main.query_sql()
    """
    conninfo = conninfo or settings.database_url
    sql = generate_sql(question)
    if sql is None:
        return {"sql": None, "rows": []}

    validated_sql = validate_sql(sql)

    with psycopg.connect(conninfo, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(validated_sql)
            rows = cur.fetchall()

    return {"sql": validated_sql, "rows": rows}

"""
API HTTP del backend RAG.
Tema 8: /query/stream expone la generación en streaming real vía SSE.
Tema 6/7: /query/sql expone el flujo de RAG estructurado (text-to-SQL).
"""
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import QueryRequest, QueryResponse, SQLQueryResponse
from rag.pipeline import rag_pipeline, rag_pipeline_stream
from rag.sql.text_to_sql import UnsafeSQLError, run_text_to_sql

app = FastAPI(title="RAG Backend")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    Función: endpoint síncrono de RAG no estructurado — devuelve la respuesta
    completa de una sola vez (sin streaming).
    Llamada desde: cliente HTTP (ver README).
    """
    answer = rag_pipeline(request.query)
    return QueryResponse(answer=answer)


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    """
    Función: endpoint de RAG no estructurado con streaming SSE — emite primero
    un evento "sources" con los documentos recuperados y luego eventos "chunk"
    con el texto de la respuesta a medida que el modelo la genera.
    Llamada desde: cliente HTTP compatible con Server-Sent Events.
    """

    def event_stream():
        for event, data in rag_pipeline_stream(request.query):
            yield f"event: {event}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/query/sql", response_model=SQLQueryResponse)
def query_sql(request: QueryRequest) -> SQLQueryResponse:
    """
    Función: endpoint de RAG estructurado — traduce la pregunta a SQL de solo
    lectura, la valida y la ejecuta contra la base de datos de productos.
    Llamada desde: cliente HTTP (ver README).
    """
    try:
        result = run_text_to_sql(request.query)
    except UnsafeSQLError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SQLQueryResponse(**result)

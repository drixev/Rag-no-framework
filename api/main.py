"""
Endpoint síncrono, SIN streaming todavía.
En el Tema 8 (Orquestación) se convierte a StreamingResponse con SSE.
"""
from fastapi import FastAPI

from api.schemas import QueryRequest, QueryResponse
from rag.pipeline import rag_pipeline

app = FastAPI(title="RAG Backend")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    answer = rag_pipeline(request.query)
    return QueryResponse(answer=answer)

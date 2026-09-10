from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


class SQLQueryResponse(BaseModel):
    sql: str | None
    rows: list[dict]

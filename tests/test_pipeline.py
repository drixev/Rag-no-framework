from rag import pipeline


def test_rag_pipeline_builds_context_and_generates(monkeypatch):
    monkeypatch.setattr(
        pipeline, "retrieve", lambda query: [{"id": "doc1", "text": "hola", "score": 0.9}]
    )

    captured = {}

    def fake_generate(system_prompt, user_message):
        captured["system_prompt"] = system_prompt
        captured["user_message"] = user_message
        return "respuesta"

    monkeypatch.setattr(pipeline, "generate", fake_generate)

    answer = pipeline.rag_pipeline("pregunta")

    assert answer == "respuesta"
    assert "[doc1] hola" in captured["user_message"]
    assert "pregunta" in captured["user_message"]


def test_rag_pipeline_stream_yields_sources_then_chunks(monkeypatch):
    sources = [{"id": "doc1", "text": "hola", "score": 0.9}]
    monkeypatch.setattr(pipeline, "retrieve", lambda query: sources)
    monkeypatch.setattr(
        pipeline, "generate_stream", lambda system_prompt, user_message: iter(["hola ", "mundo"])
    )

    events = list(pipeline.rag_pipeline_stream("pregunta"))

    assert events[0] == ("sources", sources)
    assert events[1:] == [("chunk", "hola "), ("chunk", "mundo")]

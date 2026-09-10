import pytest

from rag.ingestion.chunking import chunk_fixed


def test_chunk_fixed_returns_single_chunk_when_text_fits():
    assert chunk_fixed("hola mundo", chunk_size=100, overlap=10) == ["hola mundo"]


def test_chunk_fixed_splits_with_overlap():
    text = "a" * 25

    chunks = chunk_fixed(text, chunk_size=10, overlap=2)

    assert len(chunks) > 1
    assert all(len(c) <= 10 for c in chunks)


def test_chunk_fixed_returns_empty_list_for_blank_text():
    assert chunk_fixed("   ") == []


def test_chunk_fixed_rejects_overlap_greater_or_equal_to_chunk_size():
    with pytest.raises(ValueError):
        chunk_fixed("texto", chunk_size=10, overlap=10)

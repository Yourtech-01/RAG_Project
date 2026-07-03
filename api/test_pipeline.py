"""
test_pipeline.py  —  pytest unit tests
Run: pytest test_pipeline.py -v
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pipeline import chunk_text, extract_text

SAMPLE_TEXT = """
Introduction to machine learning. Machine learning is a subset of AI.
It enables computers to learn from data without being explicitly programmed.
There are three main types: supervised, unsupervised, and reinforcement learning.
Supervised learning uses labelled data to train models. Unsupervised learning
finds hidden patterns in unlabelled data. Reinforcement learning trains agents
via rewards and penalties. Deep learning is a subfield using neural networks.
""".strip()


def test_chunk_text_returns_list():
    chunks = chunk_text(SAMPLE_TEXT, "doc001", {"source": "test.txt"})
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_chunk_metadata():
    chunks = chunk_text(SAMPLE_TEXT, "doc001", {"source": "test.txt"})
    for c in chunks:
        assert c.doc_id == "doc001"
        assert "source" in c.metadata
        assert "chunk_index" in c.metadata


def test_chunk_ids_unique():
    chunks = chunk_text(SAMPLE_TEXT, "doc001", {"source": "test.txt"})
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_chunk_min_length():
    chunks = chunk_text(SAMPLE_TEXT, "doc001", {"source": "test.txt"})
    for c in chunks:
        assert len(c.text.strip()) >= 50, "All chunks must be >= 50 chars"


def test_chunk_text_empty():
    chunks = chunk_text("Short.", "doc002", {"source": "empty.txt"})
    assert isinstance(chunks, list)


def test_extract_txt(tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("Hello world from test file.", encoding="utf-8")
    text = extract_text(p)
    assert "Hello world" in text


def test_extract_md(tmp_path):
    p = tmp_path / "readme.md"
    p.write_text("# Title\n\nSome content here.", encoding="utf-8")
    text = extract_text(p)
    assert "Title" in text


def test_extract_unsupported(tmp_path):
    p = tmp_path / "test.xyz"
    p.write_text("data")
    import pytest
    with pytest.raises(ValueError):
        extract_text(p)

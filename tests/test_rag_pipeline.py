import json
from pathlib import Path

from backend.rag.retriever import ExampleRetriever


def _write_sample(path: Path) -> None:
    rows = [
        {
            "id": "1",
            "question": "How to groupby in pandas?",
            "answer": "Use df.groupby('team')['goals'].sum().",
            "packages_used": ["pandas"],
            "edu_score": 5,
            "primary_package": "pandas",
            "kaggle_dataset_name": "demo/a",
            "rag_text": "question: How to groupby in pandas?\nanswer: Use df.groupby('team')['goals'].sum().",
        },
        {
            "id": "2",
            "question": "How to create matplotlib plots?",
            "answer": "Use plt.plot(x, y).",
            "packages_used": ["matplotlib"],
            "edu_score": 4,
            "primary_package": "matplotlib",
            "kaggle_dataset_name": "demo/b",
            "rag_text": "question: How to create matplotlib plots?\nanswer: Use plt.plot(x, y).",
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_lexical_retrieval_returns_expected_match(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    _write_sample(processed / "sample_non_thinking_2.jsonl")

    retriever = ExampleRetriever(data_dir=processed, prefer_chroma=False)
    out = retriever.retrieve("pandas groupby", limit=1)

    assert len(out) == 1
    assert "groupby" in out[0]["question"].lower()


def test_chroma_index_noop_without_chroma_dependency(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    _write_sample(processed / "sample_non_thinking_2.jsonl")

    retriever = ExampleRetriever(
        data_dir=processed,
        chroma_dir=tmp_path / "chroma",
        collection_name="test_collection",
        prefer_chroma=False,
    )
    indexed = retriever.index_processed_files()
    assert indexed == 0

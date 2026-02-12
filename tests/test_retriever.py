import json
from pathlib import Path

from backend.rag.retriever import ExampleRetriever


def test_retriever_returns_ranked_examples(tmp_path: Path) -> None:
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_path = data_dir / "sample_non_thinking_2.jsonl"
    rows = [
        {"question": "How to groupby in pandas?", "answer": "Use df.groupby('col').sum()."},
        {"question": "How to train a model?", "answer": "Use scikit-learn pipeline."},
    ]
    with sample_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    retriever = ExampleRetriever(data_dir=data_dir)
    out = retriever.retrieve("pandas groupby", limit=1)

    assert len(out) == 1
    assert "groupby" in out[0]["question"].lower()

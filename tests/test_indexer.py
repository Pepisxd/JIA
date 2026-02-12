from backend.rag.indexer import _normalize_record, _pick_primary_package


def test_pick_primary_package_returns_unknown_when_empty() -> None:
    assert _pick_primary_package([]) == "unknown"
    assert _pick_primary_package(None) == "unknown"


def test_pick_primary_package_normalizes_first_value() -> None:
    assert _pick_primary_package([" Pandas ", "NumPy"]) == "pandas"


def test_normalize_record_shapes_expected_fields() -> None:
    row = {
        "id": "abc-123",
        "question": "  How to groupby? ",
        "answer": " Use pandas groupby. ",
        "edu_score": 5,
        "packages_used": ["Pandas", "numpy"],
        "files_used": ["file.csv"],
        "kaggle_dataset_name": "demo/ds",
        "executor_type": "e2b",
        "original_notebook": "{\"cells\": []}",
    }

    normalized = _normalize_record(row)

    assert normalized["id"] == "abc-123"
    assert normalized["question"] == "How to groupby?"
    assert normalized["answer"] == "Use pandas groupby."
    assert normalized["edu_score"] == 5
    assert normalized["primary_package"] == "pandas"
    assert normalized["notebook_excerpt"] == "{\"cells\": []}"

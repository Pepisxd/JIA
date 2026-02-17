import os

from fastapi.testclient import TestClient

os.environ["USE_LOCAL_MODEL"] = "false"
os.environ["USE_REAL_LLM"] = "false"

from backend.main import app  # noqa: E402


def test_generate_debug_attempts_matches_response_attempts() -> None:
    client = TestClient(app)
    payload = {
        "tema": "pandas_groupby",
        "nivel": "principiante",
        "contexto": "deportes",
        "tipo": "tutorial",
        "use_rag": False,
    }

    response = client.post("/generate?debug=true", json=payload)
    assert response.status_code == 200
    body = response.json()

    assert "debug" in body
    assert "attempts" in body["debug"]
    attempts_debug = body["debug"]["attempts"]
    assert isinstance(attempts_debug, list)
    assert len(attempts_debug) == body["attempts"]
    assert len(attempts_debug) >= 1

    first = attempts_debug[0]
    expected_keys = {
        "attempt",
        "model_backend",
        "used_fallback",
        "prompt_sent",
        "raw_text",
        "sanitized_text",
        "validation_errors",
        "parse_ok",
        "exec_ok",
        "exec_error",
    }
    assert expected_keys.issubset(first.keys())

from fastapi.testclient import TestClient

from backend.main import app


def test_history_detail_endpoint_returns_item() -> None:
    client = TestClient(app)
    payload = {
        "tema": "pandas_groupby",
        "nivel": "principiante",
        "contexto": "deportes",
        "tipo": "tutorial",
    }
    create = client.post("/generate", json=payload)
    assert create.status_code == 200

    history = client.get("/history?limit=1")
    assert history.status_code == 200
    latest = history.json()[0]

    detail = client.get(f"/history/{latest['id']}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["id"] == latest["id"]
    assert "codigo" in body
    assert "dataset_name" in body

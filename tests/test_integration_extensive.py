import itertools

import pytest
from fastapi.testclient import TestClient

from backend.main import app

TEMAS = ["pandas_groupby", "pandas_filtrado", "pandas_lectura"]
NIVELES = ["principiante", "intermedio", "avanzado"]
CONTEXTOS = ["deportes", "finanzas", "videojuegos", "ciencia"]


@pytest.mark.parametrize(
    "tema,nivel,contexto",
    list(itertools.product(TEMAS, NIVELES, CONTEXTOS)),
)
def test_generate_combinations_returns_structured_response(tema: str, nivel: str, contexto: str) -> None:
    client = TestClient(app)
    payload = {"tema": tema, "nivel": nivel, "contexto": contexto, "tipo": "tutorial"}

    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "codigo" in body and isinstance(body["codigo"], str)
    assert body["attempts"] >= 1
    assert "tests_passed" in body
    assert "educational_passed" in body


def test_metrics_endpoint_returns_dashboard() -> None:
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "total_generations" in body
    assert "success_rate" in body
    assert "by_tema" in body

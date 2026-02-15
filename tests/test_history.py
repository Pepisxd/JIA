from backend.db import init_db
from backend.models import DatasetInfo, GenerateRequest, GenerateResponse
from backend.store import list_history, save_generation


def test_save_and_list_history_roundtrip() -> None:
    init_db()
    request = GenerateRequest(
        tema="pandas_groupby",
        nivel="principiante",
        contexto="deportes",
        tipo="tutorial",
    )
    response = GenerateResponse(
        objetivo="obj",
        dataset=DatasetInfo(
            nombre="demo",
            data=[{"equipo": "A", "goles": 1}],
            codigo_carga="df = pd.DataFrame([...])",
        ),
        codigo="print('ok')",
        explicacion=["Paso 1", "Paso 2"],
        ejercicio="Ejercicio",
        tests_passed=True,
        educational_passed=True,
        validation_errors=[],
        attempts=1,
        output="ok",
        duration_ms=10.5,
    )
    new_id = save_generation(request, response)
    history = list_history(limit=1)
    assert history
    assert history[0].id == new_id
    assert history[0].tema == "pandas_groupby"

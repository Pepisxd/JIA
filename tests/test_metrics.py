from pathlib import Path

from backend.metrics import MetricsCollector
from backend.models import DatasetInfo, GenerateRequest, GenerateResponse


def _request() -> GenerateRequest:
    return GenerateRequest(
        tema="pandas_groupby",
        nivel="principiante",
        contexto="deportes",
        tipo="tutorial",
    )


def _response(ok: bool, edu_ok: bool, attempts: int) -> GenerateResponse:
    return GenerateResponse(
        objetivo="obj",
        dataset=DatasetInfo(nombre="d", data=[{"a": 1}], codigo_carga="df = pd.DataFrame([])"),
        codigo="print('ok')",
        explicacion=["Paso 1", "Paso 2"],
        ejercicio="Ej",
        tests_passed=ok,
        educational_passed=edu_ok,
        attempts=attempts,
        output="ok",
    )


def test_metrics_summary_aggregates_data(tmp_path: Path) -> None:
    collector = MetricsCollector(base_dir=tmp_path / "metrics")
    collector.record(request=_request(), response=_response(True, True, 1), duration_ms=100.0, status="success")
    collector.record(request=_request(), response=_response(False, False, 3), duration_ms=300.0, status="failed")

    summary = collector.summary()
    assert summary["total_generations"] == 2
    assert summary["success_rate"] == 50.0
    assert summary["educational_pass_rate"] == 50.0
    assert "pandas_groupby" in summary["by_tema"]

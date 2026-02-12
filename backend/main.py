from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI, HTTPException

from backend.executor import CodeExecutor
from backend.generator import CodeGenerator, GenerationError
from backend.metrics import MetricsCollector
from backend.models import GenerateRequest, GenerateResponse
from backend.repair import ValidatingRepairLoop
from backend.validators import EducationalValidator

app = FastAPI(
    title="Generador Educativo de Codigo Python",
    version="0.3.0",
    description="API MVP robusta con repair loop, validacion educativa y metricas.",
)

_generator = CodeGenerator()
_executor = CodeExecutor()
_validator = EducationalValidator()
_metrics = MetricsCollector()
_repair_loop = ValidatingRepairLoop(
    generator=_generator,
    executor=_executor,
    validator=_validator,
    max_attempts=3,
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "phase": "fase_1_5"}


@app.post("/generate", response_model=GenerateResponse)
def generate_example(payload: GenerateRequest) -> GenerateResponse:
    start = perf_counter()
    try:
        response = _repair_loop.run(payload)
        duration_ms = (perf_counter() - start) * 1000
        response.duration_ms = duration_ms
        status = "success" if response.tests_passed else "failed"
        _metrics.record(
            request=payload,
            response=response,
            duration_ms=duration_ms,
            status=status,
        )
        return response
    except GenerationError as exc:
        duration_ms = (perf_counter() - start) * 1000
        _metrics.record(
            request=payload,
            response=None,
            duration_ms=duration_ms,
            status="llm_error",
            error=str(exc),
        )
        raise HTTPException(status_code=502, detail=f"Error de generacion LLM: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        duration_ms = (perf_counter() - start) * 1000
        _metrics.record(
            request=payload,
            response=None,
            duration_ms=duration_ms,
            status="internal_error",
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=f"Error interno: {exc}") from exc


@app.get("/metrics")
def get_metrics() -> dict:
    return _metrics.summary()

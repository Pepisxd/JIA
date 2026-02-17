from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from time import perf_counter

from fastapi import FastAPI, HTTPException

from backend.db import init_db
from backend.executor import CodeExecutor
from backend.generator import CodeGenerator, GenerationError
from backend.metrics import MetricsCollector
from backend.models import GenerateRequest, GenerateResponse, HistoryDetailResponseItem, HistoryResponseItem
from backend.repair import ValidatingRepairLoop
from backend.store import get_history_item, list_history, save_generation
from backend.validators import EducationalValidator


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Generador Educativo de Codigo Python",
    version="0.3.0",
    description="API MVP robusta con repair loop, validacion educativa y metricas.",
    lifespan=lifespan,
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


@app.post("/generate", response_model=GenerateResponse, response_model_exclude_none=True)
def generate_example(payload: GenerateRequest, debug: bool = False) -> GenerateResponse:
    start = perf_counter()
    try:
        response = _repair_loop.run(payload, debug=debug)
        duration_ms = (perf_counter() - start) * 1000
        response.duration_ms = duration_ms
        status = "success" if response.tests_passed else "failed"
        _metrics.record(
            request=payload,
            response=response,
            duration_ms=duration_ms,
            status=status,
        )
        save_generation(payload, response)
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


@app.get("/history", response_model=list[HistoryResponseItem])
def get_history(limit: int = 20) -> list[HistoryResponseItem]:
    return [HistoryResponseItem(**asdict(item)) for item in list_history(limit=limit)]


@app.get("/history/{item_id}", response_model=HistoryDetailResponseItem)
def get_history_detail(item_id: int) -> HistoryDetailResponseItem:
    item = get_history_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"History item {item_id} no encontrado.")
    return HistoryDetailResponseItem(**asdict(item))

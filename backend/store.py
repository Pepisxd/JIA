from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import desc, select

from backend.db import SessionLocal, init_db
from backend.models import GenerateRequest, GenerateResponse
from backend.orm_models import GenerationHistory


@dataclass(slots=True)
class HistoryItem:
    id: int
    created_at: str
    tema: str
    nivel: str
    contexto: str
    tipo: str
    use_rag: bool
    tests_passed: bool
    educational_passed: bool
    attempts: int
    duration_ms: float
    error: str | None


@dataclass(slots=True)
class HistoryDetailItem(HistoryItem):
    objetivo: str
    codigo: str
    explicacion: list
    ejercicio: str
    output: str
    validation_errors: list
    dataset_name: str
    dataset_data: list
    dataset_load_code: str


def save_generation(request: GenerateRequest, response: GenerateResponse) -> int:
    init_db()
    with SessionLocal() as session:
        row = GenerationHistory(
            tema=request.tema,
            nivel=request.nivel,
            contexto=request.contexto,
            tipo=request.tipo,
            use_rag=request.use_rag,
            objetivo=response.objetivo,
            codigo=response.codigo,
            explicacion=response.explicacion,
            ejercicio=response.ejercicio,
            output=response.output,
            tests_passed=response.tests_passed,
            educational_passed=response.educational_passed,
            validation_errors=response.validation_errors,
            attempts=response.attempts,
            duration_ms=response.duration_ms or 0.0,
            error=response.error,
            dataset_name=response.dataset.nombre,
            dataset_data=response.dataset.data,
            dataset_load_code=response.dataset.codigo_carga,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id


def list_history(limit: int = 20) -> list[HistoryItem]:
    init_db()
    safe_limit = max(1, min(limit, 200))
    with SessionLocal() as session:
        rows = session.execute(
            select(GenerationHistory).order_by(desc(GenerationHistory.id)).limit(safe_limit)
        ).scalars()
        return [
            HistoryItem(
                id=row.id,
                created_at=row.created_at.isoformat(),
                tema=row.tema,
                nivel=row.nivel,
                contexto=row.contexto,
                tipo=row.tipo,
                use_rag=row.use_rag,
                tests_passed=row.tests_passed,
                educational_passed=row.educational_passed,
                attempts=row.attempts,
                duration_ms=row.duration_ms,
                error=row.error,
            )
            for row in rows
        ]


def get_history_item(item_id: int) -> HistoryDetailItem | None:
    init_db()
    with SessionLocal() as session:
        row = session.get(GenerationHistory, item_id)
        if not row:
            return None
        return HistoryDetailItem(
            id=row.id,
            created_at=row.created_at.isoformat(),
            tema=row.tema,
            nivel=row.nivel,
            contexto=row.contexto,
            tipo=row.tipo,
            use_rag=row.use_rag,
            tests_passed=row.tests_passed,
            educational_passed=row.educational_passed,
            attempts=row.attempts,
            duration_ms=row.duration_ms,
            error=row.error,
            objetivo=row.objetivo,
            codigo=row.codigo,
            explicacion=row.explicacion or [],
            ejercicio=row.ejercicio,
            output=row.output,
            validation_errors=row.validation_errors or [],
            dataset_name=row.dataset_name,
            dataset_data=row.dataset_data or [],
            dataset_load_code=row.dataset_load_code,
        )

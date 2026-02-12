from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TemaLiteral = Literal[
    "pandas_groupby",
    "pandas_filtrado",
    "pandas_lectura",
    "matplotlib_basico",
    "numpy_basico",
    "eda_basico",
]
NivelLiteral = Literal["principiante", "intermedio", "avanzado"]
ContextoLiteral = Literal["deportes", "finanzas", "videojuegos", "ciencia"]
TipoLiteral = Literal["tutorial", "desafio", "mini-proyecto"]


class GenerateRequest(BaseModel):
    tema: TemaLiteral
    nivel: NivelLiteral
    contexto: ContextoLiteral
    tipo: TipoLiteral
    use_rag: bool = False


class DatasetInfo(BaseModel):
    nombre: str = Field(min_length=1)
    data: list[dict[str, Any]] = Field(default_factory=list)
    codigo_carga: str = Field(min_length=1)


class ParsedLLMResponse(BaseModel):
    objetivo: str = Field(min_length=1)
    dataset: DatasetInfo
    codigo: str = Field(min_length=1)
    explicacion: list[str] = Field(default_factory=list)
    ejercicio: str = Field(min_length=1)


class GenerateResponse(BaseModel):
    objetivo: str
    dataset: DatasetInfo
    codigo: str
    explicacion: list[str]
    ejercicio: str
    tests_passed: bool
    educational_passed: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    attempts: int = Field(ge=1)
    output: str
    duration_ms: float | None = None
    error: str | None = None

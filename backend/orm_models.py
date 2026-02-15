from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.db import Base


class GenerationHistory(Base):
    __tablename__ = "generation_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    tema: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    nivel: Mapped[str] = mapped_column(String(32), nullable=False)
    contexto: Mapped[str] = mapped_column(String(32), nullable=False)
    tipo: Mapped[str] = mapped_column(String(32), nullable=False)
    use_rag: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    objetivo: Mapped[str] = mapped_column(Text, nullable=False)
    codigo: Mapped[str] = mapped_column(Text, nullable=False)
    explicacion: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    ejercicio: Mapped[str] = mapped_column(Text, nullable=False)
    output: Mapped[str] = mapped_column(Text, default="", nullable=False)

    tests_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    educational_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    validation_errors: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    dataset_name: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset_data: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    dataset_load_code: Mapped[str] = mapped_column(Text, nullable=False)

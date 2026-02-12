from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.models import TemaLiteral


@dataclass(slots=True)
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)


class EducationalValidator:
    def __init__(self, min_lines: int = 4, max_lines: int = 120) -> None:
        self.min_lines = min_lines
        self.max_lines = max_lines
        self._agg_patterns = [".sum(", ".mean(", ".count(", ".agg(", ".min(", ".max("]

    def validate(self, tema: TemaLiteral, code: str, explanation: list[str] | None = None) -> ValidationResult:
        errors: list[str] = []
        self._validate_generic(code, explanation or [], errors)
        self._validate_topic_specific(tema, code, errors)
        return ValidationResult(passed=not errors, errors=errors)

    def _validate_generic(self, code: str, explanation: list[str], errors: list[str]) -> None:
        lines = [line for line in code.splitlines() if line.strip()]
        if len(lines) < self.min_lines:
            errors.append(f"Codigo demasiado corto: {len(lines)} lineas (min {self.min_lines}).")
        if len(lines) > self.max_lines:
            errors.append(f"Codigo demasiado largo: {len(lines)} lineas (max {self.max_lines}).")
        if not any(line.strip().startswith("#") for line in code.splitlines()):
            errors.append("Faltan comentarios educativos en el codigo.")
        if len(explanation) < 2:
            errors.append("La explicacion debe incluir al menos 2 pasos.")

        bad_names = re.findall(r"(?m)^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=", code)
        weak = {"x", "y", "z", "tmp", "var", "foo", "bar"}
        if any(name.lower() in weak for name in bad_names):
            errors.append("Nombres de variables poco descriptivos detectados.")

    def _validate_topic_specific(self, tema: TemaLiteral, code: str, errors: list[str]) -> None:
        code_low = code.lower()

        if tema == "pandas_groupby":
            if ".groupby(" not in code_low:
                errors.append("Tema pandas_groupby requiere usar .groupby().")
            if not any(pat in code_low for pat in self._agg_patterns):
                errors.append("Tema pandas_groupby requiere una agregacion (.sum/.mean/.agg/etc).")
            return

        if tema == "pandas_filtrado":
            has_query = ".query(" in code_low
            has_boolean_indexing = bool(re.search(r"df\s*\[\s*df\s*\[", code_low))
            if not (has_query or has_boolean_indexing):
                errors.append("Tema pandas_filtrado requiere boolean indexing o .query().")
            return

        if tema == "pandas_lectura":
            has_reader = ("pd.read_csv(" in code_low) or ("pd.read_excel(" in code_low)
            if not has_reader:
                errors.append("Tema pandas_lectura requiere pd.read_csv() o pd.read_excel().")
            if ".head(" not in code_low:
                errors.append("Tema pandas_lectura requiere mostrar preview con .head().")

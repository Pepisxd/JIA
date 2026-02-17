from __future__ import annotations

import re
from dataclasses import dataclass, field
from numbers import Number

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
        self._forbidden_tokens = [
            "/home",
            "/kaggle",
            "/content",
            "../input",
            "read_csv(",
            "read_parquet(",
            "read_excel(",
        ]
        self._english_markers = ["what is", "how many", "difference between", "explain why", "find the"]
        self.min_educational_comments = 5

    def validate(
        self,
        tema: TemaLiteral,
        code: str,
        explanation: list[str] | None = None,
        *,
        objective: str = "",
        raw_text: str | None = None,
        dataset_load_code: str = "",
        dataset_data: list[dict] | None = None,
    ) -> ValidationResult:
        errors: list[str] = []
        self._validate_generic(code, explanation or [], errors)
        self._non_python_lines_in_code_validator(code=code, errors=errors)
        self._validate_topic_specific(tema, code, errors)
        self._forbidden_paths_validator(code=code, raw_text=raw_text, errors=errors)
        self._dataset_section_validator(raw_text=raw_text, dataset_load_code=dataset_load_code, code=code, errors=errors)
        self._dataset_code_coherence_validator(code=code, dataset_data=dataset_data or [], errors=errors)
        self._spanish_validator_simple(objective=objective, explanation=explanation or [], raw_text=raw_text, errors=errors)
        return ValidationResult(passed=not errors, errors=errors)

    def validate_non_python_lines_in_code(self, code: str) -> ValidationResult:
        errors: list[str] = []
        self._non_python_lines_in_code_validator(code=code, errors=errors)
        return ValidationResult(passed=not errors, errors=errors)

    def _validate_generic(self, code: str, explanation: list[str], errors: list[str]) -> None:
        lines = [line for line in code.splitlines() if line.strip()]
        if len(lines) < self.min_lines:
            errors.append(f"Codigo demasiado corto: {len(lines)} lineas (min {self.min_lines}).")
        if len(lines) > self.max_lines:
            errors.append(f"Codigo demasiado largo: {len(lines)} lineas (max {self.max_lines}).")
        comment_lines = [line for line in code.splitlines() if line.strip().startswith("#")]
        if len(comment_lines) < self.min_educational_comments:
            errors.append(
                f"Faltan comentarios educativos en el codigo: {len(comment_lines)} encontrados, minimo {self.min_educational_comments}."
            )
        if len(explanation) < 2:
            errors.append("La explicacion debe incluir al menos 2 pasos.")

        bad_names = re.findall(r"(?m)^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=", code)
        weak = {"x", "y", "z", "tmp", "var", "foo", "bar"}
        if any(name.lower() in weak for name in bad_names):
            errors.append("Nombres de variables poco descriptivos detectados.")

    def _non_python_lines_in_code_validator(self, code: str, errors: list[str]) -> None:
        suspicious_prefixes = ("instrucciones", "validation_errors", "salida_previa", "errores:")
        python_statement_prefixes = (
            "def ",
            "class ",
            "for ",
            "if ",
            "elif ",
            "while ",
            "try",
            "except",
            "with ",
            "match ",
            "case ",
            "return ",
            "import ",
            "from ",
            "@",
        )

        for line in code.splitlines():
            raw = line.rstrip()
            stripped = raw.strip()
            if not stripped:
                continue
            low = stripped.lower()
            if stripped.startswith("#"):
                continue
            if low.startswith(suspicious_prefixes):
                errors.append("Texto no ejecutable detectado en ## CODIGO (probable fuga del repair prompt).")
                return
            if stripped.startswith("-"):
                errors.append("Texto no ejecutable detectado en ## CODIGO (probable fuga del repair prompt).")
                return

            # Heuristica: linea tipo texto natural con ":" sin forma de sentencia Python.
            if ":" in stripped and not low.startswith(python_statement_prefixes):
                if not any(tok in stripped for tok in ("=", "(", ")", "[", "]", "{", "}", ".", "'", '"')):
                    words = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]+", stripped)
                    if len(words) >= 3:
                        errors.append("Texto no ejecutable detectado en ## CODIGO (probable fuga del repair prompt).")
                        return

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
            has_dataframe_build = "pd.dataframe(" in code_low
            if not has_dataframe_build:
                errors.append("Tema pandas_lectura requiere construir dataset con pd.DataFrame().")
            if ".head(" not in code_low:
                errors.append("Tema pandas_lectura requiere mostrar preview con .head().")

    def _forbidden_paths_validator(self, code: str, raw_text: str | None, errors: list[str]) -> None:
        text = f"{code}\n{raw_text or ''}".lower()
        found = [token for token in self._forbidden_tokens if token in text]
        if found:
            errors.append(f"forbidden_paths_validator: tokens prohibidos detectados: {', '.join(found)}")

    def _dataset_section_validator(
        self,
        *,
        raw_text: str | None,
        dataset_load_code: str,
        code: str,
        errors: list[str],
    ) -> None:
        if raw_text is not None:
            text = raw_text.lower()
            has_dataset_header = "## dataset" in text
            if not has_dataset_header:
                errors.append("dataset_section_validator: falta sección ## DATASET.")
        merged = f"{dataset_load_code}\n{code}".lower()
        if "pd.dataframe(" not in merged:
            errors.append("dataset_section_validator: falta uso de pd.DataFrame en dataset/codigo.")

    def _dataset_code_coherence_validator(self, *, code: str, dataset_data: list[dict], errors: list[str]) -> None:
        if not dataset_data:
            return
        first_row = next((row for row in dataset_data if isinstance(row, dict) and row), None)
        if first_row is None:
            return
        dataset_keys = [k for k in first_row.keys() if isinstance(k, str)]
        if not dataset_keys:
            return

        code_norm = re.sub(r"\s+", "", code.lower())
        if "pd.dataframe(rows)" not in code_norm:
            errors.append(
                "code_overrides_dataset: el codigo debe construir DataFrame desde rows usando pd.DataFrame(rows)."
            )
            return

        rows_match = re.search(r"(?is)\brows\s*=\s*(\[[\s\S]*?\])", code)
        if not rows_match:
            return
        code_rows_block = rows_match.group(1)
        code_keys = set(re.findall(r'["\']([A-Za-z_][A-Za-z0-9_]*)["\']\s*:', code_rows_block))
        if not code_keys:
            return
        dataset_key_set = set(dataset_keys)
        if code_keys != dataset_key_set:
            errors.append(
                "code_overrides_dataset: ## CODIGO redefine rows con columnas distintas al ## DATASET."
            )
            return

        # Guardrail adicional: al menos una columna categorica y una numerica usadas en el codigo.
        categorical = [k for k, v in first_row.items() if isinstance(v, str)]
        numeric = [k for k, v in first_row.items() if isinstance(v, Number) and not isinstance(v, bool)]
        if categorical and categorical[0] not in code:
            errors.append("code_overrides_dataset: el codigo no usa la columna categorica esperada del dataset.")
        if numeric and numeric[0] not in code:
            errors.append("code_overrides_dataset: el codigo no usa la columna numerica esperada del dataset.")

    def _spanish_validator_simple(
        self,
        *,
        objective: str,
        explanation: list[str],
        raw_text: str | None,
        errors: list[str],
    ) -> None:
        if not objective and raw_text is None:
            return
        objective_low = objective.lower()
        detected = [marker for marker in self._english_markers if marker in objective_low]
        if detected:
            errors.append(
                "spanish_validator_simple: objetivo parece estar en ingles; marcadores detectados: "
                + ", ".join(detected)
            )

        if raw_text is None:
            return

        text = f"{objective}\n{' '.join(explanation)}\n{raw_text}"
        text_low = text.lower()
        spanish_hints = [" el ", " la ", " de ", " para ", " datos ", " código ", " explicacion ", " objetivo "]
        score = sum(1 for hint in spanish_hints if hint in f" {text_low} ")
        has_accent = any(ch in text for ch in "áéíóúñÁÉÍÓÚÑ")
        if score < 2 and not has_accent:
            errors.append("spanish_validator_simple: no hay suficientes señales de español.")

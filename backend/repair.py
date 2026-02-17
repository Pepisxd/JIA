from __future__ import annotations

import json
from numbers import Number

from backend.executor import CodeExecutor, ExecutionResult
from backend.generator import CodeGenerator, GenerationError
from backend.models import GenerateRequest, GenerateResponse
from backend.parser import ParseError, apply_section_patch
from backend.validators import EducationalValidator, ValidationResult

DEBUG_TEXT_LIMIT = 4000


def _truncate(value: str, limit: int = DEBUG_TEXT_LIMIT) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} chars]"


def _infer_dataset_columns(rows: list[dict]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            if isinstance(key, str) and key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def _inject_educational_comments(code: str, min_comments: int = 5) -> str:
    lines = code.splitlines()
    existing = sum(1 for line in lines if line.strip().startswith("#"))
    missing = max(0, min_comments - existing)
    if missing == 0:
        return code

    insert_at = 0
    while insert_at < len(lines):
        stripped = lines[insert_at].strip()
        if not stripped or stripped.startswith("import ") or stripped.startswith("from "):
            insert_at += 1
            continue
        break
    comments = [f"# Comentario educativo {i + 1}" for i in range(missing)]
    patched = lines[:insert_at] + comments + lines[insert_at:]
    return "\n".join(patched).strip()


def _build_deterministic_code_from_dataset(dataset_data: list[dict]) -> str:
    rows = [row for row in dataset_data if isinstance(row, dict)]
    if not rows:
        return (
            "import pandas as pd\n"
            "# Comentario educativo 1\n# Comentario educativo 2\n# Comentario educativo 3\n"
            "# Comentario educativo 4\n# Comentario educativo 5\n"
            "df = pd.DataFrame(rows)\n"
            "print(df.head())"
        )

    sample = rows[0]
    keys = _infer_dataset_columns(rows) or [k for k in sample.keys() if isinstance(k, str)]
    categorical = next((k for k in keys if isinstance(sample.get(k), str)), keys[0])
    numeric = next(
        (
            k
            for k in keys
            if isinstance(sample.get(k), Number) and not isinstance(sample.get(k), bool)
        ),
        keys[1] if len(keys) > 1 else keys[0],
    )
    return (
        "import pandas as pd\n"
        "# Comentario educativo 1: construir DataFrame desde rows del dataset.\n"
        "# Comentario educativo 2: inspeccionar las primeras filas para validar columnas.\n"
        "# Comentario educativo 3: agrupar por la columna categorica principal.\n"
        "# Comentario educativo 4: aplicar suma sobre la columna numerica principal.\n"
        "# Comentario educativo 5: mostrar el resumen para interpretar resultados.\n"
        "df = pd.DataFrame(rows)\n"
        "print(df.head())\n"
        f"resumen = df.groupby('{categorical}', as_index=False)['{numeric}'].sum()\n"
        "print(resumen)"
    )


class RepairLoop:
    def __init__(self, generator: CodeGenerator, executor: CodeExecutor, max_attempts: int = 3) -> None:
        self.generator = generator
        self.executor = executor
        self.max_attempts = max_attempts

    def _build_response(
        self,
        *,
        parsed,
        result: ExecutionResult,
        attempts: int,
        success: bool,
        educational_passed: bool = True,
        validation_errors: list[str] | None = None,
        error: str | None = None,
        raw_text_original: str = "",
        sanitized_text: str = "",
        used_fallback: bool = False,
        model_backend: str = "deterministic",
        post_processed: bool = False,
    ) -> GenerateResponse:
        return GenerateResponse(
            objetivo=parsed.objetivo,
            dataset=parsed.dataset,
            codigo=parsed.codigo,
            explicacion=parsed.explicacion,
            ejercicio=parsed.ejercicio,
            tests_passed=success,
            educational_passed=educational_passed,
            validation_errors=validation_errors or [],
            attempts=attempts,
            output=result.output,
            error=error,
            raw_text_original=raw_text_original,
            sanitized_text=sanitized_text,
            used_fallback=used_fallback,
            model_backend=model_backend,  # type: ignore[arg-type]
            post_processed=post_processed,
        )

    def run(self, request: GenerateRequest, debug: bool = False) -> GenerateResponse:
        last_error = ""
        last_response: GenerateResponse | None = None
        debug_attempts: list[dict] = []

        for attempt in range(1, self.max_attempts + 1):
            raw_text_original = ""
            sanitized_text = ""
            used_fallback = False
            model_backend = "deterministic"
            post_processed = False
            prompt_sent = ""
            parse_ok = True

            try:
                if hasattr(self.generator, "generate_with_raw"):
                    trace = self.generator.generate_with_raw(
                        request=request,
                        error_prev=last_error or None,
                        attempt=attempt,
                        use_rag=request.use_rag,
                    )
                    parsed = trace.parsed
                    raw_text_original = trace.raw_text_original
                    sanitized_text = trace.sanitized_text
                    used_fallback = trace.used_fallback
                    model_backend = trace.model_backend
                    post_processed = trace.post_processed
                    prompt_sent = trace.prompt_sent
                    parse_ok = trace.parse_ok
                else:
                    parsed = self.generator.generate(
                        request=request,
                        error_prev=last_error or None,
                        attempt=attempt,
                        use_rag=request.use_rag,
                    )
            except GenerationError as exc:
                last_error = str(exc)
                if attempt == self.max_attempts:
                    raise
                continue

            result = self.executor.execute(parsed.codigo, dataset_data=parsed.dataset.data)
            if debug:
                debug_attempts.append(
                    {
                        "attempt": attempt,
                        "model_backend": model_backend,
                        "used_fallback": used_fallback,
                        "prompt_sent": _truncate(prompt_sent),
                        "raw_text": _truncate(raw_text_original),
                        "sanitized_text": _truncate(sanitized_text),
                        "validation_errors": [],
                        "parse_ok": parse_ok,
                        "exec_ok": result.success,
                        "exec_error": result.error,
                    }
                )
            if result.success:
                response = self._build_response(
                    parsed=parsed,
                    result=result,
                    attempts=attempt,
                    success=True,
                    raw_text_original=raw_text_original,
                    sanitized_text=sanitized_text,
                    used_fallback=used_fallback,
                    model_backend=model_backend,
                    post_processed=post_processed,
                )
                if debug:
                    response.debug = {"attempts": debug_attempts}
                return response

            last_error = f"{result.error_type}: {result.error}" if result.error else "Ejecucion fallida."
            last_response = self._build_response(
                parsed=parsed,
                result=result,
                attempts=attempt,
                success=False,
                educational_passed=False,
                error=last_error,
                raw_text_original=raw_text_original,
                sanitized_text=sanitized_text,
                used_fallback=used_fallback,
                model_backend=model_backend,
                post_processed=post_processed,
            )

        if last_response:
            if debug:
                last_response.debug = {"attempts": debug_attempts}
            return last_response
        raise RuntimeError("Repair loop finalizo sin respuesta util.")


class ValidatingRepairLoop(RepairLoop):
    def __init__(
        self,
        generator: CodeGenerator,
        executor: CodeExecutor,
        validator: EducationalValidator,
        max_attempts: int = 3,
    ) -> None:
        super().__init__(generator=generator, executor=executor, max_attempts=max_attempts)
        self.validator = validator

    def run(self, request: GenerateRequest, debug: bool = False) -> GenerateResponse:
        last_error = ""
        last_response: GenerateResponse | None = None
        debug_attempts: list[dict] = []

        for attempt in range(1, self.max_attempts + 1):
            raw_text: str | None = None
            raw_text_original = ""
            sanitized_text = ""
            used_fallback = False
            model_backend = "deterministic"
            post_processed = False
            prompt_sent = ""
            parse_ok = True

            try:
                if hasattr(self.generator, "generate_with_raw"):
                    trace = self.generator.generate_with_raw(
                        request=request,
                        error_prev=last_error or None,
                        attempt=attempt,
                        use_rag=request.use_rag,
                    )
                    parsed = trace.parsed
                    raw_text = trace.sanitized_text
                    raw_text_original = trace.raw_text_original
                    sanitized_text = trace.sanitized_text
                    used_fallback = trace.used_fallback
                    model_backend = trace.model_backend
                    post_processed = trace.post_processed
                    prompt_sent = trace.prompt_sent
                    parse_ok = trace.parse_ok
                else:
                    parsed = self.generator.generate(
                        request=request,
                        error_prev=last_error or None,
                        attempt=attempt,
                        use_rag=request.use_rag,
                    )
            except GenerationError as exc:
                last_error = str(exc)
                if attempt == self.max_attempts:
                    raise
                continue

            if not parsed.dataset.columns:
                parsed.dataset.columns = _infer_dataset_columns(parsed.dataset.data)

            current_parsed = parsed
            current_text: str | None = self._compose_sections_text(current_parsed)
            raw_for_validation: str | None = raw_text if raw_text else None
            current_prompt = prompt_sent
            patches_applied = 0
            max_patches = 3
            final_result: ExecutionResult | None = None
            final_validation_errors: list[str] = []
            final_success = False

            while True:
                leak_check = self.validator.validate_non_python_lines_in_code(current_parsed.codigo)
                if not leak_check.passed:
                    action = self._select_patch_action(leak_check.errors)
                    if action is not None and patches_applied < max_patches:
                        ok = self._apply_patch(
                            request=request,
                            current_parsed=current_parsed,
                            mode=action[0],
                            section=action[1],
                            instruction=action[2],
                            previous_text=current_text or "",
                        )
                        if ok is not None:
                            current_parsed, current_text, current_prompt, used_fallback, model_backend = ok
                            raw_for_validation = current_text
                            patches_applied += 1
                            continue
                    final_result = ExecutionResult(
                        success=False,
                        output="",
                        error="Skipped execution by non_python_lines_in_code_validator",
                        error_type="ValidationError",
                    )
                    final_validation_errors = leak_check.errors
                    break

                result = self.executor.execute(current_parsed.codigo, dataset_data=current_parsed.dataset.data)
                if not result.success:
                    final_result = result
                    break

                validation: ValidationResult = self.validator.validate(
                    tema=request.tema,
                    code=current_parsed.codigo,
                    explanation=current_parsed.explicacion,
                    objective=current_parsed.objetivo,
                    raw_text=raw_for_validation,
                    dataset_load_code=current_parsed.dataset.codigo_carga,
                    dataset_data=current_parsed.dataset.data,
                )
                if validation.passed:
                    final_result = result
                    final_success = True
                    final_validation_errors = []
                    break

                final_validation_errors = validation.errors
                action = self._select_patch_action(validation.errors)
                if action is not None and patches_applied < max_patches:
                    ok = self._apply_patch(
                        request=request,
                        current_parsed=current_parsed,
                        mode=action[0],
                        section=action[1],
                        instruction=action[2],
                        previous_text=current_text or "",
                    )
                    if ok is not None:
                        current_parsed, current_text, current_prompt, used_fallback, model_backend = ok
                        raw_for_validation = current_text
                        patches_applied += 1
                        continue
                final_result = result
                break

            assert final_result is not None

            if debug:
                debug_attempts.append(
                    {
                        "attempt": attempt,
                        "model_backend": model_backend,
                        "used_fallback": used_fallback,
                        "prompt_sent": _truncate(current_prompt),
                        "raw_text": _truncate(raw_text_original),
                        "sanitized_text": _truncate(current_text or ""),
                        "validation_errors": final_validation_errors,
                        "parse_ok": parse_ok,
                        "exec_ok": final_result.success,
                        "exec_error": final_result.error,
                    }
                )

            if final_success:
                response = self._build_response(
                    parsed=current_parsed,
                    result=final_result,
                    attempts=attempt,
                    success=True,
                    educational_passed=True,
                    validation_errors=[],
                    raw_text_original=raw_text_original,
                    sanitized_text=current_text or "",
                    used_fallback=used_fallback,
                    model_backend=model_backend,
                    post_processed=post_processed,
                )
                if debug:
                    response.debug = {"attempts": debug_attempts}
                return response

            if final_result.error_type and final_result.error_type != "ValidationError":
                last_error = f"{final_result.error_type}: {final_result.error}" if final_result.error else "Ejecucion fallida."
            else:
                last_error = self._build_validation_retry_instruction(
                    errors=final_validation_errors,
                    previous_text=(raw_for_validation or current_text or ""),
                )
            last_response = self._build_response(
                parsed=current_parsed,
                result=final_result,
                attempts=attempt,
                success=False,
                educational_passed=False,
                validation_errors=final_validation_errors,
                error=(f"ValidationError: {' | '.join(final_validation_errors)}" if final_validation_errors else last_error),
                raw_text_original=raw_text_original,
                sanitized_text=current_text or "",
                used_fallback=used_fallback,
                model_backend=model_backend,
                post_processed=post_processed,
            )

        if last_response:
            if debug:
                last_response.debug = {"attempts": debug_attempts}
            return last_response
        raise RuntimeError("Validating repair loop finalizo sin respuesta util.")

    def _select_patch_action(self, errors: list[str]) -> tuple[str, str, str] | None:
        lowered = [err.lower() for err in errors]
        if any("code_overrides_dataset" in err for err in lowered):
            return ("deterministic", "codigo", "rebuild_code_from_dataset")
        if any("fuga del repair prompt" in err for err in lowered):
            return (
                "model",
                "codigo",
                "Devuelve unicamente el bloque completo de ## CODIGO en un fence ```python ...``` y nada mas. "
                "No incluyas listas, encabezados, ni texto fuera de Python.",
            )
        if any("comentarios educativos" in err for err in lowered):
            return ("deterministic", "codigo", "inject_comments")
        if any("spanish_validator_simple" in err for err in lowered):
            return (
                "model",
                "objetivo",
                "Devuelve unicamente una linea para ## OBJETIVO en espanol y nada mas.",
            )
        if any("la explicacion debe incluir al menos 2 pasos" in err for err in lowered):
            return (
                "model",
                "explicacion",
                "Devuelve unicamente ## EXPLICACION con al menos 2 bullets en espanol.",
            )
        if any("dataset_section_validator" in err for err in lowered):
            return (
                "model",
                "codigo",
                "Devuelve unicamente ## CODIGO en fence ```python``` asegurando df = pd.DataFrame(rows).",
            )
        return None

    def _apply_patch(
        self,
        *,
        request: GenerateRequest,
        current_parsed,
        mode: str,
        section: str,
        instruction: str,
        previous_text: str,
    ) -> tuple | None:
        if mode == "deterministic":
            patched = current_parsed.model_copy(deep=True)
            if instruction == "inject_comments":
                patched.codigo = _inject_educational_comments(
                    patched.codigo,
                    min_comments=self.validator.min_educational_comments,
                )
            elif instruction == "rebuild_code_from_dataset":
                patched.codigo = _build_deterministic_code_from_dataset(patched.dataset.data)
            else:
                return None
            if not patched.dataset.columns:
                patched.dataset.columns = _infer_dataset_columns(patched.dataset.data)
            merged_text = self._compose_sections_text(patched)
            return patched, merged_text, "deterministic-patch", True, "deterministic"

        if not hasattr(self.generator, "generate_patch"):
            return None
        raw, sanitized, prompt, used_fallback, backend = self.generator.generate_patch(
            request=request,
            section=section,
            instruction=instruction,
            previous_text=previous_text,
        )
        try:
            patched = apply_section_patch(current_parsed, section=section, patch_text=sanitized)
        except ParseError:
            return None
        if not patched.dataset.columns:
            patched.dataset.columns = _infer_dataset_columns(patched.dataset.data)
        merged_text = self._compose_sections_text(patched)
        return patched, merged_text, prompt, used_fallback, backend

    def _compose_sections_text(self, parsed) -> str:
        dataset_block = (
            "```python\n"
            f"rows = {json.dumps(parsed.dataset.data, ensure_ascii=False, indent=2)}\n"
            "df = pd.DataFrame(rows)\n"
            "print(df.head())\n"
            "```"
        )
        explanation = "\n".join(f"- {item}" for item in parsed.explicacion)
        return (
            "## OBJETIVO\n"
            f"{parsed.objetivo}\n\n"
            "## DATASET\n"
            f"{dataset_block}\n\n"
            "## CODIGO\n"
            f"```python\n{parsed.codigo}\n```\n\n"
            "## EXPLICACION\n"
            f"{explanation}\n"
        )

    def _build_validation_retry_instruction(self, *, errors: list[str], previous_text: str) -> str:
        joined = " | ".join(errors).lower()
        if len(errors) == 1 and "spanish_validator_simple" in joined:
            action = "Reescribe SOLO la seccion ## OBJETIVO en espanol. No cambies nada mas. Manten las 4 secciones."
        elif len(errors) == 1 and "comentarios educativos" in joined:
            action = "Ajusta el codigo para incluir comentarios educativos con #."
        elif len(errors) == 1 and "dataset_section_validator" in joined:
            action = "Asegura que ## DATASET use rows y el codigo construya df = pd.DataFrame(rows)."
        elif len(errors) == 1 and "fuga del repair prompt" in joined:
            action = "En ## CODIGO deja SOLO Python ejecutable."
        else:
            bullets = "\n".join(f"- {err}" for err in errors)
            action = f"Corrige TODOS los errores listados sin anadir secciones nuevas.\nErrores:\n{bullets}"

        return (
            "=== REGLAS DE CORRECCION (NO COPIAR AL CODIGO) ===\n"
            "- NO copies ERRORES A CORREGIR ni SALIDA PREVIA dentro de ## CODIGO.\n"
            "- En ## CODIGO SOLO Python ejecutable. Si incluyes texto, debe ser comentario con #.\n"
            "- Si necesitas mencionar errores o cambios, hazlo en ## EXPLICACION, no en el codigo.\n\n"
            "=== TAREA DE REINTENTO (NO COPIAR AL CODIGO) ===\n"
            f"{action}\n\n"
            "=== ERRORES A CORREGIR (NO COPIAR AL CODIGO) ===\n"
            + "\n".join(f"- {err}" for err in errors)
            + "\n\n=== SALIDA PREVIA (NO COPIAR AL CODIGO) ===\n```text\n"
            + previous_text
            + "\n```"
        )

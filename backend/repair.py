from __future__ import annotations

from backend.executor import CodeExecutor, ExecutionResult
from backend.generator import CodeGenerator, GenerationError
from backend.models import GenerateRequest, GenerateResponse
from backend.validators import EducationalValidator, ValidationResult

DEBUG_TEXT_LIMIT = 4000


def _truncate(value: str, limit: int = DEBUG_TEXT_LIMIT) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} chars]"


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

            # Detect prompt leakage in code before execution.
            leak_check = self.validator.validate_non_python_lines_in_code(parsed.codigo)
            if not leak_check.passed:
                if debug:
                    debug_attempts.append(
                        {
                            "attempt": attempt,
                            "model_backend": model_backend,
                            "used_fallback": used_fallback,
                            "prompt_sent": _truncate(prompt_sent),
                            "raw_text": _truncate(raw_text_original),
                            "sanitized_text": _truncate(sanitized_text),
                            "validation_errors": leak_check.errors,
                            "parse_ok": parse_ok,
                            "exec_ok": False,
                            "exec_error": "Skipped execution by non_python_lines_in_code_validator",
                        }
                    )
                last_error = self._build_validation_retry_instruction(errors=leak_check.errors, previous_text=raw_text or "")
                last_response = self._build_response(
                    parsed=parsed,
                    result=ExecutionResult(
                        success=False,
                        output="",
                        error="Skipped execution by non_python_lines_in_code_validator",
                        error_type="ValidationError",
                    ),
                    attempts=attempt,
                    success=False,
                    educational_passed=False,
                    validation_errors=leak_check.errors,
                    error=f"ValidationError: {' | '.join(leak_check.errors)}",
                    raw_text_original=raw_text_original,
                    sanitized_text=sanitized_text,
                    used_fallback=used_fallback,
                    model_backend=model_backend,
                    post_processed=post_processed,
                )
                continue

            result = self.executor.execute(parsed.codigo, dataset_data=parsed.dataset.data)
            if not result.success:
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
                            "exec_ok": False,
                            "exec_error": result.error,
                        }
                    )
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
                continue

            validation: ValidationResult = self.validator.validate(
                tema=request.tema,
                code=parsed.codigo,
                explanation=parsed.explicacion,
                objective=parsed.objetivo,
                raw_text=raw_text,
                dataset_load_code=parsed.dataset.codigo_carga,
            )
            if debug:
                debug_attempts.append(
                    {
                        "attempt": attempt,
                        "model_backend": model_backend,
                        "used_fallback": used_fallback,
                        "prompt_sent": _truncate(prompt_sent),
                        "raw_text": _truncate(raw_text_original),
                        "sanitized_text": _truncate(sanitized_text),
                        "validation_errors": validation.errors,
                        "parse_ok": parse_ok,
                        "exec_ok": True,
                        "exec_error": None,
                    }
                )
            if validation.passed:
                response = self._build_response(
                    parsed=parsed,
                    result=result,
                    attempts=attempt,
                    success=True,
                    educational_passed=True,
                    validation_errors=[],
                    raw_text_original=raw_text_original,
                    sanitized_text=sanitized_text,
                    used_fallback=used_fallback,
                    model_backend=model_backend,
                    post_processed=post_processed,
                )
                if debug:
                    response.debug = {"attempts": debug_attempts}
                return response

            last_error = self._build_validation_retry_instruction(errors=validation.errors, previous_text=raw_text or "")
            last_response = self._build_response(
                parsed=parsed,
                result=result,
                attempts=attempt,
                success=False,
                educational_passed=False,
                validation_errors=validation.errors,
                error=f"ValidationError: {' | '.join(validation.errors)}",
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
        raise RuntimeError("Validating repair loop finalizo sin respuesta util.")

    def _build_validation_retry_instruction(self, *, errors: list[str], previous_text: str) -> str:
        joined = " | ".join(errors).lower()
        if len(errors) == 1 and "spanish_validator_simple" in joined:
            action = "Reescribe SOLO la sección ## OBJETIVO en español. No cambies nada más. Mantén las 4 secciones."
        elif len(errors) == 1 and "comentarios educativos" in joined:
            action = (
                "Agrega comentarios educativos con # en el bloque ## CODIGO (mínimo 5). "
                "No cambies la lógica. No uses read_csv ni rutas."
            )
        elif len(errors) == 1 and "dataset_section_validator" in joined:
            action = (
                "Asegura que ## DATASET incluya rows = [ {..}, ... ] (8-15 filas) "
                "y que el código construya df = pd.DataFrame(rows)."
            )
        elif len(errors) == 1 and "fuga del repair prompt" in joined:
            action = (
                "En ## CODIGO deja SOLO Python ejecutable. NO copies textos de contexto de reparación "
                "dentro del código."
            )
        else:
            bullets = "\n".join(f"- {err}" for err in errors)
            action = f"Corrige TODOS los errores listados sin añadir secciones nuevas.\nErrores:\n{bullets}"

        return (
            "=== REGLAS DE CORRECCION (NO COPIAR AL CODIGO) ===\n"
            "- NO copies ERRORES A CORREGIR ni SALIDA PREVIA dentro de ## CODIGO.\n"
            "- En ## CODIGO SOLO Python ejecutable. Si incluyes texto, debe ser comentario con #.\n"
            "- Si necesitas mencionar errores o cambios, hazlo en ## EXPLICACION, no en el código.\n\n"
            "=== TAREA DE REINTENTO (NO COPIAR AL CODIGO) ===\n"
            f"{action}\n\n"
            "=== ERRORES A CORREGIR (NO COPIAR AL CODIGO) ===\n"
            + "\n".join(f"- {err}" for err in errors)
            + "\n\n=== SALIDA PREVIA (NO COPIAR AL CODIGO) ===\n```text\n"
            + previous_text
            + "\n```"
        )

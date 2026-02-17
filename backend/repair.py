from __future__ import annotations

from backend.executor import CodeExecutor, ExecutionResult
from backend.generator import CodeGenerator, GenerationError
from backend.models import GenerateRequest, GenerateResponse
from backend.validators import EducationalValidator, ValidationResult


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
            model_backend=model_backend,
            post_processed=post_processed,
        )

    def run(self, request: GenerateRequest) -> GenerateResponse:
        last_error = ""
        last_response: GenerateResponse | None = None

        for attempt in range(1, self.max_attempts + 1):
            raw_text_original = ""
            sanitized_text = ""
            used_fallback = False
            model_backend = "deterministic"
            post_processed = False
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
            if result.success:
                return self._build_response(
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

    def run(self, request: GenerateRequest) -> GenerateResponse:
        last_error = ""
        last_response: GenerateResponse | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                raw_text: str | None = None
                raw_text_original = ""
                sanitized_text = ""
                used_fallback = False
                model_backend = "deterministic"
                post_processed = False
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
            if not result.success:
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
            if validation.passed:
                return self._build_response(
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

            last_error = self._build_validation_retry_instruction(
                errors=validation.errors,
                previous_text=raw_text or "",
            )
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
            return last_response
        raise RuntimeError("Validating repair loop finalizo sin respuesta util.")

    def _build_validation_retry_instruction(self, *, errors: list[str], previous_text: str) -> str:
        joined = " | ".join(errors).lower()
        if len(errors) == 1 and "spanish_validator_simple" in joined:
            action = (
                "Reescribe SOLO la sección ## OBJETIVO en español. "
                "No cambies nada más. Mantén las 4 secciones."
            )
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
        else:
            bullets = "\n".join(f"- {err}" for err in errors)
            action = (
                "Corrige TODOS los errores listados sin añadir secciones nuevas.\n"
                f"Errores:\n{bullets}"
            )
        return (
            "INSTRUCCIONES_DE_REPARACION:\n"
            f"{action}\n\n"
            "VALIDATION_ERRORS:\n"
            + "\n".join(f"- {err}" for err in errors)
            + "\n\nSALIDA_PREVIA:\n```text\n"
            + previous_text
            + "\n```"
        )

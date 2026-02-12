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
        )

    def run(self, request: GenerateRequest) -> GenerateResponse:
        last_error = ""
        last_response: GenerateResponse | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
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

            result = self.executor.execute(parsed.codigo)
            if result.success:
                return self._build_response(parsed=parsed, result=result, attempts=attempt, success=True)

            last_error = f"{result.error_type}: {result.error}" if result.error else "Ejecucion fallida."
            last_response = self._build_response(
                parsed=parsed,
                result=result,
                attempts=attempt,
                success=False,
                educational_passed=False,
                error=last_error,
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

            result = self.executor.execute(parsed.codigo)
            if not result.success:
                last_error = f"{result.error_type}: {result.error}" if result.error else "Ejecucion fallida."
                last_response = self._build_response(
                    parsed=parsed,
                    result=result,
                    attempts=attempt,
                    success=False,
                    educational_passed=False,
                    error=last_error,
                )
                continue

            validation: ValidationResult = self.validator.validate(
                tema=request.tema,
                code=parsed.codigo,
                explanation=parsed.explicacion,
            )
            if validation.passed:
                return self._build_response(
                    parsed=parsed,
                    result=result,
                    attempts=attempt,
                    success=True,
                    educational_passed=True,
                    validation_errors=[],
                )

            last_error = f"ValidationError: {' | '.join(validation.errors)}"
            last_response = self._build_response(
                parsed=parsed,
                result=result,
                attempts=attempt,
                success=False,
                educational_passed=False,
                validation_errors=validation.errors,
                error=last_error,
            )

        if last_response:
            return last_response
        raise RuntimeError("Validating repair loop finalizo sin respuesta util.")

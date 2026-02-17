from __future__ import annotations

import contextlib
import io
import traceback
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    output: str
    error: str | None = None
    error_type: str | None = None


class CodeExecutor:
    def __init__(self) -> None:
        self.blocked_tokens = [
            "subprocess",
            "os.system",
            "open(",
            "eval(",
            "__import__('os')",
            "__import__(\"os\")",
        ]

    def _check_safety(self, code: str) -> str | None:
        lower = code.lower()
        for token in self.blocked_tokens:
            if token in lower:
                return f"Token bloqueado detectado: {token}"
        return None

    def execute(self, code: str, dataset_data: list[dict[str, Any]] | None = None) -> ExecutionResult:
        safety_error = self._check_safety(code)
        if safety_error:
            return ExecutionResult(success=False, output="", error=safety_error, error_type="SafetyError")

        stdout = io.StringIO()
        globals_dict = {"__name__": "__generated__"}
        if dataset_data is not None:
            # Provide a conventional variable expected by generated snippets.
            globals_dict["rows"] = dataset_data

        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, globals_dict, globals_dict)
            return ExecutionResult(success=True, output=stdout.getvalue().strip())
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                success=False,
                output=stdout.getvalue().strip(),
                error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
                error_type=type(exc).__name__,
            )

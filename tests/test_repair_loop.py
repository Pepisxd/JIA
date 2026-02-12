from backend.executor import CodeExecutor
from backend.models import DatasetInfo, GenerateRequest, ParsedLLMResponse
from backend.repair import RepairLoop, ValidatingRepairLoop
from backend.validators import EducationalValidator


class StubGenerator:
    def __init__(self, responses: list[ParsedLLMResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(
        self,
        request: GenerateRequest,
        error_prev: str | None = None,
        attempt: int = 1,
        use_rag: bool = False,
    ) -> ParsedLLMResponse:
        _ = request, error_prev, attempt, use_rag
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def _request() -> GenerateRequest:
    return GenerateRequest(
        tema="pandas_groupby",
        nivel="principiante",
        contexto="deportes",
        tipo="tutorial",
    )


def _parsed(code: str) -> ParsedLLMResponse:
    return ParsedLLMResponse(
        objetivo="Objetivo",
        dataset=DatasetInfo(
            nombre="demo",
            data=[{"equipo": "A", "goles": 1}],
            codigo_carga="df = pd.DataFrame([...])",
        ),
        codigo=code,
        explicacion=["Paso 1", "Paso 2"],
        ejercicio="Ejercicio",
    )


def test_repair_loop_recovers_on_second_attempt() -> None:
    bad = _parsed("print(variable_no_definida)")
    good = _parsed("print('ok')")
    loop = RepairLoop(generator=StubGenerator([bad, good]), executor=CodeExecutor(), max_attempts=3)

    result = loop.run(_request())

    assert result.tests_passed is True
    assert result.attempts == 2
    assert result.output == "ok"


def test_repair_loop_returns_failure_after_max_attempts() -> None:
    bad = _parsed("print(variable_no_definida)")
    loop = RepairLoop(generator=StubGenerator([bad]), executor=CodeExecutor(), max_attempts=2)

    result = loop.run(_request())

    assert result.tests_passed is False
    assert result.attempts == 2
    assert result.error is not None


def test_validating_repair_loop_retries_until_educationally_valid() -> None:
    first = _parsed(
        """
import pandas as pd
df = pd.DataFrame([{"equipo":"A","goles":1}])
print(df)
"""
    )
    second = _parsed(
        """
import pandas as pd
# Comentario educativo
df = pd.DataFrame([{"equipo":"A","goles":1},{"equipo":"B","goles":2}])
resumen = df.groupby("equipo")["goles"].sum().reset_index()
print(resumen)
"""
    )
    loop = ValidatingRepairLoop(
        generator=StubGenerator([first, second]),
        executor=CodeExecutor(),
        validator=EducationalValidator(),
        max_attempts=3,
    )

    result = loop.run(_request())

    assert result.tests_passed is True
    assert result.educational_passed is True
    assert result.attempts == 2

from backend.executor import CodeExecutor
from backend.generator import GenerationTrace
from backend.models import DatasetInfo, GenerateRequest, ParsedLLMResponse
from backend.repair import RepairLoop, ValidatingRepairLoop
from backend.validators import EducationalValidator


def _request() -> GenerateRequest:
    return GenerateRequest(
        tema="pandas_groupby",
        nivel="principiante",
        contexto="deportes",
        tipo="tutorial",
    )


def _parsed(code: str) -> ParsedLLMResponse:
    return ParsedLLMResponse(
        objetivo="Objetivo de analisis de datos para deportes.",
        dataset=DatasetInfo(
            nombre="demo",
            data=[{"equipo": "A", "goles": 1}],
            codigo_carga="df = pd.DataFrame([...])",
        ),
        codigo=code,
        explicacion=["Paso 1", "Paso 2"],
        ejercicio="Ejercicio",
    )


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


def test_validating_repair_loop_injects_missing_comments_deterministically() -> None:
    first = _parsed(
        """
import pandas as pd
df = pd.DataFrame(rows)
resumen = df.groupby("equipo")["goles"].sum().reset_index()
print(resumen)
"""
    )
    loop = ValidatingRepairLoop(
        generator=StubGenerator([first]),
        executor=CodeExecutor(),
        validator=EducationalValidator(),
        max_attempts=2,
    )
    result = loop.run(_request())
    assert result.tests_passed is True
    assert result.educational_passed is True
    assert result.attempts == 1
    assert result.codigo.count("#") >= 5


class StubTraceGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate_with_raw(
        self,
        request: GenerateRequest,
        error_prev: str | None = None,
        attempt: int = 1,
        use_rag: bool = False,
    ) -> GenerationTrace:
        _ = request, attempt, use_rag
        self.calls += 1
        first = ParsedLLMResponse(
            objetivo="How many goals are there?",
            dataset=DatasetInfo(
                nombre="demo",
                data=[{"equipo": "A", "goles": 1}, {"equipo": "B", "goles": 2}],
                codigo_carga="df = pd.DataFrame(rows)",
            ),
            codigo=(
                "import pandas as pd\n"
                "# c1\n# c2\n# c3\n# c4\n# c5\n"
                "df = pd.DataFrame(rows)\n"
                "print(df.groupby('equipo')['goles'].sum())"
            ),
            explicacion=["Paso 1", "Paso 2"],
            ejercicio="Ejercicio",
        )

        second = ParsedLLMResponse(
            objetivo="Objetivo de analisis de datos para deportes.",
            dataset=first.dataset,
            codigo=first.codigo,
            explicacion=first.explicacion,
            ejercicio=first.ejercicio,
        )

        if self.calls == 1:
            parsed = first
            raw = (
                "## OBJETIVO\nHow many goals are there?\n\n"
                "## DATASET\nrows = [{'equipo':'A','goles':1}]\n\n"
                "## CODIGO\n# comentario\n# comentario\n# comentario\n# comentario\n# comentario\n"
                "df = pd.DataFrame(rows)\nprint(df)\n\n"
                "## EXPLICACION\nEste ejemplo usa datos para analisis de datos en espanol."
            )
        else:
            assert error_prev is not None
            assert "OBJETIVO" in error_prev
            parsed = second
            raw = (
                "## OBJETIVO\nObjetivo de analisis de datos para deportes.\n\n"
                "## DATASET\nrows = [{'equipo':'A','goles':1}]\n\n"
                "## CODIGO\n# comentario\n# comentario\n# comentario\n# comentario\n# comentario\n"
                "df = pd.DataFrame(rows)\nprint(df)\n\n"
                "## EXPLICACION\nEste ejemplo usa datos para analisis de datos en espanol."
            )

        return GenerationTrace(
            parsed=parsed,
            raw_text_original=raw,
            sanitized_text=raw,
            prompt_sent="prompt",
            parse_ok=True,
            used_fallback=False,
            model_backend="local",
            post_processed=False,
        )


def test_validating_repair_loop_fixes_english_objective() -> None:
    loop = ValidatingRepairLoop(
        generator=StubTraceGenerator(),
        executor=CodeExecutor(),
        validator=EducationalValidator(),
        max_attempts=3,
    )
    result = loop.run(_request())
    assert result.tests_passed is True
    assert result.educational_passed is True
    assert result.objetivo.lower().startswith("objetivo")


class StubDatasetMismatchGenerator:
    def generate_with_raw(
        self,
        request: GenerateRequest,
        error_prev: str | None = None,
        attempt: int = 1,
        use_rag: bool = False,
    ) -> GenerationTrace:
        _ = request, error_prev, attempt, use_rag
        dataset = DatasetInfo(
            nombre="demo",
            data=[{"equipo": "A", "goles": 3, "partidos": 1}, {"equipo": "B", "goles": 1, "partidos": 2}],
            codigo_carga="df = pd.DataFrame(rows)",
        )
        parsed = ParsedLLMResponse(
            objetivo="Objetivo de analisis de datos para deportes.",
            dataset=dataset,
            codigo=(
                "import pandas as pd\n"
                "# c1\n# c2\n# c3\n# c4\n# c5\n"
                "rows = [{'region':'Norte','ventas':10,'mes':1}]\n"
                "df = pd.DataFrame(rows)\n"
                "print(df.head())"
            ),
            explicacion=["Paso 1", "Paso 2"],
            ejercicio="Ejercicio",
        )
        raw = (
            "## OBJETIVO\nObjetivo de analisis de datos para deportes.\n\n"
            "## DATASET\n```python\nrows=[{'equipo':'A','goles':3,'partidos':1}]\n```\n\n"
            "## CODIGO\n```python\nrows=[{'region':'Norte','ventas':10,'mes':1}]\n```\n\n"
            "## EXPLICACION\n- Paso 1\n- Paso 2"
        )
        return GenerationTrace(
            parsed=parsed,
            raw_text_original=raw,
            sanitized_text=raw,
            prompt_sent="prompt",
            parse_ok=True,
            used_fallback=False,
            model_backend="local",
            post_processed=False,
        )


def test_validating_repair_loop_rebuilds_code_when_dataset_is_overridden() -> None:
    loop = ValidatingRepairLoop(
        generator=StubDatasetMismatchGenerator(),
        executor=CodeExecutor(),
        validator=EducationalValidator(),
        max_attempts=2,
    )
    result = loop.run(_request())
    assert result.tests_passed is True
    assert result.educational_passed is True
    assert result.attempts == 1
    code_low = result.codigo.lower().replace(" ", "")
    assert "pd.dataframe(rows)" in code_low
    assert "groupby('equipo'" in code_low
    assert "['goles']" in code_low
    assert "region" not in code_low

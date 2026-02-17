from backend.executor import CodeExecutor
from backend.generator import GenerationTrace
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
# Comentario educativo 1
# Comentario educativo 2
# Comentario educativo 3
# Comentario educativo 4
# Comentario educativo 5
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
            objetivo="What is the highest goals?",
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
            objetivo="Aprender groupby en deportes.",
            dataset=first.dataset,
            codigo=first.codigo,
            explicacion=first.explicacion,
            ejercicio=first.ejercicio,
        )

        if self.calls == 1:
            parsed = first
            raw = (
                "## OBJETIVO\nWhat is the highest goals?\n\n"
                "## DATASET\nrows = [{'equipo':'A','goles':1}]\n\n"
                "## CODIGO\n# comentario\n# comentario\n# comentario\n# comentario\n# comentario\n"
                "df = pd.DataFrame(rows)\nprint(df)\n\n"
                "## EXPLICACION\nEste ejemplo usa datos y analisis en espanol con explicacion detallada."
            )
        else:
            assert error_prev is not None
            assert "Reescribe SOLO la sección ## OBJETIVO en español" in error_prev
            parsed = second
            raw = (
                "## OBJETIVO\nAprender groupby en deportes.\n\n"
                "## DATASET\nrows = [{'equipo':'A','goles':1}]\n\n"
                "## CODIGO\n# comentario\n# comentario\n# comentario\n# comentario\n# comentario\n"
                "df = pd.DataFrame(rows)\nprint(df)\n\n"
                "## EXPLICACION\nEste ejemplo usa datos y analisis en espanol con explicacion detallada."
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


def test_validating_repair_loop_fixes_english_objective_on_retry() -> None:
    loop = ValidatingRepairLoop(
        generator=StubTraceGenerator(),
        executor=CodeExecutor(),
        validator=EducationalValidator(),
        max_attempts=3,
    )
    result = loop.run(_request())
    assert result.tests_passed is True
    assert result.educational_passed is True
    assert result.attempts == 2


class StubCommentsTraceGenerator:
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
        dataset = DatasetInfo(
            nombre="demo",
            data=[{"equipo": "A", "goles": 1}, {"equipo": "B", "goles": 2}],
            codigo_carga="df = pd.DataFrame(rows)",
        )
        if self.calls == 1:
            parsed = ParsedLLMResponse(
                objetivo="Aprender groupby en deportes.",
                dataset=dataset,
                codigo=(
                    "import pandas as pd\n"
                    "df = pd.DataFrame(rows)\n"
                    "resultado = df.groupby('equipo')['goles'].sum()\n"
                    "print(resultado)"
                ),
                explicacion=["Paso 1", "Paso 2"],
                ejercicio="Ejercicio",
            )
        else:
            assert error_prev is not None
            assert "Agrega comentarios educativos con #" in error_prev
            parsed = ParsedLLMResponse(
                objetivo="Aprender groupby en deportes.",
                dataset=dataset,
                codigo=(
                    "import pandas as pd\n"
                    "# c1\n# c2\n# c3\n# c4\n# c5\n"
                    "df = pd.DataFrame(rows)\n"
                    "print(df.groupby('equipo')['goles'].sum())"
                ),
                explicacion=["Paso 1", "Paso 2"],
                ejercicio="Ejercicio",
            )
        raw = (
            "## OBJETIVO\nAprender groupby en deportes.\n\n"
            "## DATASET\nrows = [{'equipo':'A','goles':1}]\n\n"
            "## CODIGO\n"
            "df = pd.DataFrame(rows)\nresultado = df.groupby('equipo')['goles'].sum()\nprint(resultado)\n\n"
            "## EXPLICACION\nEste ejemplo usa datos y analisis en espanol con explicacion detallada."
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


def test_validating_repair_loop_fixes_missing_comments_on_retry() -> None:
    loop = ValidatingRepairLoop(
        generator=StubCommentsTraceGenerator(),
        executor=CodeExecutor(),
        validator=EducationalValidator(),
        max_attempts=3,
    )
    result = loop.run(_request())
    assert result.tests_passed is True
    assert result.educational_passed is True
    assert result.attempts == 2


class StubLeakTraceGenerator:
    def __init__(self) -> None:
        self.calls = 0
        self.patch_calls = 0

    def generate_with_raw(
        self,
        request: GenerateRequest,
        error_prev: str | None = None,
        attempt: int = 1,
        use_rag: bool = False,
    ) -> GenerationTrace:
        _ = request, attempt, use_rag
        self.calls += 1
        dataset = DatasetInfo(
            nombre="demo",
            data=[{"equipo": "A", "goles": 1}, {"equipo": "B", "goles": 2}],
            codigo_carga="df = pd.DataFrame(rows)",
        )
        parsed = ParsedLLMResponse(
            objetivo="Aprender groupby en deportes.",
            dataset=dataset,
            codigo=(
                "import pandas as pd\n"
                "df = pd.DataFrame(rows)\n"
                "Errores:\n"
                "- Falta comentario\n"
                "print(df.head())"
            ),
            explicacion=["Paso 1", "Paso 2"],
            ejercicio="Ejercicio",
        )
        raw = "## OBJETIVO\n...\n## DATASET\n...\n## CODIGO\nErrores:\n- Falta comentario\n## EXPLICACION\n..."

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

    def generate_patch(
        self,
        *,
        request: GenerateRequest,
        section: str,
        instruction: str,
        previous_text: str,
    ) -> tuple[str, str, str, bool, str]:
        _ = request
        self.patch_calls += 1
        if section == "codigo":
            assert ("bloque completo de ## CODIGO" in instruction) or ("asegurando df = pd.DataFrame(rows)" in instruction)
            raw = (
                "```python\n"
                "import pandas as pd\n"
                "# c1\n# c2\n# c3\n# c4\n# c5\n"
                "df = pd.DataFrame(rows)\n"
                "print(df.groupby('equipo')['goles'].sum())\n"
                "```"
            )
            return raw, raw, "patch-prompt-codigo", False, "local"
        assert section == "objetivo"
        raw = "Aprender groupby en deportes con análisis de goles."
        return raw, raw, "patch-prompt-objetivo", False, "local"


def test_validating_repair_loop_fixes_prompt_leak_in_code_on_retry() -> None:
    loop = ValidatingRepairLoop(
        generator=StubLeakTraceGenerator(),
        executor=CodeExecutor(),
        validator=EducationalValidator(),
        max_attempts=3,
    )
    result = loop.run(_request())
    assert result.tests_passed is True
    assert result.educational_passed is True
    assert result.attempts == 1
    code_low = result.codigo.lower()
    assert "errores:" not in code_low
    assert "salida_previa" not in code_low
    assert "instrucciones_de_reparacion" not in code_low

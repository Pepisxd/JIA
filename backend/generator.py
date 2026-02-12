from __future__ import annotations

import json
import os
from pathlib import Path

from backend.llm_client import AnthropicClient, LLMClientError
from backend.models import GenerateRequest, ParsedLLMResponse
from backend.parser import ParseError, parse_llm_response
from backend.rag.retriever import ExampleRetriever

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "template_base.txt"


class GenerationError(RuntimeError):
    pass


def _context_rows(contexto: str) -> list[dict[str, int | str]]:
    packs: dict[str, list[dict[str, int | str]]] = {
        "deportes": [
            {"equipo": "Tigres", "goles": 3},
            {"equipo": "Tigres", "goles": 2},
            {"equipo": "Lobos", "goles": 1},
            {"equipo": "Lobos", "goles": 4},
        ],
        "finanzas": [
            {"region": "Norte", "ventas": 1200},
            {"region": "Norte", "ventas": 900},
            {"region": "Sur", "ventas": 1400},
            {"region": "Sur", "ventas": 1100},
        ],
        "videojuegos": [
            {"juego": "RacingX", "sesiones": 120},
            {"juego": "RacingX", "sesiones": 95},
            {"juego": "ArenaZ", "sesiones": 140},
            {"juego": "ArenaZ", "sesiones": 100},
        ],
        "ciencia": [
            {"experimento": "A", "medicion": 12},
            {"experimento": "A", "medicion": 9},
            {"experimento": "B", "medicion": 14},
            {"experimento": "B", "medicion": 10},
        ],
    }
    return packs[contexto]


def _context_columns(contexto: str) -> tuple[str, str]:
    if contexto == "deportes":
        return "equipo", "goles"
    if contexto == "finanzas":
        return "region", "ventas"
    if contexto == "videojuegos":
        return "juego", "sesiones"
    return "experimento", "medicion"


def _render_fallback_response(request: GenerateRequest, error_prev: str | None) -> str:
    rows = _context_rows(request.contexto)
    group_col, value_col = _context_columns(request.contexto)
    dataset_name = f"dataset_{request.contexto}"
    repair_comment = f"# Reparacion aplicada: {error_prev}\n" if error_prev else ""
    if request.tema == "pandas_filtrado":
        code = f"""import pandas as pd
# Cargamos datos tabulares para practicar filtrado.
{repair_comment}df = pd.DataFrame({rows})
filtrado = df[df["{value_col}"] > df["{value_col}"].mean()]
print(filtrado)
"""
        objetivo = "Aprender filtrado de datos con pandas usando condiciones booleanas."
        explicacion = [
            "Creamos un DataFrame con datos de contexto.",
            "Calculamos una condicion booleana sobre la metrica.",
            "Mostramos solo los registros que cumplen la condicion.",
        ]
        ejercicio = "Prueba ahora un filtro con `<` en lugar de `>`."
    elif request.tema == "pandas_lectura":
        code = """import pandas as pd
from io import StringIO
# Simulamos un CSV en memoria para practicar lectura.
csv_text = "item,valor\\nA,10\\nB,20\\nC,30\\n"
df = pd.read_csv(StringIO(csv_text))
print(df.head())
"""
        objetivo = "Aprender lectura de datos con pandas y vista preliminar con head()."
        explicacion = [
            "Creamos una fuente CSV de ejemplo.",
            "Leemos los datos con `pd.read_csv`.",
            "Visualizamos las primeras filas con `head()`.",
        ]
        ejercicio = "Cambia el CSV agregando otra columna y vuelve a mostrar `head()`."
        rows = [{"item": "A", "valor": 10}, {"item": "B", "valor": 20}, {"item": "C", "valor": 30}]
        dataset_name = "dataset_lectura_csv"
    else:
        code = f"""import pandas as pd
# Cargamos datos de ejemplo en un DataFrame.
{repair_comment}df = pd.DataFrame({rows})
resultado = df.groupby("{group_col}")["{value_col}"].sum().reset_index()
print(resultado)
"""
        objetivo = "Aprender a agrupar datos y calcular agregaciones con pandas."
        explicacion = [
            "Cargamos datos de ejemplo en un DataFrame.",
            f"Agrupamos por la columna categórica `{group_col}`.",
            "Aplicamos una agregacion para resumir resultados.",
        ]
        ejercicio = "Cambia la suma por promedio usando `.mean()` y compara resultados."

    dataset_json = json.dumps(
        {"nombre": dataset_name, "data": rows, "codigo_carga": f"df = pd.DataFrame({rows})"},
        ensure_ascii=False,
        indent=2,
    )
    exp_lines = "\n".join(f"- {line}" for line in explicacion)

    return f"""OBJETIVO: {objetivo}

DATASET_JSON:
```json
{dataset_json}
```

CODIGO:
```python
{code}
```

EXPLICACION:
{exp_lines}

EJERCICIO: {ejercicio}
"""


class CodeGenerator:
    def __init__(
        self,
        llm_client: AnthropicClient | None = None,
        template_path: Path = TEMPLATE_PATH,
        retriever: ExampleRetriever | None = None,
    ) -> None:
        self.llm_client = llm_client or AnthropicClient()
        self.template = template_path.read_text(encoding="utf-8")
        self.use_real_llm = os.getenv("USE_REAL_LLM", "false").strip().lower() == "true"
        prefer_chroma = os.getenv("RAG_PREFER_CHROMA", "false").strip().lower() == "true"
        self.retriever = retriever or ExampleRetriever(prefer_chroma=prefer_chroma)

    def _build_prompt(self, request: GenerateRequest, error_prev: str | None, use_rag: bool) -> str:
        prompt = self.template
        prompt = prompt.replace("{tema}", request.tema)
        prompt = prompt.replace("{nivel}", request.nivel)
        prompt = prompt.replace("{contexto}", request.contexto)
        prompt = prompt.replace("{tipo}", request.tipo)
        prompt = prompt.replace("{error_prev}", error_prev or "N/A")
        if use_rag:
            query = f"{request.tema} {request.contexto} {request.nivel}"
            examples = self.retriever.retrieve(query=query, limit=3)
            if examples:
                rag_lines = ["\nEJEMPLOS_RAG_RELEVANTES:"]
                for idx, item in enumerate(examples, start=1):
                    rag_lines.append(f"{idx}. Q: {item.get('question', '')}")
                    rag_lines.append(f"   A: {item.get('answer', '')}")
                prompt = f"{prompt}\n" + "\n".join(rag_lines)
        return prompt

    def _generate_raw(self, request: GenerateRequest, error_prev: str | None, use_rag: bool) -> str:
        prompt = self._build_prompt(request, error_prev, use_rag=use_rag)
        if not self.use_real_llm or not self.llm_client.is_configured:
            return _render_fallback_response(request, error_prev)
        return self.llm_client.generate(prompt=prompt, system_prompt="Genera material educativo claro y ejecutable.")

    def generate(
        self,
        request: GenerateRequest,
        error_prev: str | None = None,
        attempt: int = 1,
        use_rag: bool = False,
    ) -> ParsedLLMResponse:
        try:
            raw = self._generate_raw(request, error_prev, use_rag=use_rag)
            return parse_llm_response(raw)
        except ParseError:
            # Si el LLM no respeta el formato exacto, mantenemos el flujo vivo con fallback determinista.
            fallback_raw = _render_fallback_response(request, error_prev)
            return parse_llm_response(fallback_raw)
        except LLMClientError as exc:
            raise GenerationError(f"Fallo en generacion (attempt {attempt}): {exc}") from exc

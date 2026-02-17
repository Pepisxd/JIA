from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from random import Random

from backend.llm_client import AnthropicClient, LLMClientError
from backend.local_model import get_local_model
from backend.models import GenerateRequest, ParsedLLMResponse
from backend.parser import ParseError, parse_llm_response
from backend.rag.retriever import ExampleRetriever

SYSTEM_PROMPT_V2 = """Eres un asistente educativo de Python para ciencia de datos.
REGLAS OBLIGATORIAS:
1) Responde EXCLUSIVAMENTE en espanol.
2) Prohibido usar rutas o leer archivos: /home, /kaggle, /content, ../input, pd.read_csv, read_parquet, read_excel.
3) SIEMPRE incluye dataset sintetico pequeno y el codigo lo construye con pd.DataFrame.
4) EXPLICACION especifica, menciona columnas/variables reales.
FORMATO EXACTO:
## OBJETIVO
## DATASET
## CODIGO
## EXPLICACION"""

LOGGER = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    pass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class GenerationTrace:
    parsed: ParsedLLMResponse
    raw_text_original: str
    sanitized_text: str
    prompt_sent: str
    parse_ok: bool
    used_fallback: bool
    model_backend: str
    post_processed: bool = False


def _sanitize_model_text(text: str) -> str:
    # Sanitizer only: remove generation residue tokens and normalize whitespace.
    out = text.replace("</s>", " ").replace("<s>", " ")
    out = re.sub(r"<\|[^>]+?\|>", " ", out)
    out = re.sub(r"\r\n?", "\n", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _context_rows(contexto: str) -> list[dict[str, int | str]]:
    packs: dict[str, list[dict[str, int | str]]] = {
        "deportes": [
            {"equipo": "Tigres", "goles": 3, "partidos": 2},
            {"equipo": "Tigres", "goles": 2, "partidos": 2},
            {"equipo": "Lobos", "goles": 1, "partidos": 1},
            {"equipo": "Lobos", "goles": 4, "partidos": 3},
            {"equipo": "Rojos", "goles": 2, "partidos": 2},
            {"equipo": "Rojos", "goles": 5, "partidos": 4},
            {"equipo": "Azules", "goles": 1, "partidos": 1},
            {"equipo": "Azules", "goles": 3, "partidos": 2},
        ],
        "finanzas": [
            {"region": "Norte", "ventas": 1200, "costos": 700},
            {"region": "Norte", "ventas": 900, "costos": 500},
            {"region": "Sur", "ventas": 1400, "costos": 850},
            {"region": "Sur", "ventas": 1100, "costos": 620},
            {"region": "Este", "ventas": 1700, "costos": 910},
            {"region": "Este", "ventas": 1550, "costos": 870},
            {"region": "Oeste", "ventas": 980, "costos": 560},
            {"region": "Oeste", "ventas": 1320, "costos": 790},
        ],
        "videojuegos": [
            {"juego": "RacingX", "sesiones": 120, "usuarios": 50},
            {"juego": "RacingX", "sesiones": 95, "usuarios": 41},
            {"juego": "ArenaZ", "sesiones": 140, "usuarios": 60},
            {"juego": "ArenaZ", "sesiones": 100, "usuarios": 46},
            {"juego": "MysticQ", "sesiones": 130, "usuarios": 58},
            {"juego": "MysticQ", "sesiones": 110, "usuarios": 49},
            {"juego": "NovaR", "sesiones": 84, "usuarios": 36},
            {"juego": "NovaR", "sesiones": 90, "usuarios": 38},
        ],
        "ciencia": [
            {"experimento": "A", "medicion": 12, "temperatura": 21.4},
            {"experimento": "A", "medicion": 9, "temperatura": 20.8},
            {"experimento": "B", "medicion": 14, "temperatura": 22.6},
            {"experimento": "B", "medicion": 10, "temperatura": 21.9},
            {"experimento": "C", "medicion": 11, "temperatura": 23.1},
            {"experimento": "C", "medicion": 13, "temperatura": 22.8},
            {"experimento": "D", "medicion": 8, "temperatura": 20.1},
            {"experimento": "D", "medicion": 12, "temperatura": 21.0},
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


def _rows_for_prompt(contexto: str, seed: int) -> list[dict[str, int | str]]:
    rows = _context_rows(contexto)
    rng = Random(seed)
    target = rng.randint(8, 12)
    if len(rows) <= target:
        return rows
    return rows[:target]


def _render_fallback_response(request: GenerateRequest, error_prev: str | None) -> str:
    rows = _rows_for_prompt(request.contexto, seed=abs(hash(f"{request.tema}:{request.contexto}:{request.nivel}")) % 10_000)
    group_col, value_col = _context_columns(request.contexto)
    _ = error_prev

    if request.tema == "pandas_filtrado":
        code = (
            "import pandas as pd\n\n"
            f"rows = {rows}\n"
            "df = pd.DataFrame(rows)\n"
            f"resultado = df[df['{value_col}'] > df['{value_col}'].mean()].copy()\n"
            "print(resultado)\n"
        )
    elif request.tema == "pandas_lectura":
        code = (
            "import pandas as pd\n\n"
            f"rows = {rows}\n"
            "df = pd.DataFrame(rows)\n"
            "print(df.head())\n"
            "print(df.describe(include='all'))\n"
        )
    elif request.tema == "matplotlib_basico":
        code = (
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            f"rows = {rows}\n"
            "df = pd.DataFrame(rows)\n"
            f"serie = df.groupby('{group_col}', as_index=False)['{value_col}'].sum()\n"
            f"plt.bar(serie['{group_col}'], serie['{value_col}'])\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    elif request.tema == "numpy_basico":
        code = (
            "import numpy as np\n"
            "import pandas as pd\n\n"
            f"rows = {rows}\n"
            "df = pd.DataFrame(rows)\n"
            "matriz = df.select_dtypes(include='number').to_numpy()\n"
            "print('media:', np.mean(matriz).round(2))\n"
        )
    else:
        code = (
            "import pandas as pd\n\n"
            f"rows = {rows}\n"
            "df = pd.DataFrame(rows)\n"
            f"resultado = df.groupby('{group_col}', as_index=False)['{value_col}'].sum()\n"
            "print(resultado)\n"
        )

    dataset_block = (
        "```python\n"
        f"rows = {json.dumps(rows, ensure_ascii=False, indent=2)}\n"
        "df = pd.DataFrame(rows)\n"
        "print(df.head())\n"
        "```"
    )
    explanation = (
        f"- Se define un dataset sintetico con columnas como `{group_col}` y `{value_col}` para trabajar offline.\n"
        f"- El codigo construye `df` con `pd.DataFrame(rows)` y aplica la operacion principal del tema `{request.tema}`.\n"
        "- Se imprime un resultado verificable para facilitar la validacion y la practica."
    )
    return (
        "## OBJETIVO\n"
        f"Aprender {request.tema} en contexto de {request.contexto} con un ejemplo reproducible.\n\n"
        "## DATASET\n"
        f"{dataset_block}\n\n"
        "## CODIGO\n"
        f"```python\n{code}```\n\n"
        "## EXPLICACION\n"
        f"{explanation}\n\n"
        "EJERCICIO: Cambia una agregacion o filtro y compara el resultado."
    )


def _render_fallback_patch(request: GenerateRequest, section: str) -> str:
    rows = _rows_for_prompt(request.contexto, seed=42 + abs(hash(request.tema)) % 1000)
    group_col, value_col = _context_columns(request.contexto)
    sec = section.lower().strip()
    if sec == "codigo":
        return (
            "```python\n"
            "import pandas as pd\n"
            "# Comentario educativo 1\n"
            "# Comentario educativo 2\n"
            "# Comentario educativo 3\n"
            "# Comentario educativo 4\n"
            "# Comentario educativo 5\n"
            f"rows = {rows}\n"
            "df = pd.DataFrame(rows)\n"
            f"resultado = df.groupby('{group_col}', as_index=False)['{value_col}'].sum()\n"
            "print(resultado)\n"
            "```"
        )
    if sec == "objetivo":
        return f"Aprender {request.tema} en contexto de {request.contexto} con codigo ejecutable en Python."
    if sec == "explicacion":
        return (
            "## EXPLICACION\n"
            f"- El ejemplo usa columnas `{group_col}` y `{value_col}` para practicar el tema.\n"
            "- Se construye el DataFrame con rows y se ejecuta una transformacion principal.\n"
            "- El resultado final se imprime para verificar el comportamiento."
        )
    return ""


class CodeGenerator:
    def __init__(
        self,
        llm_client: AnthropicClient | None = None,
        retriever: ExampleRetriever | None = None,
    ) -> None:
        self.use_local_model = _env_bool("USE_LOCAL_MODEL", True)
        self.llm_client = llm_client or AnthropicClient()
        self.use_real_llm = _env_bool("USE_REAL_LLM", False)
        prefer_chroma = _env_bool("RAG_PREFER_CHROMA", False)
        self.retriever = retriever or ExampleRetriever(prefer_chroma=prefer_chroma)
        self.local_model = None
        if self.use_local_model:
            model_base = os.getenv("MODEL_BASE", "codellama/CodeLlama-7b-Instruct-hf")
            adapter_dir = os.getenv("MODEL_PATH", "./models/codellama-edugen-v2")
            device_map = os.getenv("MODEL_DEVICE_MAP", "auto")
            local_model_required = _env_bool("LOCAL_MODEL_REQUIRED", False)
            try:
                self.local_model = get_local_model(model_base=model_base, adapter_dir=adapter_dir, device_map=device_map)
            except Exception as exc:  # noqa: BLE001
                if local_model_required:
                    raise
                self.local_model = None
                self.use_local_model = False
                LOGGER.warning(
                    "No se pudo cargar LocalEduModel (%s). Se usa fallback (LLM remoto o respuesta local determinista).",
                    exc,
                )

    def _build_prompt(self, request: GenerateRequest, error_prev: str | None, use_rag: bool) -> str:
        rows = _rows_for_prompt(request.contexto, seed=42 + abs(hash(request.tema)) % 1000)
        rag_text = ""
        if use_rag:
            query = f"{request.tema} {request.contexto} {request.nivel}"
            examples = self.retriever.retrieve(query=query, limit=2)
            if examples:
                rag_lines = ["Ejemplos recuperados (resumen):"]
                for idx, item in enumerate(examples, start=1):
                    rag_lines.append(f"{idx}. {item.get('question', '')[:180]}")
                rag_text = "\n" + "\n".join(rag_lines)

        err_text = f"\nError previo a corregir: {error_prev}" if error_prev else ""
        dataset_hint = json.dumps(rows, ensure_ascii=False, indent=2)
        return (
            f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT_V2}\n<</SYS>>\n\n"
            "Genera un ejemplo educativo ejecutable.\n"
            f"Tema: {request.tema}\n"
            f"Nivel: {request.nivel}\n"
            f"Contexto: {request.contexto}\n"
            f"Tipo: {request.tipo}\n"
            "Usa 8-15 filas y columnas coherentes con el contexto.\n"
            f"Sugerencia de dataset base:\n{dataset_hint}\n"
            f"{rag_text}"
            f"{err_text}\n"
            "Recuerda: SOLO espanol y sin lecturas de archivos.\n"
            "[/INST]"
        )

    def _generate_from_backend(self, prompt: str, request: GenerateRequest, error_prev: str | None) -> tuple[str, str, bool]:
        if self.use_local_model and self.local_model is not None:
            return (
                self.local_model.generate(
                    prompt=prompt,
                    max_new_tokens=int(os.getenv("LOCAL_MODEL_MAX_NEW_TOKENS", "700")),
                ),
                "local",
                False,
            )

        if not self.use_real_llm or not self.llm_client.is_configured:
            return _render_fallback_response(request, error_prev), "deterministic", True

        return self.llm_client.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT_V2), "remote", False

    def _generate_raw(self, request: GenerateRequest, error_prev: str | None, use_rag: bool) -> tuple[str, str, bool, str]:
        prompt = self._build_prompt(request, error_prev, use_rag=use_rag)
        raw, backend, used_fallback = self._generate_from_backend(prompt=prompt, request=request, error_prev=error_prev)
        return raw, backend, used_fallback, prompt

    def _build_patch_prompt(self, *, section: str, instruction: str, previous_text: str) -> str:
        sec = section.upper()
        return (
            f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT_V2}\n<</SYS>>\n\n"
            "MODO PATCH QUIRURGICO.\n"
            f"Seccion a corregir: ## {sec}\n"
            f"Instruccion: {instruction}\n\n"
            "REGLAS ESTRICTAS:\n"
            "- NO copies ERRORES ni SALIDA_PREVIA dentro de codigo.\n"
            "- Devuelve solamente la seccion solicitada.\n"
            "- Si se solicita CODIGO, devuelve un unico bloque fence ```python ...``` y nada mas.\n"
            "- No incluyas encabezados adicionales.\n\n"
            "=== SALIDA PREVIA (NO COPIAR) ===\n"
            f"{previous_text}\n"
            "[/INST]"
        )

    def generate_patch(
        self,
        *,
        request: GenerateRequest,
        section: str,
        instruction: str,
        previous_text: str,
    ) -> tuple[str, str, str, bool, str]:
        prompt = self._build_patch_prompt(section=section, instruction=instruction, previous_text=previous_text)

        if not self.use_real_llm and (not self.use_local_model or self.local_model is None):
            raw = _render_fallback_patch(request, section=section)
            return raw, _sanitize_model_text(raw), prompt, True, "deterministic"

        raw, backend, used_fallback = self._generate_from_backend(prompt=prompt, request=request, error_prev=None)
        sanitized = _sanitize_model_text(raw)
        return raw, sanitized, prompt, used_fallback, backend

    def generate_with_raw(
        self,
        request: GenerateRequest,
        error_prev: str | None = None,
        attempt: int = 1,
        use_rag: bool = False,
    ) -> GenerationTrace:
        raw_original = ""
        sanitized = ""
        model_backend = "deterministic"
        used_fallback = False
        prompt_sent = ""
        try:
            raw_original, model_backend, used_fallback, prompt_sent = self._generate_raw(request, error_prev, use_rag=use_rag)
            sanitized = _sanitize_model_text(raw_original)
            parsed = parse_llm_response(sanitized)
            return GenerationTrace(
                parsed=parsed,
                raw_text_original=raw_original,
                sanitized_text=sanitized,
                prompt_sent=prompt_sent,
                parse_ok=True,
                used_fallback=used_fallback,
                model_backend=model_backend,
                post_processed=False,
            )
        except ParseError:
            fallback_raw = _render_fallback_response(request, error_prev)
            fallback_sanitized = _sanitize_model_text(fallback_raw)
            parsed = parse_llm_response(fallback_sanitized)
            return GenerationTrace(
                parsed=parsed,
                raw_text_original=raw_original or fallback_raw,
                sanitized_text=fallback_sanitized,
                prompt_sent=prompt_sent,
                parse_ok=False,
                used_fallback=True,
                model_backend="deterministic",
                post_processed=False,
            )
        except LLMClientError as exc:
            raise GenerationError(f"Fallo en generacion (attempt {attempt}): {exc}") from exc

    def generate(
        self,
        request: GenerateRequest,
        error_prev: str | None = None,
        attempt: int = 1,
        use_rag: bool = False,
    ) -> ParsedLLMResponse:
        trace = self.generate_with_raw(request=request, error_prev=error_prev, attempt=attempt, use_rag=use_rag)
        return trace.parsed

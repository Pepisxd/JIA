from __future__ import annotations

import ast
import json
import re

from backend.models import DatasetInfo, ParsedLLMResponse


class ParseError(ValueError):
    pass


def _extract_fenced_block(text: str, language: str) -> str | None:
    pattern = rf"```{language}\s*(.*?)```"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _extract_section_block(text: str, key: str) -> str | None:
    # V2 style: ## KEY
    md_header = re.search(rf"(?im)^\s*##\s*{re.escape(key)}\s*$", text)
    if md_header:
        start = md_header.end()
        remaining = text[start:]
        stop = re.search(r"(?im)^\s*##\s*[A-Z_]+\s*$", remaining)
        block = remaining[: stop.start()] if stop else remaining
        value = block.strip()
        return value or None

    # Legacy style: KEY:
    legacy_header = re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*$", text)
    if legacy_header:
        start = legacy_header.end()
        remaining = text[start:]
        stop = re.search(r"(?im)^\s*[A-Z_]+?\s*:\s*", remaining)
        block = remaining[: stop.start()] if stop else remaining
        value = block.strip()
        return value or None
    return None


def _extract_line_value(text: str, key: str) -> str | None:
    pattern = rf"(?im)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$"
    match = re.search(pattern, text)
    if not match:
        return None
    return match.group(1).strip()


def _extract_list_items(text: str, key: str) -> list[str]:
    block = _extract_section_block(text, key) or ""
    items: list[str] = []
    for line in block.splitlines():
        clean = line.strip()
        if clean.startswith("-"):
            value = clean[1:].strip()
            if value:
                items.append(value)
        elif clean and not clean.startswith("```"):
            items.append(clean)
    return items


def _parse_dataset_json(text: str) -> DatasetInfo | None:
    json_block = _extract_fenced_block(text, "json")
    if not json_block:
        return None
    try:
        raw = json.loads(json_block)
    except json.JSONDecodeError as exc:
        raise ParseError(f"JSON de dataset invalido: {exc}") from exc
    return DatasetInfo.model_validate(raw)


def _parse_dataset_v2(text: str) -> DatasetInfo | None:
    section = _extract_section_block(text, "DATASET")
    if not section:
        return None
    python_block = _extract_fenced_block(section, "python") or section
    rows_match = re.search(r"(?is)\brows\s*=\s*(\[[\s\S]*?\])", python_block)
    if not rows_match:
        return None
    try:
        rows = ast.literal_eval(rows_match.group(1))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(rows, list):
        return None
    return DatasetInfo(
        nombre="dataset_sintetico",
        data=rows,
        codigo_carga="df = pd.DataFrame(rows)",
    )


def _parse_dataset(text: str) -> DatasetInfo:
    dataset = _parse_dataset_json(text)
    if dataset:
        return dataset
    dataset_v2 = _parse_dataset_v2(text)
    if dataset_v2:
        return dataset_v2
    raise ParseError("No se encontro seccion de dataset valida.")


def _parse_objective(text: str) -> str | None:
    legacy = _extract_line_value(text, "OBJETIVO")
    if legacy:
        return legacy
    section = _extract_section_block(text, "OBJETIVO")
    if not section:
        return None
    for line in section.splitlines():
        clean = line.strip()
        if clean and not clean.startswith("```"):
            return clean
    return None


def _parse_code(text: str) -> str | None:
    code_section = _extract_section_block(text, "CODIGO")
    if code_section:
        block = _extract_fenced_block(code_section, "python")
        if block:
            return block
        return code_section.strip() or None
    # Legacy fallback: first python fenced block
    return _extract_fenced_block(text, "python")


def parse_llm_response(text: str) -> ParsedLLMResponse:
    objetivo = _parse_objective(text)
    ejercicio = _extract_line_value(text, "EJERCICIO")
    codigo = _parse_code(text)
    explicacion = _extract_list_items(text, "EXPLICACION")

    if not objetivo:
        raise ParseError("No se encontro OBJETIVO.")
    if not codigo:
        raise ParseError("No se encontro bloque de CODIGO en python.")
    if not ejercicio:
        ejercicio = "Extiende el ejercicio cambiando una agregacion o filtro y compara el resultado."

    dataset = _parse_dataset(text)
    return ParsedLLMResponse(
        objetivo=objetivo,
        dataset=dataset,
        codigo=codigo,
        explicacion=explicacion,
        ejercicio=ejercicio,
    )

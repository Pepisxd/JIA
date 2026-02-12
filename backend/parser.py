from __future__ import annotations

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


def _extract_line_value(text: str, key: str) -> str | None:
    pattern = rf"(?im)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$"
    match = re.search(pattern, text)
    if not match:
        return None
    return match.group(1).strip()


def _extract_list_items(text: str, key: str) -> list[str]:
    header = re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*$", text)
    if not header:
        return []

    start = header.end()
    remaining = text[start:]
    stop = re.search(r"(?im)^\s*[A-Z_]+?\s*:\s*", remaining)
    block = remaining[: stop.start()] if stop else remaining
    items: list[str] = []
    for line in block.splitlines():
        clean = line.strip()
        if clean.startswith("-"):
            value = clean[1:].strip()
            if value:
                items.append(value)
    return items


def _parse_dataset(text: str) -> DatasetInfo:
    json_block = _extract_fenced_block(text, "json")
    if not json_block:
        raise ParseError("No se encontro bloque JSON para dataset.")
    try:
        raw = json.loads(json_block)
    except json.JSONDecodeError as exc:
        raise ParseError(f"JSON de dataset invalido: {exc}") from exc
    return DatasetInfo.model_validate(raw)


def parse_llm_response(text: str) -> ParsedLLMResponse:
    objetivo = _extract_line_value(text, "OBJETIVO")
    ejercicio = _extract_line_value(text, "EJERCICIO")
    codigo = _extract_fenced_block(text, "python")
    explicacion = _extract_list_items(text, "EXPLICACION")

    if not objetivo:
        raise ParseError("No se encontro OBJETIVO.")
    if not ejercicio:
        raise ParseError("No se encontro EJERCICIO.")
    if not codigo:
        raise ParseError("No se encontro bloque de CODIGO en python.")

    dataset = _parse_dataset(text)
    return ParsedLLMResponse(
        objetivo=objetivo,
        dataset=dataset,
        codigo=codigo,
        explicacion=explicacion,
        ejercicio=ejercicio,
    )

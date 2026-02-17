import pytest

from backend.parser import ParseError, parse_llm_response


def test_parse_llm_response_ok() -> None:
    raw = """
OBJETIVO: Aprender groupby con pandas

DATASET_JSON:
```json
{
  "nombre": "estadisticas_futbol",
  "data": [{"equipo": "A", "goles": 2}, {"equipo": "B", "goles": 3}],
  "codigo_carga": "df = pd.DataFrame([...])"
}
```

CODIGO:
```python
import pandas as pd
df = pd.DataFrame([{"equipo":"A","goles":2}])
print(df)
```

EXPLICACION:
- Paso 1
- Paso 2

EJERCICIO: Cambia sum por mean
"""
    parsed = parse_llm_response(raw)
    assert parsed.objetivo == "Aprender groupby con pandas"
    assert parsed.dataset.nombre == "estadisticas_futbol"
    assert "import pandas as pd" in parsed.codigo
    assert parsed.explicacion == ["Paso 1", "Paso 2"]
    assert parsed.ejercicio == "Cambia sum por mean"


def test_parse_llm_response_fails_without_python_block() -> None:
    raw = """
OBJETIVO: X
DATASET_JSON:
```json
{"nombre":"x","data":[],"codigo_carga":"df = pd.DataFrame([])"}
```
EJERCICIO: Y
"""
    with pytest.raises(ParseError):
        parse_llm_response(raw)


def test_parse_llm_response_v2_sections() -> None:
    raw = """
## OBJETIVO
Practicar groupby con datos de deportes.

## DATASET
```python
rows = [
  {"equipo": "A", "goles": 2},
  {"equipo": "B", "goles": 3}
]
df = pd.DataFrame(rows)
print(df.head())
```

## CODIGO
```python
import pandas as pd
rows = [{"equipo": "A", "goles": 2}, {"equipo": "B", "goles": 3}]
df = pd.DataFrame(rows)
print(df.groupby("equipo")["goles"].sum())
```

## EXPLICACION
- Creamos el DataFrame con datos sintéticos.
- Aplicamos groupby sobre equipo y sumamos goles.
"""
    parsed = parse_llm_response(raw)
    assert parsed.objetivo.startswith("Practicar groupby")
    assert len(parsed.dataset.data) == 2
    assert "pd.DataFrame" in parsed.codigo
    assert len(parsed.explicacion) >= 2

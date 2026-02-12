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

from backend.validators import EducationalValidator


def test_groupby_validator_passes_valid_code() -> None:
    validator = EducationalValidator()
    code = """
import pandas as pd
# Comentario educativo 1
# Comentario educativo 2
# Comentario educativo 3
# Comentario educativo 4
# Comentario educativo 5
df = pd.DataFrame([{"equipo":"A","goles":2},{"equipo":"B","goles":3}])
resumen = df.groupby("equipo")["goles"].sum().reset_index()
print(resumen)
"""
    result = validator.validate("pandas_groupby", code, ["Paso 1", "Paso 2"])
    assert result.passed is True


def test_filtrado_validator_fails_without_filter_pattern() -> None:
    validator = EducationalValidator()
    code = """
import pandas as pd
# Comentario educativo
df = pd.DataFrame([{"valor": 1}, {"valor": 2}])
resultado = df.groupby("valor").sum()
print(resultado)
"""
    result = validator.validate("pandas_filtrado", code, ["Paso 1", "Paso 2"])
    assert result.passed is False
    assert any("boolean indexing" in err.lower() or ".query" in err for err in result.errors)


def test_lectura_validator_requires_dataframe_and_head() -> None:
    validator = EducationalValidator()
    code = """
import pandas as pd
# Comentario educativo 1
# Comentario educativo 2
# Comentario educativo 3
# Comentario educativo 4
# Comentario educativo 5
rows = [{"item": "A", "valor": 1}, {"item": "B", "valor": 2}]
df = pd.DataFrame(rows)
print(df.head())
"""
    result = validator.validate("pandas_lectura", code, ["Paso 1", "Paso 2"])
    assert result.passed is True


def test_non_python_lines_in_code_validator_detects_prompt_leak() -> None:
    validator = EducationalValidator()
    code = """
import pandas as pd
df = pd.DataFrame(rows)
Errores:
- Falta comentario
print(df.head())
"""
    result = validator.validate_non_python_lines_in_code(code)
    assert result.passed is False
    assert any("fuga del repair prompt" in err.lower() for err in result.errors)


def test_dataset_code_coherence_validator_detects_overridden_rows() -> None:
    validator = EducationalValidator()
    code = """
import pandas as pd
# Comentario educativo 1
# Comentario educativo 2
# Comentario educativo 3
# Comentario educativo 4
# Comentario educativo 5
rows = [{"region": "Norte", "ventas": 10, "mes": 1}]
df = pd.DataFrame(rows)
print(df)
"""
    dataset_data = [{"equipo": "A", "goles": 2, "partidos": 1}]
    result = validator.validate(
        "pandas_groupby",
        code,
        ["Paso 1", "Paso 2"],
        objective="Objetivo",
        raw_text="## DATASET",
        dataset_load_code="df = pd.DataFrame(rows)",
        dataset_data=dataset_data,
    )
    assert result.passed is False
    assert any("code_overrides_dataset" in err for err in result.errors)

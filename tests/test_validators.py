from backend.validators import EducationalValidator


def test_groupby_validator_passes_valid_code() -> None:
    validator = EducationalValidator()
    code = """
import pandas as pd
# Comentario educativo
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


def test_lectura_validator_requires_reader_and_head() -> None:
    validator = EducationalValidator()
    code = """
import pandas as pd
# Comentario educativo
from io import StringIO
df = pd.read_csv(StringIO("a,b\\n1,2"))
print(df.head())
"""
    result = validator.validate("pandas_lectura", code, ["Paso 1", "Paso 2"])
    assert result.passed is True

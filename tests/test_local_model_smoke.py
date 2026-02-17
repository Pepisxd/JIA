from __future__ import annotations

import os

import pytest


@pytest.mark.slow
def test_local_model_smoke_generation() -> None:
    if os.getenv("RUN_LOCAL_MODEL_SMOKE", "false").lower() != "true":
        pytest.skip("RUN_LOCAL_MODEL_SMOKE!=true; smoke test opcional.")

    from backend.local_model import get_local_model

    model = get_local_model(
        model_base=os.getenv("MODEL_BASE", "codellama/CodeLlama-7b-Instruct-hf"),
        adapter_dir=os.getenv("MODEL_PATH", "./models/codellama-edugen-v2"),
        device_map=os.getenv("MODEL_DEVICE_MAP", "auto"),
    )

    prompt = """
<s>[INST] <<SYS>>
Eres un asistente educativo de Python para ciencia de datos.
REGLAS OBLIGATORIAS:
1) Responde EXCLUSIVAMENTE en español.
2) Prohibido usar rutas o leer archivos: /home, /kaggle, /content, ../input, pd.read_csv, read_parquet, read_excel.
3) SIEMPRE incluye dataset sintético pequeño y el código lo construye con pd.DataFrame.
4) EXPLICACION específica, menciona columnas/variables reales.
FORMATO EXACTO:
## OBJETIVO
## DATASET
## CODIGO
## EXPLICACION
<</SYS>>

Genera un ejemplo con Tema: pandas_groupby, Nivel: principiante, Contexto: deportes, Tipo: tutorial.
[/INST]
""".strip()

    out = model.generate(prompt, max_new_tokens=380, temperature=0.2, top_p=0.9)
    out_low = out.lower()

    assert "## objetivo" in out_low
    assert "## dataset" in out_low
    assert "## codigo" in out_low
    assert "## explicacion" in out_low

    forbidden = ["/home", "/kaggle", "/content", "../input", "read_csv(", "read_parquet(", "read_excel("]
    assert all(token not in out_low for token in forbidden)

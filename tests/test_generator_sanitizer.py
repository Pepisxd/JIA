from backend.generator import _sanitize_model_text


def test_sanitizer_removes_special_tokens_without_semantic_change() -> None:
    raw = "<s>## OBJETIVO\nAprender groupby</s>\n\n## EXPLICACION\n- Paso 1</s>"
    sanitized = _sanitize_model_text(raw)
    assert "<s>" not in sanitized
    assert "</s>" not in sanitized
    assert "## OBJETIVO" in sanitized
    assert "Aprender groupby" in sanitized
    assert "## EXPLICACION" in sanitized


def test_sanitizer_does_not_translate_or_insert_comments() -> None:
    raw = "## OBJETIVO\nWhat is the max goals?\n\n## CODIGO\n```python\nprint('x')\n```"
    sanitized = _sanitize_model_text(raw)
    assert "What is the max goals?" in sanitized
    assert "# Comentario educativo" not in sanitized

from backend.executor import CodeExecutor


def test_executor_injects_rows_from_dataset() -> None:
    executor = CodeExecutor()
    dataset = [{"equipo": "A", "goles": 2}, {"equipo": "B", "goles": 3}]
    code = """
import pandas as pd
df = pd.DataFrame(rows)
print(df["goles"].sum())
"""
    result = executor.execute(code, dataset_data=dataset)
    assert result.success is True
    assert result.error is None
    assert result.output.strip() == "5"

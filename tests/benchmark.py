from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from time import perf_counter

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app


def run_benchmark(runs: int, output_path: Path, use_rag: bool = False) -> dict:
    client = TestClient(app)
    temas = ["pandas_groupby", "pandas_filtrado", "pandas_lectura"]
    niveles = ["principiante", "intermedio", "avanzado"]
    contextos = ["deportes", "finanzas", "videojuegos", "ciencia"]
    combos = list(itertools.product(temas, niveles, contextos))

    records = []
    start = perf_counter()
    for i in range(runs):
        tema, nivel, contexto = combos[i % len(combos)]
        payload = {"tema": tema, "nivel": nivel, "contexto": contexto, "tipo": "tutorial", "use_rag": use_rag}
        t0 = perf_counter()
        response = client.post("/generate", json=payload)
        dt_ms = (perf_counter() - t0) * 1000
        body = response.json() if response.status_code == 200 else {}
        records.append(
            {
                "index": i,
                "status_code": response.status_code,
                "tema": tema,
                "nivel": nivel,
                "contexto": contexto,
                "tests_passed": bool(body.get("tests_passed")),
                "educational_passed": bool(body.get("educational_passed")),
                "attempts": int(body.get("attempts", 0)),
                "duration_ms": round(dt_ms, 2),
            }
        )

    total_ms = (perf_counter() - start) * 1000
    ok = [r for r in records if r["status_code"] == 200 and r["tests_passed"]]
    edu_ok = [r for r in records if r["status_code"] == 200 and r["educational_passed"]]
    avg_attempts = sum(r["attempts"] for r in records) / len(records) if records else 0.0
    avg_duration = sum(r["duration_ms"] for r in records) / len(records) if records else 0.0

    result = {
        "runs": runs,
        "use_rag": use_rag,
        "success_rate": round((len(ok) / len(records)) * 100, 2) if records else 0.0,
        "educational_rate": round((len(edu_ok) / len(records)) * 100, 2) if records else 0.0,
        "avg_attempts": round(avg_attempts, 2),
        "avg_duration_ms": round(avg_duration, 2),
        "total_duration_ms": round(total_ms, 2),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local del endpoint /generate")
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--output", type=str, default="tests/benchmark_results.json")
    parser.add_argument("--use-rag", action="store_true")
    args = parser.parse_args()

    result = run_benchmark(runs=args.runs, output_path=Path(args.output), use_rag=args.use_rag)
    print(json.dumps({k: result[k] for k in result if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

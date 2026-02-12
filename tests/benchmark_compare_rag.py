from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.benchmark import run_benchmark


def summarize_delta(without_rag: dict, with_rag: dict) -> dict:
    return {
        "success_rate_delta": round(with_rag["success_rate"] - without_rag["success_rate"], 2),
        "educational_rate_delta": round(with_rag["educational_rate"] - without_rag["educational_rate"], 2),
        "avg_attempts_delta": round(with_rag["avg_attempts"] - without_rag["avg_attempts"], 2),
        "avg_duration_ms_delta": round(with_rag["avg_duration_ms"] - without_rag["avg_duration_ms"], 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara benchmark con y sin RAG.")
    parser.add_argument("--runs", type=int, default=36)
    parser.add_argument("--output", type=str, default="tests/benchmark_compare_rag.json")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    without_rag = run_benchmark(
        runs=args.runs,
        output_path=output.parent / "benchmark_without_rag.json",
        use_rag=False,
    )
    with_rag = run_benchmark(
        runs=args.runs,
        output_path=output.parent / "benchmark_with_rag.json",
        use_rag=True,
    )

    comparison = {
        "runs_per_scenario": args.runs,
        "without_rag": {k: v for k, v in without_rag.items() if k != "records"},
        "with_rag": {k: v for k, v in with_rag.items() if k != "records"},
        "delta": summarize_delta(without_rag, with_rag),
    }
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

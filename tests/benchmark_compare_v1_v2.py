from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

TOPICS = ["pandas_groupby", "pandas_filtrado", "pandas_lectura", "matplotlib_basico", "numpy_basico"]
LEVELS = ["principiante", "intermedio", "avanzado"]
CONTEXTS = ["deportes", "finanzas", "videojuegos", "ciencia"]
TYPES = ["tutorial", "desafio", "mini-proyecto"]
FORBIDDEN_MARKERS = ("/home/", "/kaggle/", "/content/", "../input", "pd.read_csv(", "read_parquet(", "read_excel(")


@dataclass(slots=True)
class Metrics:
    runs: int = 0
    validators_ok: int = 0
    has_dataset: int = 0
    no_forbidden: int = 0
    spanish_ok: int = 0

    def as_percentages(self) -> dict[str, float]:
        total = max(1, self.runs)
        return {
            "runs": self.runs,
            "validators_ok_pct": round(100.0 * self.validators_ok / total, 2),
            "has_dataset_pct": round(100.0 * self.has_dataset / total, 2),
            "no_forbidden_pct": round(100.0 * self.no_forbidden / total, 2),
            "spanish_ok_pct": round(100.0 * self.spanish_ok / total, 2),
        }


def build_prompts(runs: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    prompts: list[dict[str, str]] = []
    for _ in range(runs):
        prompts.append(
            {
                "tema": rng.choice(TOPICS),
                "nivel": rng.choice(LEVELS),
                "contexto": rng.choice(CONTEXTS),
                "tipo": rng.choice(TYPES),
                "use_rag": True,
            }
        )
    return prompts


def looks_spanish(text: str) -> bool:
    lowered = text.lower()
    if any(ch in lowered for ch in "áéíóúñ"):
        return True
    keywords = [" el ", " la ", " de ", " y ", " para ", " datos ", " objetivo ", " explicacion "]
    return sum(1 for k in keywords if k in f" {lowered} ") >= 3


def has_dataset_section(text: str) -> bool:
    return ("## DATASET" in text) or ("DATASET_JSON" in text)


def has_forbidden(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in FORBIDDEN_MARKERS)


def evaluate_response(payload: dict[str, Any], response: dict[str, Any], metrics: Metrics) -> None:
    metrics.runs += 1
    output_text = f"{response.get('codigo', '')}\n{response.get('output', '')}\n{response.get('objetivo', '')}"

    if bool(response.get("educational_passed")) and bool(response.get("tests_passed")):
        metrics.validators_ok += 1
    if has_dataset_section(output_text):
        metrics.has_dataset += 1
    if not has_forbidden(output_text):
        metrics.no_forbidden += 1
    if looks_spanish(f"{payload}\n{output_text}\n{response.get('explicacion', [])}"):
        metrics.spanish_ok += 1


def run_suite(base_url: str, prompts: list[dict[str, Any]], timeout: float) -> Metrics:
    metrics = Metrics()
    url = f"{base_url.rstrip('/')}/generate"
    with httpx.Client(timeout=timeout) as client:
        for prompt in prompts:
            resp = client.post(url, json=prompt)
            resp.raise_for_status()
            evaluate_response(prompt, resp.json(), metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara calidad V1 vs V2 con prompts identicos.")
    parser.add_argument("--v1-url", required=True, help="URL base del backend V1, ej: http://localhost:8001")
    parser.add_argument("--v2-url", required=True, help="URL base del backend V2, ej: http://localhost:8002")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--out", default="tests/benchmark_compare_v1_v2.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = build_prompts(runs=args.runs, seed=args.seed)
    v1_metrics = run_suite(base_url=args.v1_url, prompts=prompts, timeout=args.timeout)
    v2_metrics = run_suite(base_url=args.v2_url, prompts=prompts, timeout=args.timeout)

    result = {
        "runs": args.runs,
        "seed": args.seed,
        "v1": v1_metrics.as_percentages(),
        "v2": v2_metrics.as_percentages(),
        "delta_v2_minus_v1": {
            "validators_ok_pct": round(v2_metrics.as_percentages()["validators_ok_pct"] - v1_metrics.as_percentages()["validators_ok_pct"], 2),
            "has_dataset_pct": round(v2_metrics.as_percentages()["has_dataset_pct"] - v1_metrics.as_percentages()["has_dataset_pct"], 2),
            "no_forbidden_pct": round(v2_metrics.as_percentages()["no_forbidden_pct"] - v1_metrics.as_percentages()["no_forbidden_pct"], 2),
            "spanish_ok_pct": round(v2_metrics.as_percentages()["spanish_ok_pct"] - v1_metrics.as_percentages()["spanish_ok_pct"], 2),
        },
        "prompts": prompts,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

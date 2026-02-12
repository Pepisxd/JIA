from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from backend.rag.retriever import ExampleRetriever

DATASET_ID = "jupyter-agent/jupyter-agent-dataset"


def _pick_primary_package(packages: list[str] | None) -> str:
    if not packages:
        return "unknown"
    normalized = [pkg.strip().lower() for pkg in packages if isinstance(pkg, str) and pkg.strip()]
    return normalized[0] if normalized else "unknown"


def _normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    packages_used = row.get("packages_used") or []
    files_used = row.get("files_used") or []
    edu_score = int(row.get("edu_score") or 0)
    notebook_raw = row.get("original_notebook") or ""

    normalized = {
        "id": row.get("id"),
        "question": question,
        "answer": answer,
        "edu_score": edu_score,
        "packages_used": packages_used,
        "primary_package": _pick_primary_package(packages_used),
        "files_used": files_used,
        "kaggle_dataset_name": row.get("kaggle_dataset_name"),
        "executor_type": row.get("executor_type"),
        "notebook_excerpt": notebook_raw[:1200],
    }
    normalized["rag_text"] = (
        f"question: {normalized['question']}\n"
        f"answer: {normalized['answer']}\n"
        f"packages: {', '.join(packages_used) if packages_used else 'none'}\n"
        f"edu_score: {edu_score}\n"
        f"dataset: {normalized.get('kaggle_dataset_name') or 'unknown'}"
    )
    return normalized


def build_sample(
    split: str,
    target_count: int,
    min_edu_score: int,
    max_per_package: int,
    cache_dir: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    random.seed(seed)
    package_counts: Counter[str] = Counter()
    selected_ids: set[str] = set()
    samples: list[dict[str, Any]] = []

    def pass_over_stream(use_package_limit: bool) -> None:
        nonlocal samples
        stream = load_dataset(DATASET_ID, split=split, streaming=True, cache_dir=str(cache_dir))
        for row in stream:
            if len(samples) >= target_count:
                break

            normalized = _normalize_record(row)
            sample_id = normalized["id"]
            if not sample_id or sample_id in selected_ids:
                continue
            if normalized["edu_score"] < min_edu_score:
                continue
            if not normalized["question"] or not normalized["answer"]:
                continue

            primary_package = normalized["primary_package"]
            if use_package_limit and package_counts[primary_package] >= max_per_package:
                continue

            samples.append(normalized)
            selected_ids.add(sample_id)
            package_counts[primary_package] += 1

    pass_over_stream(use_package_limit=True)
    if len(samples) < target_count:
        pass_over_stream(use_package_limit=False)

    random.shuffle(samples)
    return samples[:target_count], package_counts


def save_outputs(
    samples: list[dict[str, Any]],
    package_counts: Counter[str],
    output_dir: Path,
    split: str,
    target_count: int,
    min_edu_score: int,
    max_per_package: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"sample_{split}_{len(samples)}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "dataset_id": DATASET_ID,
        "split": split,
        "requested_target_count": target_count,
        "final_count": len(samples),
        "min_edu_score": min_edu_score,
        "max_per_package": max_per_package,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "package_distribution": dict(package_counts.most_common()),
    }
    metadata_path = output_dir / f"sample_{split}_{len(samples)}_meta.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return jsonl_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Muestrea y normaliza ejemplos de jupyter-agent para preparar RAG."
    )
    parser.add_argument("--split", default="non_thinking", choices=["non_thinking", "thinking"])
    parser.add_argument("--target-count", type=int, default=500)
    parser.add_argument("--min-edu-score", type=int, default=4)
    parser.add_argument("--max-per-package", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data/jupyter-agent")
    parser.add_argument("--output-dir", default="data/jupyter-agent/processed")
    parser.add_argument("--build-chroma", action="store_true")
    parser.add_argument("--chroma-dir", default="data/chroma")
    parser.add_argument("--collection-name", default="jupyter_agent_examples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)

    samples, package_counts = build_sample(
        split=args.split,
        target_count=args.target_count,
        min_edu_score=args.min_edu_score,
        max_per_package=args.max_per_package,
        cache_dir=cache_dir,
        seed=args.seed,
    )

    jsonl_path, metadata_path = save_outputs(
        samples=samples,
        package_counts=package_counts,
        output_dir=output_dir,
        split=args.split,
        target_count=args.target_count,
        min_edu_score=args.min_edu_score,
        max_per_package=args.max_per_package,
    )
    print(f"Saved sample file: {jsonl_path}")
    print(f"Saved metadata file: {metadata_path}")
    print(f"Final sample count: {len(samples)}")
    if args.build_chroma:
        retriever = ExampleRetriever(
            data_dir=output_dir,
            chroma_dir=Path(args.chroma_dir),
            collection_name=args.collection_name,
            prefer_chroma=True,
        )
        indexed = retriever.index_processed_files()
        print(f"Chroma index updated. Indexed records: {indexed}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from datasets import load_dataset

DATASET_ID = "jupyter-agent/jupyter-agent-dataset"
DEFAULT_OUTPUT = Path("data/finetuning/dataset_analysis.json")


def normalize_edu_score(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    if value <= 1.0:
        return value
    if value <= 5.0:
        return value / 5.0
    if value <= 10.0:
        return value / 10.0
    return None


def extract_code_text(row: dict[str, Any]) -> str:
    code_candidates: list[str] = []
    messages = row.get("messages") or []
    if isinstance(messages, list):
        for msg in messages:
            tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else None
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                function_obj = call.get("function") if isinstance(call, dict) else None
                arguments = function_obj.get("arguments") if isinstance(function_obj, dict) else None
                if isinstance(arguments, dict):
                    code_value = arguments.get("code")
                    if isinstance(code_value, str) and code_value.strip():
                        code_candidates.append(code_value)
    answer = row.get("answer")
    if isinstance(answer, str) and answer.strip():
        code_candidates.append(answer)
    return max(code_candidates, key=len, default="")


def update_edu_bins(score_norm: float | None, bins: Counter[str]) -> None:
    if score_norm is None:
        bins["unknown"] += 1
    elif score_norm < 0.2:
        bins["0.0-0.2"] += 1
    elif score_norm < 0.4:
        bins["0.2-0.4"] += 1
    elif score_norm < 0.6:
        bins["0.4-0.6"] += 1
    elif score_norm < 0.8:
        bins["0.6-0.8"] += 1
    else:
        bins["0.8-1.0"] += 1


def analyze_split(split: str, max_examples: int, cache_dir: Path) -> dict[str, Any]:
    stream = load_dataset(DATASET_ID, split=split, streaming=True, cache_dir=str(cache_dir))

    package_counter: Counter[str] = Counter()
    edu_bins: Counter[str] = Counter()
    code_char_lengths: list[int] = []
    code_line_lengths: list[int] = []
    normalized_scores: list[float] = []

    high_quality_threshold = 0.7
    high_quality_examples = 0
    pandas_examples = 0
    numpy_examples = 0
    matplotlib_examples = 0

    processed = 0
    for row in stream:
        if processed >= max_examples:
            break
        processed += 1

        score_norm = normalize_edu_score(row.get("edu_score"))
        update_edu_bins(score_norm, edu_bins)
        if score_norm is not None:
            normalized_scores.append(score_norm)
            if score_norm > high_quality_threshold:
                high_quality_examples += 1

        packages = row.get("packages_used") or []
        if isinstance(packages, list):
            lowered = []
            for pkg in packages:
                if isinstance(pkg, str):
                    norm = pkg.strip().lower()
                    if norm:
                        lowered.append(norm)
                        package_counter[norm] += 1
            if "pandas" in lowered:
                pandas_examples += 1
            if "numpy" in lowered:
                numpy_examples += 1
            if "matplotlib" in lowered:
                matplotlib_examples += 1

        code_text = extract_code_text(row)
        code_char_lengths.append(len(code_text))
        code_line_lengths.append(len([line for line in code_text.splitlines() if line.strip()]))

    return {
        "split": split,
        "processed_examples": processed,
        "high_quality_threshold": high_quality_threshold,
        "high_quality_examples": high_quality_examples,
        "high_quality_ratio": round(high_quality_examples / processed, 4) if processed else 0.0,
        "edu_score_distribution": dict(edu_bins),
        "edu_score_mean_normalized": round(mean(normalized_scores), 4) if normalized_scores else None,
        "packages_top_20": package_counter.most_common(20),
        "target_packages_presence": {
            "pandas_examples": pandas_examples,
            "numpy_examples": numpy_examples,
            "matplotlib_examples": matplotlib_examples,
        },
        "code_length": {
            "avg_chars": round(mean(code_char_lengths), 2) if code_char_lengths else 0.0,
            "avg_lines": round(mean(code_line_lengths), 2) if code_line_lengths else 0.0,
            "max_chars": max(code_char_lengths, default=0),
            "max_lines": max(code_line_lengths, default=0),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze jupyter-agent dataset for local fine-tuning readiness.")
    parser.add_argument("--split", choices=["thinking", "non_thinking", "both"], default="both")
    parser.add_argument("--max-examples", type=int, default=10000)
    parser.add_argument("--cache-dir", default="data/jupyter-agent")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)

    splits = ["thinking", "non_thinking"] if args.split == "both" else [args.split]
    split_reports = [analyze_split(split=s, max_examples=args.max_examples, cache_dir=cache_dir) for s in splits]

    total_processed = sum(r["processed_examples"] for r in split_reports)
    total_hq = sum(r["high_quality_examples"] for r in split_reports)
    report = {
        "dataset_id": DATASET_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "max_examples_per_split": args.max_examples,
        "total_processed_examples": total_processed,
        "total_high_quality_examples": total_hq,
        "total_high_quality_ratio": round(total_hq / total_processed, 4) if total_processed else 0.0,
        "splits": split_reports,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved analysis report: {output_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

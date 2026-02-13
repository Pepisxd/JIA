from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def signature(row: dict[str, Any]) -> str:
    rid = str(row.get("id", "")).strip()
    if rid:
        return f"id::{rid}"
    raw = f"{row.get('instruction','')}||{row.get('input','')}||{row.get('output','')}"
    return f"hash::{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def extract_code_from_output(output: str) -> str:
    marker = "```python"
    if marker not in output.lower():
        return ""
    lower = output.lower()
    start = lower.find(marker)
    if start < 0:
        return ""
    start = output.find("\n", start)
    if start < 0:
        return ""
    end = output.find("```", start + 1)
    if end < 0:
        return output[start + 1 :].strip()
    return output[start + 1 : end].strip()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * q))
    idx = max(0, min(idx, len(arr) - 1))
    return float(arr[idx])


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        meta = row.get("meta") or {}
        counter[str(meta.get(key, "unknown"))] += 1
    return dict(counter)


def format_ratio(rows: list[dict[str, Any]]) -> tuple[float, dict[str, int]]:
    ok = 0
    missing = {"OBJETIVO": 0, "CODIGO": 0, "EXPLICACION": 0}
    for row in rows:
        output = str(row.get("output", ""))
        has_obj = "OBJETIVO:" in output
        has_code = "CODIGO:" in output
        has_exp = "EXPLICACION:" in output
        if has_obj and has_code and has_exp:
            ok += 1
        else:
            if not has_obj:
                missing["OBJETIVO"] += 1
            if not has_code:
                missing["CODIGO"] += 1
            if not has_exp:
                missing["EXPLICACION"] += 1
    ratio = (ok / len(rows)) if rows else 0.0
    return ratio, missing


def package_target_coverage(rows: list[dict[str, Any]]) -> dict[str, float]:
    pandas_n = 0
    numpy_n = 0
    matplotlib_n = 0
    for row in rows:
        code = extract_code_from_output(str(row.get("output", ""))).lower()
        meta = row.get("meta") or {}
        packages = [str(x).lower() for x in (meta.get("packages_used") or [])]
        text = f"{code}\n{' '.join(packages)}"
        if "pandas" in text or "import pandas" in text:
            pandas_n += 1
        if "numpy" in text or "import numpy" in text or " np." in text:
            numpy_n += 1
        if "matplotlib" in text or "import matplotlib" in text or "plt." in text:
            matplotlib_n += 1
    total = max(1, len(rows))
    return {
        "pandas_ratio": round(pandas_n / total, 4),
        "numpy_ratio": round(numpy_n / total, 4),
        "matplotlib_ratio": round(matplotlib_n / total, 4),
    }


def length_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    chars: list[float] = []
    lines: list[float] = []
    for row in rows:
        code = extract_code_from_output(str(row.get("output", "")))
        chars.append(float(len(code)))
        lines.append(float(len([l for l in code.splitlines() if l.strip()])))
    if not chars:
        return {
            "chars_min": 0.0,
            "chars_max": 0.0,
            "chars_p50": 0.0,
            "chars_p90": 0.0,
            "chars_p95": 0.0,
            "lines_min": 0.0,
            "lines_max": 0.0,
            "lines_p50": 0.0,
            "lines_p90": 0.0,
            "lines_p95": 0.0,
        }
    return {
        "chars_min": min(chars),
        "chars_max": max(chars),
        "chars_p50": float(median(chars)),
        "chars_p90": percentile(chars, 0.90),
        "chars_p95": percentile(chars, 0.95),
        "lines_min": min(lines),
        "lines_max": max(lines),
        "lines_p50": float(median(lines)),
        "lines_p90": percentile(lines, 0.90),
        "lines_p95": percentile(lines, 0.95),
    }


def duplicate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sigs = [signature(r) for r in rows]
    count = Counter(sigs)
    dup = sum(v - 1 for v in count.values() if v > 1)
    return {
        "duplicate_count": dup,
        "duplicate_ratio": round((dup / len(rows)), 6) if rows else 0.0,
    }


def overlap_count(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> int:
    train_sigs = {signature(r) for r in train_rows}
    val_sigs = {signature(r) for r in val_rows}
    return len(train_sigs.intersection(val_sigs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal validation for fine-tuning train/val datasets.")
    parser.add_argument("--train", default="data/finetuning/train.jsonl")
    parser.add_argument("--val", default="data/finetuning/val.jsonl")
    parser.add_argument("--out", default="data/finetuning/validate_report.json")
    parser.add_argument("--expected-train-count", type=int, default=1620)
    parser.add_argument("--expected-val-count", type=int, default=180)
    parser.add_argument("--max-missing-section-ratio", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    val_path = Path(args.val)
    out_path = Path(args.out)

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)

    train_format_ratio, train_missing = format_ratio(train_rows)
    val_format_ratio, val_missing = format_ratio(val_rows)

    train_dup = duplicate_stats(train_rows)
    val_dup = duplicate_stats(val_rows)
    overlap = overlap_count(train_rows, val_rows)

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "train": str(train_path),
            "val": str(val_path),
            "expected_train_count": args.expected_train_count,
            "expected_val_count": args.expected_val_count,
            "max_missing_section_ratio": args.max_missing_section_ratio,
        },
        "counts": {
            "train_count": len(train_rows),
            "val_count": len(val_rows),
        },
        "distribution": {
            "train": {
                "topics": distribution(train_rows, "topic"),
                "levels": distribution(train_rows, "level"),
                "contexts": distribution(train_rows, "context"),
            },
            "val": {
                "topics": distribution(val_rows, "topic"),
                "levels": distribution(val_rows, "level"),
                "contexts": distribution(val_rows, "context"),
            },
        },
        "format": {
            "train_format_ok_ratio": round(train_format_ratio, 6),
            "val_format_ok_ratio": round(val_format_ratio, 6),
            "train_missing_sections": train_missing,
            "val_missing_sections": val_missing,
        },
        "duplicates": {
            "train": train_dup,
            "val": val_dup,
            "train_val_overlap": overlap,
        },
        "length_stats": {
            "train": length_stats(train_rows),
            "val": length_stats(val_rows),
        },
        "target_package_coverage": {
            "train": package_target_coverage(train_rows),
            "val": package_target_coverage(val_rows),
        },
        "status": "pass",
        "errors": [],
    }

    errors: list[str] = []
    if len(train_rows) == 0 or len(val_rows) == 0:
        errors.append("train/val vacios.")
    if len(train_rows) != args.expected_train_count:
        errors.append(f"train_count invalido: {len(train_rows)} != {args.expected_train_count}")
    if len(val_rows) != args.expected_val_count:
        errors.append(f"val_count invalido: {len(val_rows)} != {args.expected_val_count}")
    if overlap > 0:
        errors.append(f"train_val_overlap invalido: {overlap} > 0")

    # Fail if missing any section in >1% of examples
    train_missing_ratio_obj = (train_missing["OBJETIVO"] / len(train_rows)) if train_rows else 1.0
    train_missing_ratio_cod = (train_missing["CODIGO"] / len(train_rows)) if train_rows else 1.0
    train_missing_ratio_exp = (train_missing["EXPLICACION"] / len(train_rows)) if train_rows else 1.0
    if train_missing_ratio_obj > args.max_missing_section_ratio:
        errors.append(f"Falta OBJETIVO en > {args.max_missing_section_ratio:.2%} del train.")
    if train_missing_ratio_cod > args.max_missing_section_ratio:
        errors.append(f"Falta CODIGO en > {args.max_missing_section_ratio:.2%} del train.")
    if train_missing_ratio_exp > args.max_missing_section_ratio:
        errors.append(f"Falta EXPLICACION en > {args.max_missing_section_ratio:.2%} del train.")

    val_missing_ratio_obj = (val_missing["OBJETIVO"] / len(val_rows)) if val_rows else 1.0
    val_missing_ratio_cod = (val_missing["CODIGO"] / len(val_rows)) if val_rows else 1.0
    val_missing_ratio_exp = (val_missing["EXPLICACION"] / len(val_rows)) if val_rows else 1.0
    if val_missing_ratio_obj > args.max_missing_section_ratio:
        errors.append(f"Falta OBJETIVO en > {args.max_missing_section_ratio:.2%} del val.")
    if val_missing_ratio_cod > args.max_missing_section_ratio:
        errors.append(f"Falta CODIGO en > {args.max_missing_section_ratio:.2%} del val.")
    if val_missing_ratio_exp > args.max_missing_section_ratio:
        errors.append(f"Falta EXPLICACION en > {args.max_missing_section_ratio:.2%} del val.")

    report["errors"] = errors
    if errors:
        report["status"] = "fail"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== VALIDATION SUMMARY ===")
    print(f"status: {report['status']}")
    print(f"train_count: {len(train_rows)}")
    print(f"val_count: {len(val_rows)}")
    print(f"format_ok_ratio_train: {report['format']['train_format_ok_ratio']}")
    print(f"format_ok_ratio_val: {report['format']['val_format_ok_ratio']}")
    print(f"duplicate_ratio_train: {report['duplicates']['train']['duplicate_ratio']}")
    print(f"duplicate_ratio_val: {report['duplicates']['val']['duplicate_ratio']}")
    print(f"train_val_overlap: {report['duplicates']['train_val_overlap']}")
    print(f"report_path: {out_path}")
    if errors:
        print("errors:")
        for err in errors:
            print(f"- {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()

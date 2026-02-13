from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASET_ID = "jupyter-agent/jupyter-agent-dataset"
TARGET_PACKAGES = {"pandas", "numpy", "matplotlib"}


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
            if not isinstance(msg, dict):
                continue
            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                fn_obj = call.get("function")
                if not isinstance(fn_obj, dict):
                    continue
                arguments = fn_obj.get("arguments")
                if isinstance(arguments, dict):
                    code = arguments.get("code")
                    if isinstance(code, str) and code.strip():
                        code_candidates.append(code)
    answer = row.get("answer")
    if isinstance(answer, str) and answer.strip():
        code_candidates.append(answer)
    return max(code_candidates, key=len, default="")


def parse_packages(row: dict[str, Any]) -> list[str]:
    packages = row.get("packages_used") or []
    out: list[str] = []
    if isinstance(packages, list):
        for pkg in packages:
            if isinstance(pkg, str):
                p = pkg.strip().lower()
                if p:
                    out.append(p)
    return out


def infer_topic(code: str, question: str, packages: list[str]) -> str:
    text = f"{question}\n{code}".lower()
    if "groupby" in text:
        return "pandas_groupby"
    if ".query(" in text or re.search(r"df\s*\[\s*df\s*\[", text):
        return "pandas_filtrado"
    if "read_csv(" in text or "read_excel(" in text:
        return "pandas_lectura"
    if "matplotlib" in packages or "plt." in text:
        return "matplotlib_basico"
    if "numpy" in packages or "np." in text:
        return "numpy_basico"
    if "pandas" in packages:
        return "pandas_groupby"
    return "eda_basico"


def infer_level(code: str, edu_score_norm: float | None) -> str:
    lines = [line for line in code.splitlines() if line.strip()]
    n_lines = len(lines)
    if edu_score_norm is not None and edu_score_norm >= 0.9 and n_lines > 35:
        return "avanzado"
    if n_lines > 22:
        return "intermedio"
    return "principiante"


def infer_context(question: str, dataset_name: str | None, packages: list[str]) -> str:
    text = f"{question} {dataset_name or ''}".lower()
    if any(k in text for k in ["futbol", "football", "nba", "team", "player", "deporte", "match"]):
        return "deportes"
    if any(k in text for k in ["stock", "revenue", "finance", "finanzas", "sales", "market"]):
        return "finanzas"
    if any(k in text for k in ["game", "gaming", "videogame", "steam", "player level"]):
        return "videojuegos"
    if any(k in text for k in ["science", "experiment", "biology", "physics", "chemistry", "climate"]):
        return "ciencia"
    if "tensorflow" in packages:
        return "ciencia"
    if "pandas" in packages:
        return "finanzas"
    return "deportes"


def build_instruction(topic: str, level: str, context: str) -> tuple[str, str]:
    instruction = (
        "Genera un ejercicio educativo de Python para ciencia de datos con el formato: "
        "OBJETIVO, DATASET_JSON, CODIGO, EXPLICACION y EJERCICIO."
    )
    input_text = f"Tema: {topic}, Nivel: {level}, Contexto: {context}, Tipo: tutorial"
    return instruction, input_text


def build_output(code: str, question: str, answer: str, dataset_name: str | None) -> str:
    dataset_name = dataset_name or "dataset_educativo"
    objective = question.strip() or "Aprender una tecnica de analisis de datos en Python."
    explanation = [
        "Paso 1: Revisa el objetivo y el dataset de entrada.",
        "Paso 2: Ejecuta el codigo y valida el resultado.",
        "Paso 3: Modifica una parte del analisis para practicar.",
    ]
    explanation_md = "\n".join(f"- {item}" for item in explanation)
    return (
        f"OBJETIVO: {objective}\n\n"
        "DATASET_JSON:\n"
        "```json\n"
        "{\n"
        f'  "nombre": "{dataset_name}",\n'
        '  "data": [],\n'
        '  "codigo_carga": "df = pd.DataFrame([])"\n'
        "}\n"
        "```\n\n"
        "CODIGO:\n"
        "```python\n"
        f"{code.strip()}\n"
        "```\n\n"
        "EXPLICACION:\n"
        f"{explanation_md}\n\n"
        f"EJERCICIO: {answer.strip() or 'Extiende el ejemplo agregando una nueva metrica.'}\n"
    )


@dataclass(slots=True)
class Candidate:
    record: dict[str, Any]
    packages: list[str]
    primary_package: str


def collect_candidates(
    splits: list[str],
    max_source_examples: int,
    min_edu_score: float,
    cache_dir: Path,
    seed: int,
) -> tuple[list[Candidate], dict[str, int]]:
    random.seed(seed)
    candidates: list[Candidate] = []
    seen_ids: set[str] = set()
    stats = defaultdict(int)

    for split in splits:
        stream = load_dataset(DATASET_ID, split=split, streaming=True, cache_dir=str(cache_dir))
        processed_split = 0
        for row in stream:
            if processed_split >= max_source_examples:
                break
            processed_split += 1
            stats["processed_total"] += 1

            row_id = row.get("id")
            if not row_id or row_id in seen_ids:
                stats["skipped_id"] += 1
                continue

            score_norm = normalize_edu_score(row.get("edu_score"))
            if score_norm is None or score_norm <= min_edu_score:
                stats["skipped_score"] += 1
                continue

            packages = parse_packages(row)
            if not TARGET_PACKAGES.intersection(packages):
                stats["skipped_packages"] += 1
                continue

            code = extract_code_text(row)
            if not code.strip():
                stats["skipped_no_code"] += 1
                continue
            if len(code.splitlines()) < 6:
                stats["skipped_too_short"] += 1
                continue

            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            dataset_name = str(row.get("kaggle_dataset_name") or "").strip()
            topic = infer_topic(code, question, packages)
            level = infer_level(code, score_norm)
            context = infer_context(question, dataset_name, packages)
            instruction, input_text = build_instruction(topic, level, context)
            output = build_output(code, question, answer, dataset_name)

            formatted = {
                "id": row_id,
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "meta": {
                    "topic": topic,
                    "level": level,
                    "context": context,
                    "edu_score_norm": score_norm,
                    "packages_used": packages,
                    "split": split,
                },
            }
            primary_package = packages[0] if packages else "unknown"
            candidates.append(Candidate(record=formatted, packages=packages, primary_package=primary_package))
            seen_ids.add(row_id)
            stats["accepted"] += 1
        stats[f"processed_{split}"] = processed_split

    return candidates, dict(stats)


def balance_candidates(candidates: list[Candidate], target_count: int, seed: int) -> list[dict[str, Any]]:
    random.seed(seed)
    by_package: dict[str, list[Candidate]] = defaultdict(list)
    for item in candidates:
        by_package[item.primary_package].append(item)

    if not by_package:
        return []

    for pkg_items in by_package.values():
        random.shuffle(pkg_items)

    # Round-robin package balancing, then fill remaining with leftovers.
    package_keys = sorted(by_package.keys())
    selected: list[Candidate] = []
    while len(selected) < target_count:
        progressed = False
        for pkg in package_keys:
            pkg_items = by_package[pkg]
            if pkg_items:
                selected.append(pkg_items.pop())
                progressed = True
                if len(selected) >= target_count:
                    break
        if not progressed:
            break

    if len(selected) < target_count:
        leftovers: list[Candidate] = []
        for pkg in package_keys:
            leftovers.extend(by_package[pkg])
        random.shuffle(leftovers)
        selected.extend(leftovers[: target_count - len(selected)])

    return [item.record for item in selected[:target_count]]


def balance_candidates_with_caps(
    candidates: list[Candidate],
    target_count: int,
    seed: int,
    max_per_topic: int,
    max_per_context_per_topic: int,
    min_per_topic: int,
) -> list[dict[str, Any]]:
    random.seed(seed)
    shuffled = candidates[:]
    random.shuffle(shuffled)

    topic_counts: Counter[str] = Counter()
    context_topic_counts: Counter[tuple[str, str]] = Counter()
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    by_topic: dict[str, list[Candidate]] = defaultdict(list)
    for item in shuffled:
        topic = str((item.record.get("meta") or {}).get("topic", "unknown"))
        by_topic[topic].append(item)

    # Phase 1: enforce minimum per topic when possible.
    for topic in sorted(by_topic.keys()):
        if min_per_topic <= 0:
            break
        for item in by_topic[topic]:
            if len(selected) >= target_count or topic_counts[topic] >= min_per_topic:
                break
            meta = item.record.get("meta") or {}
            context = str(meta.get("context", "unknown"))
            rec_id = str(item.record.get("id"))
            if rec_id in selected_ids:
                continue
            if topic_counts[topic] >= max_per_topic:
                continue
            if context_topic_counts[(topic, context)] >= max_per_context_per_topic:
                continue
            selected.append(item.record)
            selected_ids.add(rec_id)
            topic_counts[topic] += 1
            context_topic_counts[(topic, context)] += 1

    # Phase 2: fill the remaining slots with caps.
    for item in shuffled:
        if len(selected) >= target_count:
            break
        meta = item.record.get("meta") or {}
        topic = str(meta.get("topic", "unknown"))
        context = str(meta.get("context", "unknown"))
        rec_id = str(item.record.get("id"))
        if rec_id in selected_ids:
            continue

        if topic_counts[topic] >= max_per_topic:
            continue
        if context_topic_counts[(topic, context)] >= max_per_context_per_topic:
            continue

        selected.append(item.record)
        selected_ids.add(rec_id)
        topic_counts[topic] += 1
        context_topic_counts[(topic, context)] += 1

    return selected


def split_train_val(records: list[dict[str, Any]], val_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    random.seed(seed)
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        topic = str((row.get("meta") or {}).get("topic", "unknown"))
        by_topic[topic].append(row)

    for rows in by_topic.values():
        random.shuffle(rows)

    total_records = len(records)
    desired_val = min(val_size, max(1, total_records // 10))
    val: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []

    # First pass: proportional 10% val per topic.
    for topic in sorted(by_topic.keys()):
        topic_rows = by_topic[topic]
        topic_val_n = max(1, round(len(topic_rows) * 0.1))
        topic_val_n = min(topic_val_n, len(topic_rows))
        val.extend(topic_rows[:topic_val_n])
        train.extend(topic_rows[topic_val_n:])

    # Adjust to desired exact val size while preserving stratification as much as possible.
    if len(val) > desired_val:
        random.shuffle(val)
        overflow = val[desired_val:]
        val = val[:desired_val]
        train.extend(overflow)
    elif len(val) < desired_val:
        random.shuffle(train)
        needed = desired_val - len(val)
        val.extend(train[:needed])
        train = train[needed:]

    random.shuffle(train)
    random.shuffle(val)
    return train, val


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def analyze_balance(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    topic_counter: Counter[str] = Counter()
    level_counter: Counter[str] = Counter()
    context_counter: Counter[str] = Counter()
    for row in records:
        meta = row.get("meta") or {}
        topic_counter[str(meta.get("topic", "unknown"))] += 1
        level_counter[str(meta.get("level", "unknown"))] += 1
        context_counter[str(meta.get("context", "unknown"))] += 1
    return {
        "topics": dict(topic_counter),
        "levels": dict(level_counter),
        "contexts": dict(context_counter),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare balanced instruction-tuning dataset from jupyter-agent.")
    parser.add_argument("--split", choices=["thinking", "non_thinking", "both"], default="both")
    parser.add_argument("--max-source-examples", type=int, default=30000)
    parser.add_argument("--target-count", type=int, default=5500)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--min-edu-score", type=float, default=0.7)
    parser.add_argument("--min-per-topic", type=int, default=200)
    parser.add_argument("--max-per-topic", type=int, default=500)
    parser.add_argument("--max-per-context-per-topic", type=int, default=200)
    parser.add_argument("--cache-dir", default="data/jupyter-agent")
    parser.add_argument("--output-dir", default="data/finetuning")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = ["thinking", "non_thinking"] if args.split == "both" else [args.split]
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)

    candidates, collection_stats = collect_candidates(
        splits=splits,
        max_source_examples=args.max_source_examples,
        min_edu_score=args.min_edu_score,
        cache_dir=cache_dir,
        seed=args.seed,
    )

    balanced = balance_candidates_with_caps(
        candidates,
        target_count=args.target_count,
        seed=args.seed,
        min_per_topic=args.min_per_topic,
        max_per_topic=args.max_per_topic,
        max_per_context_per_topic=args.max_per_context_per_topic,
    )
    train_records, val_records = split_train_val(balanced, val_size=args.val_size, seed=args.seed)
    full_balance = analyze_balance(balanced)
    train_balance = analyze_balance(train_records)
    val_balance = analyze_balance(val_records)

    topic_min_count_train = min(train_balance["topics"].values()) if train_balance["topics"] else 0
    has_advanced_examples = train_balance["levels"].get("avanzado", 0) > 0
    format_checks = {
        "all_have_objetivo": all("OBJETIVO:" in r.get("output", "") for r in train_records[:200]),
        "all_have_codigo": all("CODIGO:" in r.get("output", "") for r in train_records[:200]),
        "all_have_explicacion": all("EXPLICACION:" in r.get("output", "") for r in train_records[:200]),
    }

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    save_jsonl(train_path, train_records)
    save_jsonl(val_path, val_records)

    report = {
        "dataset_id": DATASET_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "params": {
            "splits": splits,
            "max_source_examples_per_split": args.max_source_examples,
            "target_count": args.target_count,
            "val_size_requested": args.val_size,
            "min_edu_score": args.min_edu_score,
            "min_per_topic": args.min_per_topic,
            "max_per_topic": args.max_per_topic,
            "max_per_context_per_topic": args.max_per_context_per_topic,
            "seed": args.seed,
        },
        "collection_stats": collection_stats,
        "counts": {
            "candidates": len(candidates),
            "balanced_total": len(balanced),
            "train": len(train_records),
            "val": len(val_records),
        },
        "balance": {
            "all_selected": full_balance,
            "train": train_balance,
            "val": val_balance,
        },
        "pretrain_checks": {
            "each_topic_train_ge_200": topic_min_count_train >= 200,
            "topic_train_min_count": topic_min_count_train,
            "has_advanced_examples_train": has_advanced_examples,
            "output_format_checks_train_sample": format_checks,
        },
        "files": {
            "train_jsonl": str(train_path),
            "val_jsonl": str(val_path),
        },
    }

    report_path = output_dir / "prepare_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved train: {train_path}")
    print(f"Saved val: {val_path}")
    print(f"Saved report: {report_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

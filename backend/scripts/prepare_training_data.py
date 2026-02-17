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
FORBIDDEN_MARKERS = (
    "/home/",
    "/kaggle/",
    "/content/",
    "../input",
    "pd.read_csv(",
    "read_parquet(",
    "read_excel(",
)


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
    if "read_csv(" in text or "read_excel(" in text or "read_parquet(" in text:
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


def has_forbidden_markers(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in FORBIDDEN_MARKERS)


def synthetic_rows(topic: str, context: str, seed_text: str) -> list[dict[str, Any]]:
    seed = sum(ord(ch) for ch in seed_text) % (2**31 - 1)
    rng = random.Random(seed)
    n_rows = rng.randint(8, 15)

    if context == "deportes":
        equipos = ["Tigres", "Pumas", "Leones", "Halcones", "Rojos"]
        base = []
        for i in range(n_rows):
            base.append(
                {
                    "equipo": equipos[i % len(equipos)],
                    "goles": rng.randint(0, 5),
                    "partidos": rng.randint(1, 4),
                    "temporada": 2025,
                }
            )
        return base
    if context == "videojuegos":
        juegos = ["Apex", "Valor", "Craft", "RallyX", "Quest"]
        return [
            {
                "juego": juegos[i % len(juegos)],
                "jugadores": rng.randint(100, 500),
                "horas": rng.randint(2, 30),
                "nivel": rng.randint(1, 20),
            }
            for i in range(n_rows)
        ]
    if context == "ciencia":
        return [
            {
                "experimento": f"exp_{(i % 4) + 1}",
                "temperatura": round(rng.uniform(18.0, 40.0), 1),
                "presion": round(rng.uniform(0.9, 1.3), 2),
                "resultado": round(rng.uniform(10.0, 95.0), 2),
            }
            for i in range(n_rows)
        ]

    # finanzas (default)
    regiones = ["Norte", "Sur", "Este", "Oeste"]
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "region": regiones[i % len(regiones)],
                "ventas": rng.randint(1200, 9800),
                "costos": rng.randint(800, 6500),
                "mes": (i % 6) + 1,
            }
        )
    return rows


def code_from_topic(topic: str, context: str, dataset_var: str) -> str:
    if topic == "pandas_groupby":
        group_col = "equipo" if context == "deportes" else "region"
        value_col = "goles" if context == "deportes" else "ventas"
        return (
            f"import pandas as pd\n\n"
            f"df = pd.DataFrame({dataset_var})\n"
            f"resumen = df.groupby('{group_col}', as_index=False)['{value_col}'].sum()\n"
            "resumen = resumen.sort_values(by='"
            f"{value_col}', ascending=False)\n"
            "print(resumen)\n"
        )
    if topic == "pandas_filtrado":
        if context == "deportes":
            return (
                f"import pandas as pd\n\n"
                f"df = pd.DataFrame({dataset_var})\n"
                "filtro = (df['goles'] >= 2) & (df['partidos'] >= 2)\n"
                "resultado = df[filtro].copy()\n"
                "resultado['promedio_goles'] = (resultado['goles'] / resultado['partidos']).round(2)\n"
                "print(resultado)\n"
            )
        return (
            f"import pandas as pd\n\n"
            f"df = pd.DataFrame({dataset_var})\n"
            "resultado = df.query('ventas > 3000 and costos < 5000').copy()\n"
            "resultado['margen'] = resultado['ventas'] - resultado['costos']\n"
            "print(resultado)\n"
        )
    if topic == "pandas_lectura":
        return (
            f"import pandas as pd\n\n"
            f"df = pd.DataFrame({dataset_var})\n"
            "print(df.head())\n"
            "print('\\nResumen de columnas:')\n"
            "print(df.describe(include='all'))\n"
        )
    if topic == "matplotlib_basico":
        x_col = "equipo" if context == "deportes" else "region"
        y_col = "goles" if context == "deportes" else "ventas"
        return (
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            f"df = pd.DataFrame({dataset_var})\n"
            f"serie = df.groupby('{x_col}', as_index=False)['{y_col}'].sum()\n"
            "plt.figure(figsize=(7, 4))\n"
            f"plt.bar(serie['{x_col}'], serie['{y_col}'])\n"
            f"plt.title('Total de {y_col} por {x_col}')\n"
            f"plt.xlabel('{x_col}')\n"
            f"plt.ylabel('{y_col}')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    if topic == "numpy_basico":
        return (
            "import numpy as np\n"
            "import pandas as pd\n\n"
            f"df = pd.DataFrame({dataset_var})\n"
            "valores = df.select_dtypes(include='number').to_numpy()\n"
            "print('Media global:', np.mean(valores).round(2))\n"
            "print('Desviacion estandar global:', np.std(valores).round(2))\n"
        )
    return (
        f"import pandas as pd\n\n"
        f"df = pd.DataFrame({dataset_var})\n"
        "print(df.head())\n"
    )


def explanation_from_topic(topic: str, context: str) -> list[str]:
    if topic == "pandas_groupby":
        return [
            "Primero construimos el DataFrame desde una lista de diccionarios para mantener el flujo 100% offline.",
            "Despues agrupamos por la columna categorica del contexto y aplicamos una suma sobre la metrica principal.",
            "Finalmente ordenamos el resultado para interpretar rapidamente que categoria aporta mas valor.",
        ]
    if topic == "pandas_filtrado":
        return [
            "Creamos el DataFrame de forma local para evitar dependencias de archivos externos.",
            "Aplicamos un filtro booleano para quedarnos solo con filas que cumplen condiciones relevantes del contexto.",
            "Generamos una columna derivada para reforzar el analisis y practicar transformaciones de pandas.",
        ]
    if topic == "pandas_lectura":
        return [
            "En lugar de leer CSV/Excel, simulamos la carga creando el DataFrame directamente en memoria.",
            "Usamos head() para inspeccionar rapidamente la estructura inicial de los datos.",
            "Con describe(include='all') obtenemos un resumen estadistico util para entender calidad y distribucion.",
        ]
    if topic == "matplotlib_basico":
        return [
            "Partimos de datos sinteticos y construimos el DataFrame con pandas.",
            "Agregamos la metrica por categoria para obtener una vista resumida antes de graficar.",
            "Visualizamos con un grafico de barras y etiquetas claras para que la interpretacion sea inmediata.",
        ]
    if topic == "numpy_basico":
        return [
            "Convertimos las columnas numericas del DataFrame a una matriz numpy para operaciones vectorizadas.",
            "Calculamos media y desviacion estandar global para practicar estadistica descriptiva basica.",
            "Este flujo combina pandas y numpy, que es una habilidad central en analisis de datos.",
        ]
    return [
        "Construimos un DataFrame sintetico para practicar analisis de datos sin depender de archivos externos.",
        "Aplicamos una transformacion simple y verificamos el resultado en consola.",
        "La idea es que puedas modificar columnas y condiciones para extender el ejercicio.",
    ]


def build_output_v2(topic: str, level: str, context: str, question: str, row_id: str) -> str:
    objective = question or f"Aprender {topic} en un escenario de {context}."
    rows = synthetic_rows(topic=topic, context=context, seed_text=row_id)
    rows_json = json.dumps(rows, ensure_ascii=False, indent=2)
    code = code_from_topic(topic=topic, context=context, dataset_var="rows")
    explanation = explanation_from_topic(topic=topic, context=context)
    explanation_md = "\n".join(f"- {item}" for item in explanation)

    # Estructura exacta requerida para V2.
    return (
        "## OBJETIVO\n"
        f"{objective}\n\n"
        "## DATASET\n"
        "```python\n"
        f"rows = {rows_json}\n"
        "df = pd.DataFrame(rows)\n"
        "print(df.head())\n"
        "```\n\n"
        "## CODIGO\n"
        "```python\n"
        f"{code.strip()}\n"
        "```\n\n"
        "## EXPLICACION\n"
        f"{explanation_md}\n"
    )


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
    version: str,
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
            if len([line for line in code.splitlines() if line.strip()]) < 5:
                stats["skipped_too_short"] += 1
                continue
            raw_text = f"{row.get('question', '')}\n{row.get('answer', '')}\n{code}"
            needs_sanitization = has_forbidden_markers(raw_text)
            if needs_sanitization:
                stats["sanitized_count"] += 1

            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            dataset_name = str(row.get("kaggle_dataset_name") or "").strip()
            topic = infer_topic(code, question, packages)
            level = infer_level(code, score_norm)
            context = infer_context(question, dataset_name, packages)
            instruction, input_text = build_instruction(topic, level, context)
            if version == "v2":
                output = build_output_v2(
                    topic=topic,
                    level=level,
                    context=context,
                    question=question,
                    row_id=str(row_id),
                )
            else:
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
                    "needs_sanitization": needs_sanitization,
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
    parser.add_argument("--max-source-examples", type=int, default=12000)
    parser.add_argument("--target-count", type=int, default=5500)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--min-edu-score", type=float, default=0.7)
    parser.add_argument("--min-per-topic", type=int, default=200)
    parser.add_argument("--max-per-topic", type=int, default=500)
    parser.add_argument("--max-per-context-per-topic", type=int, default=200)
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
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
        version=args.version,
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
    if args.version == "v2":
        format_checks = {
            "all_have_objetivo": all("## OBJETIVO" in r.get("output", "") for r in train_records[:200]),
            "all_have_dataset": all("## DATASET" in r.get("output", "") for r in train_records[:200]),
            "all_have_codigo": all("## CODIGO" in r.get("output", "") for r in train_records[:200]),
            "all_have_explicacion": all("## EXPLICACION" in r.get("output", "") for r in train_records[:200]),
            "all_offline_pd_dataframe": all("pd.DataFrame(" in r.get("output", "") for r in train_records[:200]),
        }
        train_path = output_dir / "train_v2.jsonl"
        val_path = output_dir / "val_v2.jsonl"
        report_path = output_dir / "prepare_report_v2.json"
    else:
        format_checks = {
            "all_have_objetivo": all("OBJETIVO:" in r.get("output", "") for r in train_records[:200]),
            "all_have_codigo": all("CODIGO:" in r.get("output", "") for r in train_records[:200]),
            "all_have_explicacion": all("EXPLICACION:" in r.get("output", "") for r in train_records[:200]),
        }
        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"
        report_path = output_dir / "prepare_report.json"
    save_jsonl(train_path, train_records)
    save_jsonl(val_path, val_records)

    report = {
        "dataset_id": DATASET_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "params": {
            "version": args.version,
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
        "sanitized_count": int(collection_stats.get("sanitized_count", 0)),
        "skipped_total": sum(
            int(v)
            for k, v in collection_stats.items()
            if k.startswith("skipped_")
        ),
        "skip_reasons": {
            k: int(v) for k, v in collection_stats.items() if k.startswith("skipped_")
        },
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

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved train: {train_path}")
    print(f"Saved val: {val_path}")
    print(f"Saved report: {report_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

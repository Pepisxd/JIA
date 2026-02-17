from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from backend.models import GenerateRequest, GenerateResponse


class MetricsCollector:
    def __init__(self, base_dir: Path | str = "logs/metrics") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.base_dir / "generations.jsonl"

    def record(
        self,
        *,
        request: GenerateRequest,
        response: GenerateResponse | None,
        duration_ms: float,
        status: str,
        error: str | None = None,
    ) -> None:
        event = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "tema": request.tema,
            "nivel": request.nivel,
            "contexto": request.contexto,
            "tipo": request.tipo,
            "status": status,
            "duration_ms": duration_ms,
            "attempts": response.attempts if response else 0,
            "tests_passed": response.tests_passed if response else False,
            "educational_passed": response.educational_passed if response else False,
            "validation_errors": response.validation_errors if response else [],
            "error": error or (response.error if response else None),
            "used_fallback": response.used_fallback if response else False,
            "model_backend": response.model_backend if response else "unknown",
            "post_processed": response.post_processed if response else False,
        }
        with self.events_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _read_events(self) -> list[dict[str, Any]]:
        if not self.events_file.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.events_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

    def summary(self) -> dict[str, Any]:
        events = self._read_events()
        total = len(events)
        if total == 0:
            return {
                "total_generations": 0,
                "success_rate": 0.0,
                "educational_pass_rate": 0.0,
                "avg_attempts": 0.0,
                "avg_duration_ms": 0.0,
                "by_tema": {},
                "recent_failures": [],
            }

        successes = [e for e in events if e.get("tests_passed")]
        educational_ok = [e for e in events if e.get("educational_passed")]
        attempts = [float(e.get("attempts", 0)) for e in events]
        durations = [float(e.get("duration_ms", 0.0)) for e in events]

        by_tema: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"total": 0, "success": 0, "educational": 0}
        )
        for event in events:
            tema = str(event.get("tema", "unknown"))
            by_tema[tema]["total"] += 1
            by_tema[tema]["success"] += int(bool(event.get("tests_passed")))
            by_tema[tema]["educational"] += int(bool(event.get("educational_passed")))

        dashboard = {}
        for tema, row in by_tema.items():
            total_t = int(row["total"])
            dashboard[tema] = {
                "total": total_t,
                "success_rate": round((row["success"] / total_t) * 100, 2) if total_t else 0.0,
                "educational_rate": round((row["educational"] / total_t) * 100, 2) if total_t else 0.0,
            }

        failures = [e for e in events if not e.get("tests_passed")]
        return {
            "total_generations": total,
            "success_rate": round((len(successes) / total) * 100, 2),
            "educational_pass_rate": round((len(educational_ok) / total) * 100, 2),
            "avg_attempts": round(mean(attempts), 2),
            "avg_duration_ms": round(mean(durations), 2),
            "by_tema": dashboard,
            "recent_failures": failures[-10:],
        }

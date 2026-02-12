# Generador Educativo de Codigo Python

Estado actual: Fase 0 (11 de febrero de 2026).

## Setup rapido (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecutar backend

```powershell
cd backend
..\venv\Scripts\python -m uvicorn main:app --reload
```

## Endpoint inicial

- `GET /health` -> estado de la API
- `POST /generate` -> genera ejemplo educativo con repair loop (max 3 intentos)
- `GET /metrics` -> dashboard basico de metricas de generaciones

Ejemplo rapido:

```powershell
Invoke-RestMethod -Method POST http://127.0.0.1:8000/generate -ContentType "application/json" -Body '{"tema":"pandas_groupby","nivel":"principiante","contexto":"deportes","tipo":"tutorial","use_rag":false}'
```

Modo ahorro de costos:

- `USE_REAL_LLM=false` usa fallback local (recomendado para desarrollo y tests).
- `USE_REAL_LLM=true` habilita llamadas reales a Anthropic.
- `RAG_PREFER_CHROMA=true` habilita recuperación semántica si existe índice vectorial.

Benchmark local (60 generaciones):

```powershell
.\venv\Scripts\python tests\benchmark.py --runs 60 --output tests\benchmark_results.json
```

## Muestreo dataset (Fase 0)

Genera un subconjunto local limpio para analisis y futura indexacion RAG:

```powershell
.\venv\Scripts\python -m backend.rag.indexer --split non_thinking --target-count 500 --min-edu-score 4 --max-per-package 40
```

Construir índice vectorial Chroma:

```powershell
.\venv\Scripts\python -m backend.rag.indexer --split non_thinking --target-count 500 --min-edu-score 4 --max-per-package 40 --build-chroma --chroma-dir data/chroma --collection-name jupyter_agent_examples
```

Salida esperada:

- `data/jupyter-agent/processed/sample_non_thinking_*.jsonl`
- `data/jupyter-agent/processed/sample_non_thinking_*_meta.json`

## Estructura base

- `backend/` API y pipeline de generacion
- `frontend/` interfaz web (Next.js/React)
- `data/` datasets y vectores
- `docs/` taxonomia, esquema API y analisis
- `tests/` pruebas unitarias e integracion

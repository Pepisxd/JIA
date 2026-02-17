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
- `GET /history` -> historial persistido de generaciones
- `GET /history/{id}` -> detalle completo de una generación

Ejemplo rapido:

```powershell
Invoke-RestMethod -Method POST http://127.0.0.1:8000/generate -ContentType "application/json" -Body '{"tema":"pandas_groupby","nivel":"principiante","contexto":"deportes","tipo":"tutorial","use_rag":false}'
```

Nota:

- Si no envias `use_rag`, el backend usa `use_rag=true` por defecto.
- Envia `use_rag=false` solo cuando quieras comparar contra baseline sin recuperacion.

Modo ahorro de costos:

- `USE_REAL_LLM=false` usa fallback local (recomendado para desarrollo y tests).
- `USE_REAL_LLM=true` habilita llamadas reales a Anthropic.
- `RAG_PREFER_CHROMA=true` habilita recuperación semántica si existe índice vectorial.

Modo modelo local fine-tuned (V2):

- `USE_LOCAL_MODEL=true` activa inferencia con modelo local (default recomendado).
- `LOCAL_MODEL_REQUIRED=false` evita que el backend caiga si faltan dependencias/modelo (fallback automático).
- `MODEL_BASE=codellama/CodeLlama-7b-Instruct-hf`
- `MODEL_PATH=./models/codellama-edugen-v2`
- `MODEL_DEVICE_MAP=auto`
- Opcional offline estricto: `LOCAL_MODEL_LOCAL_FILES_ONLY=true`

Ejemplo `.env`:

```env
USE_LOCAL_MODEL=true
MODEL_BASE=codellama/CodeLlama-7b-Instruct-hf
MODEL_PATH=./models/codellama-edugen-v2
MODEL_DEVICE_MAP=auto
LOCAL_MODEL_LOCAL_FILES_ONLY=false
USE_REAL_LLM=false
```

Arranque backend con modelo local:

```powershell
python -m uvicorn backend.main:app --reload
```

Si falta `peft/transformers/bitsandbytes`:

```powershell
python -m pip install -r requirements-local-model.txt
```

Persistencia:

- Usa `DATABASE_URL` en `.env`.
- Default local: `sqlite:///./data/app.db`.
- Producción PostgreSQL ejemplo:
  `postgresql+psycopg://usuario:password@localhost:5432/ia_generator`.

## Docker (API + PostgreSQL)

Levantar servicios:

```powershell
docker-compose up -d --build
```

Verificar:

```powershell
docker-compose ps
```

API quedara en:

- `http://127.0.0.1:8000`

Si ejecutas API local (fuera de Docker) y quieres usar el Postgres del compose:

```env
DATABASE_URL=postgresql+psycopg://ia_user:ia_pass@localhost:5432/ia_generator
```

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

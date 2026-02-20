# Generador Educativo de Codigo Python

Backend FastAPI para generar ejercicios de Python orientados a ciencia de datos con:

- generacion por LLM local/remoto
- repair loop con validacion educativa
- ejecucion segura de codigo
- metricas y persistencia de historial
- soporte opcional de RAG

## Estado actual

- Fase: `fase_1_5` (`GET /health`)
- Endpoint principal: `POST /generate`
- Repair loop maximo: `3` intentos
- Modo debug por intento: `POST /generate?debug=true`
- Tests locales recientes: `69 passed, 1 skipped`

## Arquitectura

Flujo principal:

1. `backend/generator.py` genera respuesta (local model, remoto Anthropic o fallback determinista).
2. `backend/parser.py` parsea secciones:
   - `## OBJETIVO`
   - `## DATASET`
   - `## CODIGO`
   - `## EXPLICACION`
3. `backend/executor.py` ejecuta codigo en entorno controlado.
4. `backend/validators.py` valida calidad educativa y coherencia dataset/codigo.
5. `backend/repair.py` aplica retries:
   - parcheo por modelo para texto corto (objetivo/explicacion)
   - parcheo determinista para incoherencia de codigo
6. `backend/metrics.py` guarda eventos en `logs/metrics/generations.jsonl`.
7. `backend/store.py` persiste historial en DB (`sqlite` o `postgres`).

## Requisitos

- Python 3.11 recomendado
- pip actualizado
- (Opcional) Docker Desktop para stack con Postgres
- (Opcional) GPU CUDA para inferencia local con LoRA

Dependencias base: `requirements.txt`  
Dependencias modelo local: `requirements-local-model.txt`

## Setup local rapido (Windows PowerShell)

```powershell
py -3.11 -m venv venv311
.\venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copia variables de entorno:

```powershell
Copy-Item .env.example .env
```

## Configuracion `.env`

Variables clave:

```env
# Backend generation mode
USE_LOCAL_MODEL=true
USE_REAL_LLM=false

# Local model
MODEL_BASE=codellama/CodeLlama-7b-Instruct-hf
MODEL_PATH=./models/codellama-edugen-v2
MODEL_DEVICE_MAP=auto
LOCAL_MODEL_REQUIRED=false
LOCAL_MODEL_LOCAL_FILES_ONLY=false

# Remote model (Anthropic), solo si USE_REAL_LLM=true
ANTHROPIC_API_KEY=tu_api_key
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
ANTHROPIC_MAX_TOKENS=700

# RAG
RAG_PREFER_CHROMA=false

# Persistence
DATABASE_URL=sqlite:///./data/app.db
```

Modos de ejecucion recomendados:

- Desarrollo barato: `USE_LOCAL_MODEL=false`, `USE_REAL_LLM=false` (fallback determinista).
- Local LoRA: `USE_LOCAL_MODEL=true`, `USE_REAL_LLM=false`.
- Remoto Anthropic: `USE_LOCAL_MODEL=false`, `USE_REAL_LLM=true`.

## Ejecutar API local

```powershell
.\venv311\Scripts\python -m uvicorn backend.main:app --reload
```

API:

- `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`

## Endpoints

- `GET /health`
- `POST /generate`
- `GET /metrics`
- `GET /history`
- `GET /history/{id}`

Ejemplo `POST /generate`:

```powershell
curl -X POST "http://127.0.0.1:8000/generate" `
  -H "Content-Type: application/json" `
  -d "{\"tema\":\"pandas_groupby\",\"nivel\":\"principiante\",\"contexto\":\"deportes\",\"tipo\":\"tutorial\",\"use_rag\":false}"
```

Ejemplo `debug`:

```powershell
curl -X POST "http://127.0.0.1:8000/generate?debug=true" `
  -H "Content-Type: application/json" `
  -d "{\"tema\":\"pandas_filtrado\",\"nivel\":\"intermedio\",\"contexto\":\"finanzas\",\"tipo\":\"desafio\",\"use_rag\":false}"
```

`debug.attempts` incluye por intento:

- `prompt_sent`
- `raw_text`
- `sanitized_text`
- `validation_errors`
- `parse_ok`
- `exec_ok`
- `exec_error`

## Tests

Suite completa:

```powershell
.\venv311\Scripts\python -m pytest -q
```

Algunos tests importantes:

- `tests/test_repair_loop.py`: repair deterministic-first y patching.
- `tests/test_validators.py`: reglas educativas y coherencia dataset/codigo.
- `tests/test_generate_debug.py`: trazabilidad de intentos en `debug=true`.
- `tests/test_executor_dataset_injection.py`: inyeccion de `rows` al ejecutar.
- `tests/test_local_model_smoke.py`: smoke opcional del modelo local.

Smoke local model (opcional):

```powershell
$env:RUN_LOCAL_MODEL_SMOKE="true"
.\venv311\Scripts\python -m pytest -q tests/test_local_model_smoke.py
```

## Benchmarks

Benchmark general:

```powershell
.\venv311\Scripts\python tests\benchmark.py --runs 60 --output tests\benchmark_results.json
```

Comparativas disponibles:

- `tests/benchmark_compare_rag.py`
- `tests/benchmark_compare_v1_v2.py`

## RAG y dataset scripts

Indexado de muestra:

```powershell
.\venv311\Scripts\python -m backend.rag.indexer --split non_thinking --target-count 500 --min-edu-score 4 --max-per-package 40
```

Construir indice Chroma:

```powershell
.\venv311\Scripts\python -m backend.rag.indexer --split non_thinking --target-count 500 --min-edu-score 4 --max-per-package 40 --build-chroma --chroma-dir data/chroma --collection-name jupyter_agent_examples
```

Scripts de fine-tuning dataset:

- `backend/scripts/analyze_for_finetuning.py`
- `backend/scripts/prepare_training_data.py`
- `backend/scripts/validate_training_data.py`

## Docker (API + PostgreSQL)

Levantar:

```powershell
docker-compose up -d --build
docker-compose ps
```

Servicios:

- API: `http://127.0.0.1:8000`
- Postgres: `localhost:5432`

Compose de comparacion V1 vs V2:

```powershell
docker compose -f docker-compose.compare.yml up -d --build api_v1 api_v2
```

## Persistencia y logs

- DB por defecto: `sqlite:///./data/app.db`
- Historial: tabla `generation_history`
- Eventos de metricas: `logs/metrics/generations.jsonl`

## Troubleshooting rapido

1. `No module named peft`:
   - instala dependencias opcionales:
   ```powershell
   .\venv311\Scripts\python -m pip install -r requirements-local-model.txt
   ```

2. `No GPU found. A GPU is needed for quantization.`:
   - usa WSL2/Linux con CUDA para modelo local en 4-bit
   - o desactiva local model (`USE_LOCAL_MODEL=false`)

3. Requests muy lentos:
   - baja `LOCAL_MODEL_MAX_NEW_TOKENS`
   - usa fallback determinista para desarrollo
   - valida que no estes en `debug=true` para pruebas de latencia

4. Conflictos de entorno:
   - usa `venv311`
   - evita mezclar Python 3.13 con dependencias de inferencia local

## Estructura del proyecto

```text
backend/
  main.py
  generator.py
  parser.py
  executor.py
  repair.py
  validators.py
  metrics.py
  db.py
  store.py
  rag/
data/
docs/
logs/
models/
tests/
```

## Nota sobre `jupyter-course/`

La carpeta `jupyter-course/` se maneja como repo separado dentro del workspace.  
No forma parte del flujo principal de este backend salvo que quieras integrarlo manualmente.


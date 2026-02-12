# Analisis inicial del dataset `jupyter-agent/jupyter-agent-dataset`

Fecha de analisis: 2026-02-11.
Metodo usado: metadata + streaming (sin descarga completa del corpus).

## Hallazgos principales

- Configuracion detectada: `default`.
- Splits disponibles:
  - `thinking`: 51,389 ejemplos
  - `non_thinking`: 51,389 ejemplos
- Tamaño aproximado por split (`num_bytes`): ~51 GB cada uno (dataset muy grande para descarga completa inmediata).

## Esquema de campos observado

- `messages` (historial estructurado de conversacion + tool calls)
- `id`
- `edu_score`
- `files_used`
- `packages_used`
- `question`
- `answer`
- `kaggle_dataset_name`
- `executor_type`
- `original_notebook`
- `tools`

## Implicaciones para arquitectura

1. Para Fase 0-1 conviene explorar por streaming y muestrear, evitando descarga full.
2. Para Fase 2 (RAG), indexar un subconjunto curado (por ejemplo 5k-20k ejemplos) para mantener costos/latencia controlados.
3. `question`, `answer`, `packages_used`, `edu_score` y secciones parseadas de `original_notebook` son buenos candidatos de chunking.

## Siguientes pasos recomendados

1. Implementar script de muestreo estratificado por `packages_used` y `edu_score`.
2. Definir formato intermedio limpio para RAG en `data/jupyter-agent/processed/`.
3. Construir notebook de EDA inicial y tabla de frecuencias de temas/paquetes.

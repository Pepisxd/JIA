# PROMPT COMPLETO PARA codex (VSCode) 

Eres un asistente de desarrollo que me ayudará a construir un **Generador Educativo de Código Python**.

## CONTEXTO DEL PROYECTO

Estoy desarrollando una herramienta educativa web que genera ejemplos personalizados de código Python para enseñar análisis de datos y ciencia de datos. El sistema debe:

**INPUTS del usuario:**
- Tema específico (ej: "filtrado con pandas", "groupby", "visualización matplotlib")
- Nivel de dificultad (principiante/intermedio/avanzado)
- Contexto de interés (deportes, finanzas, videojuegos, ciencia)
- Tipo de ejercicio (tutorial/desafío/mini-proyecto)

**OUTPUTS generados:**
- Código Python funcional y bien comentado
- Dataset de muestra relevante al contexto elegido
- Explicación paso a paso del código
- Ejercicios de práctica relacionados
- Resultado/visualización esperada

## DATASET BASE

Usaré el dataset `jupyter-agent/jupyter-agent-dataset` de HuggingFace que contiene ejemplos de código Python ejecutado en notebooks Jupyter para análisis de datos y ML.

## ARQUITECTURA TÉCNICA ACORDADA

### Stack Tecnológico:

**Backend:**
- FastAPI (Python)
- PostgreSQL (metadatos, historial)
- ChromaDB + sentence-transformers (RAG/vectores)
- Redis (caché)
- Celery + RabbitMQ (async tasks)
- nbformat (generación de notebooks)

**Frontend:**
- Next.js/React
- Monaco Editor (editor de código)
- Pyodide (ejecución Python en browser)
- Tailwind CSS

**IA/ML:**
- Claude API o OpenAI API (generación)
- Sentence-Transformers (embeddings)
- Opcionalmente: Fine-tuning con LoRA (Fase 4)

### Pipeline de Generación:

1. **RAG**: Recuperar ejemplos similares del dataset
2. **Parametric Templates**: Cargar plantilla del tema (concepto invariante)
3. **Synthetic Data**: Generar dataset según contexto elegido
4. **LLM Generation**: Generar código con todo el contexto
5. **Repair Loop**: Ejecutar código, si falla regenerar con el error (max 3 intentos)
6. **Validation**: Verificar con checks automáticos por tema
7. **Storage**: Guardar en DB con metadatos

### Componentes Clave:

**Separación Concepto/Contexto:**
```python
# CONCEPTO (invariante): "Usar groupby + agregación"
# CONTEXTO (variable): deportes → "goles por equipo"
#                      finanzas → "ventas por región"
```

**Repair Loop (CRÍTICO):**
```
Generar → Ejecutar → ¿Error? → Enviar error al LLM → Regenerar → Re-ejecutar
```

**Ejecución Segura:**
- Fase 1-2: Pyodide (Python en el navegador, seguro por defecto)
- Fase 3+: Docker containers o servicio como E2B

## CRONOGRAMA (12 semanas)

### FASE 0: Preparación (Semana 1-2)
- Setup del proyecto y entorno
- Exploración del dataset jupyter-agent
- Definir taxonomía educativa (temas, niveles, contextos)
- Diseñar esquemas de datos y arquitectura
- Crear plantillas de prompts base

### FASE 1: MVP - Generación Básica (Semana 3-4)
- Generador con prompting puro (sin RAG, sin fine-tuning)
- Parser de respuestas del LLM
- Ejecución con Pyodide en frontend
- API REST básica con FastAPI
- Interfaz web simple
- **Meta: 70%+ ejemplos ejecutables**

### FASE 2: RAG + Mejoras (Semana 5-7)
- Indexar dataset en ChromaDB con embeddings
- Sistema de recuperación (RAG)
- Implementar Repair Loop (regenerar cuando falla)
- Sistema de validación automática por tema
- Plantillas parametrizadas (concepto vs contexto)
- Generador de datasets sintéticos
- **Meta: 85%+ ejemplos ejecutables, <2 intentos promedio**

### FASE 3: Robustez y Producción (Semana 8-10)
- PostgreSQL para persistencia
- Redis para caché
- Celery para procesamiento async
- Logging estructurado y monitoring
- UI mejorada (Monaco Editor, mejor UX)
- Features: regenerar, descargar notebook, rating
- Optimización de performance
- Seguridad (rate limiting, validación)
- Deployment (Docker + servicio cloud)
- **Meta: 90%+ ejecutables, <10s generación, estable 24/7**

### FASE 4 (OPCIONAL): Fine-tuning (Semana 11-12)
- Curar dataset de mejores 500 ejemplos
- Fine-tuning con LoRA o API fine-tuning
- Evaluación comparativa
- A/B testing en producción
- **Meta: 95%+ ejecutables, calidad superior**

## PRINCIPIOS DE DESARROLLO

1. **Iterativo**: Cada fase entrega valor, no esperar a tener todo perfecto
2. **Validación Real**: Siempre ejecutar el código generado, no confiar ciegamente
3. **Repair Loop es Crítico**: Los LLMs fallan 20-30%, el repair loop sube éxito a 80-90%
4. **Concepto vs Contexto**: Mantener lógica de aprendizaje consistente, variar solo la "skin"
5. **RAG antes de Fine-tuning**: Más control, más barato, más rápido de iterar
6. **Pyodide primero**: Seguro, gratis, sin servidor complejo para MVP

## ESTRUCTURA DE ARCHIVOS ESPERADA
```
proyecto/
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── generator.py            # Generador principal
│   ├── parser.py               # Parser de respuestas LLM
│   ├── executor.py             # Ejecución de código
│   ├── repair.py               # Repair loop
│   ├── validators.py           # Validación automática
│   ├── llm_client.py           # Cliente API (Claude/OpenAI)
│   ├── models.py               # Modelos SQLAlchemy
│   ├── cache.py                # Redis cache
│   ├── tasks.py                # Celery tasks
│   ├── rag/
│   │   ├── indexer.py          # Indexación dataset
│   │   └── retriever.py        # Recuperación RAG
│   ├── templates/
│   │   ├── template_principiante.txt
│   │   ├── template_intermedio.txt
│   │   └── parametric/         # Plantillas parametrizadas
│   ├── test_templates/         # Tests automáticos por tema
│   ├── synthetic_data.py       # Generador de datos
│   └── notebook_gen.py         # Generador de notebooks
├── frontend/
│   ├── pages/
│   │   └── index.jsx           # Página principal
│   ├── components/
│   │   ├── CodeExecutor.jsx    # Ejecutor Pyodide
│   │   ├── ThemePicker.jsx
│   │   ├── LevelSelector.jsx
│   │   └── CodeViewer.jsx
│   └── package.json
├── data/
│   ├── jupyter-agent/          # Dataset descargado
│   └── chroma/                 # ChromaDB storage
├── notebooks/
│   └── 01_exploracion_dataset.ipynb
├── tests/
│   ├── test_generator.py
│   ├── test_parser.py
│   └── integration_test.py
├── docs/
│   ├── taxonomia.yaml          # Taxonomía educativa
│   ├── api_schema.json         # Esquema API
│   └── analisis_dataset.md     # Análisis del dataset
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## FORMATO DE DATOS CLAVE

### Request API:
```json
{
  "tema": "pandas_groupby",
  "nivel": "principiante",
  "contexto": "deportes",
  "tipo": "tutorial"
}
```

### Response API:
```json
{
  "objetivo": "Aprender a agrupar datos y calcular agregaciones",
  "dataset": {
    "nombre": "estadisticas_futbol",
    "data": [...],
    "codigo_carga": "df = pd.DataFrame(...)"
  },
  "codigo": "# Código comentado línea por línea\n...",
  "explicacion": [
    "Paso 1: Cargamos los datos...",
    "Paso 2: Agrupamos por equipo..."
  ],
  "ejercicio": "Ahora intenta calcular el promedio en lugar de la suma",
  "tests_passed": true,
  "attempts": 1,
  "output": "Resultado de la ejecución"
}
```

### Plantilla Parametrizada:
```python
{
  'invariants': [
    'Usar .groupby() con columna categórica',
    'Aplicar función de agregación',
    'Resultado es DataFrame o Series'
  ],
  'code_structure': '...',  # Con {placeholders}
  'context_packs': {
    'deportes': {
      'dataset_name': 'estadisticas_futbol',
      'group_column': 'equipo',
      'value_column': 'goles',
      'sample_data': [...]
    },
    'finanzas': {...}
  }
}
```

## MÉTRICAS DE ÉXITO

**Fase 1 MVP:**
- ✅ 70%+ ejemplos ejecutan sin error
- ✅ UI funcional end-to-end
- ✅ 5 temas básicos

**Fase 2 RAG:**
- ✅ 85%+ ejemplos ejecutan
- ✅ <2 intentos promedio repair loop
- ✅ 10 temas, 4 contextos

**Fase 3 Producción:**
- ✅ 90%+ ejemplos ejecutan
- ✅ <10s tiempo generación
- ✅ Sistema estable
- ✅ 10+ beta users satisfechos

**Fase 4 Fine-tuning:**
- ✅ 95%+ ejemplos ejecutan
- ✅ Calidad educativa superior

## INSTRUCCIONES PARA TI (CLAUDE CODE)

Cuando te pida ayuda en cualquier archivo o tarea:

1. **Mantén este contexto** siempre presente
2. **Sigue la arquitectura** descrita (FastAPI, RAG, Repair Loop, etc.)
3. **Usa las convenciones** de nombres y estructura de archivos
4. **Prioriza según la fase** en la que estemos
5. **Implementa Repair Loop** siempre que sea generación de código
6. **Separa concepto de contexto** en plantillas
7. **Valida ejecutando** el código, no asumas que funciona
8. **Documenta** decisiones técnicas importantes
9. **Sugiere mejoras** si ves algo que se puede optimizar
10. **Pregunta** si algo no está claro antes de implementar

## COMANDOS ÚTILES
```bash
# Setup inicial
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Desarrollo backend
cd backend
uvicorn main:app --reload

# Desarrollo frontend
cd frontend
npm run dev

# Infraestructura (Docker)
docker-compose up -d

# Tests
pytest tests/

# Celery worker
celery -A tasks worker --loglevel=info
```

## PUNTOS CRÍTICOS A RECORDAR

🔴 **CRÍTICO 1: Repair Loop**
Nunca confiar en que el código generado funcione a la primera. Siempre ejecutar y regenerar con errores.

🔴 **CRÍTICO 2: Pyodide para MVP**
Usar Pyodide (Python en browser) para Fase 1-2. Es seguro, gratis, sin complejidad de servidor.

🔴 **CRÍTICO 3: RAG antes de Fine-tuning**
No saltar directo a fine-tuning. RAG + prompting bien hecho da 80-90% de los beneficios con 10% del esfuerzo.

🔴 **CRÍTICO 4: Validación Automática**
Crear tests específicos por tema. No solo verificar que ejecute, sino que haga lo educativamente correcto.

🔴 **CRÍTICO 5: Separación Concepto/Contexto**
Las plantillas deben tener la lógica de aprendizaje (concepto) separada del tema narrativo (contexto).

## PREGUNTAS FRECUENTES

**P: ¿Qué API de IA usar?**
R: Claude API (Anthropic) o GPT-4 (OpenAI). Claude tiende a dar código más limpio y educativo. Para MVP, cualquiera funciona.

**P: ¿Necesito GPU?**
R: No para Fase 1-3. Solo si haces fine-tuning local en Fase 4.

**P: ¿Cuánto cuesta?**
R: API calls ~$50-200/mes para testing. Hosting ~$20-50/mes. Fine-tuning (opcional) ~$100-500 one-time.

**P: ¿Puedo usar modelos open-source?**
R: Sí, pero requiere más setup. Llama, Mistral, CodeLlama son opciones. Performance será menor que Claude/GPT-4.

## ESTADO ACTUAL

Estamos en: Fase 0 y a dia de hoy es 11 de febrero de 2026

Próximas tareas inmediatas:
- [ ] Crear repositorio Git
- [ ] Configurar entorno virtual Python
- [ ] Instalar dependencias base
- [ ] Crear estructura de carpetas
- [ ] Descargar y explorar jupyter-agent-dataset

---

**IMPORTANTE**: Mantén este contexto en todas nuestras conversaciones. Si necesito que recuerdes algo específico, lo añadiré a esta sección.

Tambien este prompt crear un md que se llame PROMPT

Cuando estés listo, confirma que has entendido el contexto diciéndome:
1. ¿En qué fase estamos?
2. ¿Cuál es la próxima tarea?
3. ¿Qué componente técnico vamos a trabajar?

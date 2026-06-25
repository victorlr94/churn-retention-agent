# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/);
el proyecto sigue [SemVer](https://semver.org/lang/es/).

## [Unreleased]

## [0.7.1] — 2026-06-24

### Added

- `docs/index.html`: página de presentación del proyecto para reclutadores — propósito, arquitectura, stack, métricas del modelo, guardrails y aprendizajes, en una sola página autocontenida (sin dependencias externas).
- `docs/screenshots/demo-overview.png`: captura de la demo Streamlit embebida en el README.

### Changed

- `src/churn_agent/demo/streamlit_app.py`: layout de la demo reorganizado en **tres columnas** (ofertas disponibles · factores de riesgo · valores reales); los valores reales se dividen en **dos sub-columnas** para mejor aprovechamiento del espacio.
- `README.md`: sección **Demo en vivo** con el screenshot embebido y descripción de la pantalla.

### Fixed

- Typo «Propensidad» → «Propensión» en el prompt del agente (`agent/graph.py`) y en la CLI (`scripts/run_agent.py`).

## [0.7.0] — 2026-06-23

### Added — Fase 7: Demo interactiva para reclutadores

- `scripts/make_sample.py`: genera `data/sample/telco_sample.csv` (500 filas estratificadas) y `models/demo/lgbm_demo.pkl`; se ejecuta una vez en local y los artefactos se commitean.
- `data/sample/telco_sample.csv` (121 KB): fixture de muestra commiteada; misma tasa de churn (~26.6%) que el dataset completo.
- `models/demo/lgbm_demo.pkl` (769 KB): `LightGBMChurnModel` entrenado sobre el sample; commiteado con allowlist en `.gitignore`.
- `src/churn_agent/demo/service.py`: `DemoService` con `analyze(customer_id)` y `approve(thread_id, approved)` en modo offline (StatefulFakeLLM + modelo demo). Propensión, EV y HITL son reales; solo el LLM es determinista.
- `src/churn_agent/demo/streamlit_app.py`: UI de una pantalla — selector de cliente, métricas (propensión, CLTV, EV), factores de riesgo por desviación, tabla EV por tier, compuerta HITL (Aprobar/Rechazar), mensaje final.
- `app.py` (raíz): shim para Hugging Face Spaces (SDK Streamlit).
- `requirements.txt`: exportado con `uv export --extra demo --no-hashes`.
- `Dockerfile`: imagen mínima para deploy alternativo en Render/Fly.
- `src/churn_agent/config.py`: campos `demo_sample_path` y `demo_model_path`.
- `pyproject.toml`: grupo opcional `demo = ["streamlit>=1.40"]`; overrides mypy para `churn_agent.demo.*`.
- `docs/architecture/adr/ADR-0008.md`: decisión de modo offline + fixtures commiteadas.
- 8 tests unitarios de demo (`tests/demo/test_demo_service.py`) sin API key, sin CSV real, sin pkl real. Total: 145 tests.

## [0.6.0] — 2026-06-23

### Added — Fase 6: Observabilidad

- `src/churn_agent/observability/session_log.py`: `ObservabilityStore` JSONL append-only, `SessionRecord` y `SessionMetrics`; modo no-op cuando `log_path=None`.
- `src/churn_agent/observability/cost.py`: `estimate_session_cost` — estimacion de tokens y coste USD basada en Claude Haiku 4.5 ($0.80/1M input, $4.00/1M output, ~4 chars/token).
- `src/churn_agent/api/routes/metrics.py`: `GET /api/v1/metrics` — metricas agregadas de sesiones (latencia media, tasa HITL, tasa bloqueados, distribucion de tiers, coste estimado).
- `src/churn_agent/api/dependencies.py`: `get_observability_store` — devuelve no-op si el store no esta inicializado (evita cambios en tests existentes).
- `src/churn_agent/api/routes/agent.py`: instrumentacion en `analyze_customer` y `approve_session` — mide latencia con `time.monotonic()`, estima coste y registra `SessionRecord`.
- `src/churn_agent/api/app.py`: inicializa `ObservabilityStore` en el lifespan y lo expone en `app.state`.
- `src/churn_agent/config.py`: campo `session_log_path` (default: `logs/sessions.jsonl`).
- `scripts/show_sessions.py`: CLI `--last N`, `--summary`, `--log PATH` para inspeccionar el log de sesiones desde la terminal.
- `docs/architecture/adr/ADR-0007.md`: decision de JSONL sobre OTEL/SQLite/LangSmith.
- `.gitignore`: `logs/*` + `!logs/.gitkeep` para gitignorear el log de sesiones.
- 15 tests unitarios de observabilidad (no-op store, escritura/lectura, metricas, `build_session_record`, `estimate_session_cost`). Total: 137 tests.

## [0.5.0] — 2026-06-23

### Added — Fase 5: Evaluacion determinista del agente

- `src/churn_agent/evaluation/cases.py`: `EvalCase` (frozen dataclass) y `EVAL_SUITE` con 5 casos cubriendo las 4 rutas del grafo (baja propension, sin HITL, con HITL aprobado, bloqueado por seguridad, rechazo humano).
- `src/churn_agent/evaluation/fake_components.py`: `FakeChurnModel` (implementa `ChurnModel` Protocol) y `StatefulFakeLLM` (extiende `BaseChatModel`; simula la secuencia query_propensity → select_best_offer → mensaje final sin red ni API key).
- `src/churn_agent/evaluation/metrics.py`: `EvalMetrics`, `compute_metrics` y `gate_passes` con umbrales: `pass_rate >= 0.80`, `tier_accuracy >= 0.80`, `hitl_recall = 1.00`, `block_rate = 1.00`.
- `src/churn_agent/evaluation/runner.py`: `run_eval_case` y `run_eval_suite` — ejecutan cada caso contra el grafo real con componentes falsos; `EvalResult` recoge el veredicto por caso.
- `scripts/evaluate_agent.py`: CLI `--exit-code` — sale con codigo 1 si el gate falla; integrado en CI.
- `.github/workflows/ci.yml`: paso `uv run python scripts/evaluate_agent.py --exit-code` que bloquea el merge si las metricas caen por debajo del umbral.
- `docs/architecture/adr/ADR-0006.md`: decision de evaluacion determinista (sin pkl, sin CSV, sin API key).
- 24 tests unitarios (11 de metricas, 13 de runner). Total: 122 tests.

## [0.4.0] — 2026-06-23

### Added — Fase 4: API REST

- `src/churn_agent/api/app.py`: aplicación FastAPI con lifespan (carga modelo + grafo una vez al arrancar; modo degradado si el pkl no existe).
- `src/churn_agent/api/schemas.py`: modelos Pydantic de request/response (`AnalysisResult`, `AnalyzeResponse`, `ApproveRequest`, `ApproveResponse`).
- `src/churn_agent/api/dependencies.py`: inyección de dependencias (`get_compiled_graph` → 503 si modelo no disponible).
- `src/churn_agent/api/routes/health.py`: `GET /health` — estado de la API y número de clientes cargados.
- `src/churn_agent/api/routes/agent.py`:
  - `POST /api/v1/customers/{customer_id}/analyze` — ejecuta el agente; devuelve `completed`, `pending_approval` (HITL) o `blocked` (injection).
  - `POST /api/v1/sessions/{session_id}/approve` — reanuda el grafo tras la aprobación humana.
- `docs/architecture/adr/ADR-0005.md`: decisión de FastAPI sobre Flask/DRF.
- 8 tests unitarios de API con `TestClient` y dependency overrides (sin servidor real).
- Swagger UI disponible en `http://localhost:8000/docs` cuando el servidor está corriendo.

## [0.3.0] — 2026-06-23

### Added — Fase 3: Agente LangGraph con HITL

- `src/churn_agent/economics/ev.py`: `OfferTier`, `compute_ev(propensity, cltv, uplift, cost)` y `select_best_offer` con catálogo LIGHT/STANDARD/PREMIUM.
- `src/churn_agent/agent/guards.py`: input guard (11 regex de prompt injection, case-insensitive) y output guard (catálogo de tiers).
- `src/churn_agent/agent/tools.py`: tools como closures que capturan modelo y datos — `query_propensity` y `select_best_offer`.
- `src/churn_agent/agent/graph.py`: `StateGraph` con loop ReAct (analyst ↔ tools) y nodo `human_gate` con `interrupt()` nativo de LangGraph.
- `src/churn_agent/agent/runner.py`: `run_retention_agent()` orquesta el flujo completo con `approve_callback` configurable.
- `scripts/run_agent.py`: CLI `--customer-id` con opción `--no-hitl` para demos.
- `docs/architecture/adr/ADR-0004.md`: decisión de LangGraph sobre LCEL/CrewAI.
- `src/churn_agent/exceptions.py`: añadido `SecurityError`.
- `src/churn_agent/config.py`: añadidos `anthropic_api_key`, `llm_model`, `hitl_ev_threshold`, `model_path`.
- 27 tests: 11 de economics, 8 de guards, 5 de tools, 3 de graph.

## [0.2.0] — 2026-06-22

### Added — Fase 2: Modelo de propensión

- `src/churn_agent/model/preprocessor.py`: `build_preprocessor` — `ColumnTransformer`
  con auto-detección de tipos, descarte de alta cardinalidad e imputación.
- `src/churn_agent/model/baseline.py`: `train_majority_baseline` y `train_logistic_baseline`.
- `src/churn_agent/model/trainer.py`: `train_lgbm` (Pipeline LightGBM) y `calibrate`
  (isotónica con `IsotonicRegression`; reemplaza `CalibratedClassifierCV(cv="prefit")` eliminado en sklearn ≥1.6).
- `src/churn_agent/model/evaluator.py`: `compute_metrics` (AUC-ROC, AUC-PR, Brier,
  recall@decil1, lift@decil1) y `compare_models`.
- `src/churn_agent/model/explainer.py`: `get_shap_values` y `top_drivers` (SHAP TreeExplainer).
- `src/churn_agent/model/lgbm_model.py`: `LightGBMChurnModel` — adapter que implementa
  el `ChurnModel` Protocol para uso single-customer del agente.
- `scripts/train_model.py`: pipeline end-to-end (carga → split 60/20/20 → baseline → LGBM
  → calibración → evaluación → guardado del modelo).
- `models/.gitkeep`: directorio de artefactos (`.pkl` gitignoreados).
- `docs/architecture/adr/ADR-0003.md`: decisión de modelo (LightGBM vs alternativas).
- 22 tests unitarios para todos los módulos del modelo.

### Fixed

- Esquema adaptado al dataset IBM Telco extendido (`CustomerID` sin espacio, sin `Satisfaction Score`).
- `_coerce_to_numeric` via `FunctionTransformer` para manejar blancos en `Total Charges`.
- Checksum en `data/checksums.txt` actualizado al dataset correcto.

## [0.1.0] — 2026-06-20

### Added — Fase 1: Datos

- `scripts/download_data.py`: instrucciones de descarga manual + verificación SHA-256.
- `data/checksums.txt`: archivo de checksums versionado (sin el CSV).
- `src/churn_agent/data/schema.py`: `LEAKAGE_COLUMNS`, `EXCLUDED_FROM_FEATURES`,
  constantes de columnas y `TelcoRawSchema` (pandera DataFrameModel).
- `src/churn_agent/data/loader.py`: `load_raw`, `check_no_leakage`, `load_features`
  con guardia explícita que lanza `LeakageError`.
- `src/churn_agent/data/splitter.py`: `make_split` estratificado reproducible.
- Tests unitarios para schema, leakage (parametrizado) y estratificación.

### Added — Fase 0: Setup del proyecto

- Entorno reproducible con uv (`pyproject.toml`, `uv.lock`, `.python-version`).
- Estructura núcleo/dominio (`src/churn_agent/`).
- Configuración centralizada (`Settings` con Pydantic) y excepciones de dominio.
- Calidad: Ruff + mypy (strict en el núcleo) + pre-commit.
- CI en GitHub Actions: lint + tipos + tests.
- Documentación base: README, ARCHITECTURE, DATA_CARD, SECURITY, ADR-0001, ADR-0002.

### Security

- Secretos fuera del repo: `.gitignore` de `.env*`, `.env.example` sin valores,
  gitleaks y `detect-private-key` en pre-commit.

[Unreleased]: https://github.com/victorlr94/churn-retention-agent/compare/v0.7.1...HEAD
[0.7.1]: https://github.com/victorlr94/churn-retention-agent/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/victorlr94/churn-retention-agent/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/victorlr94/churn-retention-agent/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/victorlr94/churn-retention-agent/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/victorlr94/churn-retention-agent/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/victorlr94/churn-retention-agent/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/victorlr94/churn-retention-agent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/victorlr94/churn-retention-agent/releases/tag/v0.1.0

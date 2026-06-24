# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/);
el proyecto sigue [SemVer](https://semver.org/lang/es/).

## [Unreleased]

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

[Unreleased]: https://github.com/victorlr94/churn-retention-agent/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/victorlr94/churn-retention-agent/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/victorlr94/churn-retention-agent/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/victorlr94/churn-retention-agent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/victorlr94/churn-retention-agent/releases/tag/v0.1.0

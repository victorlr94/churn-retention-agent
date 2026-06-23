# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/);
el proyecto sigue [SemVer](https://semver.org/lang/es/).

## [Unreleased]

### Added
- Fase 2 — Modelo de propensión:
  - `src/churn_agent/model/preprocessor.py`: `build_preprocessor` — ColumnTransformer
    con auto-detección de tipos, descarte de alta cardinalidad e imputación.
  - `src/churn_agent/model/baseline.py`: `train_majority_baseline` y `train_logistic_baseline`.
  - `src/churn_agent/model/trainer.py`: `train_lgbm` (Pipeline LightGBM) y `calibrate`
    (isotonic/Platt con cv="prefit").
  - `src/churn_agent/model/evaluator.py`: `compute_metrics` (AUC-ROC, AUC-PR, Brier,
    recall@decil1, lift@decil1) y `compare_models`.
  - `src/churn_agent/model/explainer.py`: `get_shap_values` y `top_drivers` (SHAP TreeExplainer).
  - `src/churn_agent/model/lgbm_model.py`: `LightGBMChurnModel` — adapter que implementa
    el `ChurnModel` Protocol para uso single-customer del agente.
  - `scripts/train_model.py`: pipeline end-to-end (carga → split → baseline → LGBM →
    calibración → evaluación → guardado del modelo).
  - `models/.gitkeep`: directorio de artefactos de modelo (`.pkl` gitignoreados).
  - `docs/architecture/adr/ADR-0003.md`: decisión de modelo (LightGBM vs alternativas).
  - 22 tests unitarios para todos los módulos del modelo — sin necesitar el CSV real.
- Fase 1 — Datos:
  - `scripts/download_data.py`: instrucciones de descarga manual + verificación SHA-256.
  - `data/checksums.txt`: archivo de checksums versionado (sin el CSV).
  - `src/churn_agent/data/schema.py`: `LEAKAGE_COLUMNS`, `EXCLUDED_FROM_FEATURES`,
    constantes de columnas y `TelcoRawSchema` (pandera DataFrameModel).
  - `src/churn_agent/data/loader.py`: `load_raw`, `check_no_leakage`, `load_features`
    con guardia explícita que lanza `LeakageError` si las columnas prohibidas aparecen.
  - `src/churn_agent/data/splitter.py`: `make_split` estratificado reproducible.
  - Tests unitarios para schema, leakage (parametrizado por columna) y stratificación
    — todos en memoria, sin necesitar el CSV real.
- Fase 0 — Setup del proyecto:
  - Entorno reproducible con uv (`pyproject.toml`, `uv.lock`, `.python-version`).
  - Estructura núcleo/dominio (`src/churn_agent/` con `core`, `data`, `model`,
    `economics`, `agent`, `api`).
  - Configuración centralizada (`Settings` con Pydantic) y excepciones de dominio.
  - Calidad: Ruff + mypy (strict en el núcleo) + pre-commit.
  - CI en GitHub Actions: lint + tipos + tests.
  - Smoke tests de la Fase 0.
  - Documentación: README, ARCHITECTURE, DATA_CARD, SECURITY, ADR-0001 y ADR-0002.

### Security
- Secretos fuera del repo: `.gitignore` de `.env*`, `.env.example` sin valores,
  gitleaks y `detect-private-key` en pre-commit.

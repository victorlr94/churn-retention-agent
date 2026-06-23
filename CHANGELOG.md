# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/);
el proyecto sigue [SemVer](https://semver.org/lang/es/).

## [Unreleased]

### Added
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

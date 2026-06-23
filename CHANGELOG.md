# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/);
el proyecto sigue [SemVer](https://semver.org/lang/es/).

## [Unreleased]

### Added
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

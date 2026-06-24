"""Configuración centralizada: una única fuente de verdad.

Carga desde variables de entorno y `.env`. Ningún número mágico ni ruta
hardcodeada debe vivir fuera de aquí (skill: python-code-quality).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raíz del proyecto: dos niveles arriba de este archivo (src/churn_agent/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configuración del proyecto, inyectada hacia abajo en el sistema."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CHURN_",
        extra="ignore",
    )

    # Rutas de datos (gitignored; ver scripts/download_data.py en Fase 1).
    data_raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    data_processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")

    # Reproducibilidad.
    random_seed: int = Field(default=42)

    # Artefacto del modelo (gitignored; genera con scripts/train_model.py).
    model_path: Path = Field(
        default=PROJECT_ROOT / "models" / "lgbm_churn_calibrated.pkl"
    )

    # LLM — ANTHROPIC_API_KEY sin prefijo CHURN_ (convención estándar de Anthropic).
    anthropic_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "CHURN_ANTHROPIC_API_KEY"),
    )
    llm_model: str = Field(default="claude-haiku-4-5-20251001")

    # Human-in-the-loop: EV mínimo (MXN) para requerir aprobación humana.
    hitl_ev_threshold: float = Field(default=300.0)

    # Observabilidad: ruta del log JSONL de sesiones (gitignoreado).
    session_log_path: Path = Field(default=PROJECT_ROOT / "logs" / "sessions.jsonl")


def get_settings() -> Settings:
    """Devuelve la configuración del proyecto."""
    return Settings()

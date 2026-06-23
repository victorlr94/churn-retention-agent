"""Smoke test de la Fase 0: el paquete importa y la configuración carga.

Garantiza que el esqueleto es coherente y que CI tiene algo real que ejecutar
desde el primer commit (skill: testing-strategy).
"""

import pytest

import churn_agent
from churn_agent.config import Settings, get_settings


@pytest.mark.unit
def test_package_has_version() -> None:
    assert churn_agent.__version__ == "0.1.0"


@pytest.mark.unit
def test_settings_load_with_defaults() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.random_seed == 42
    assert settings.data_raw_dir.name == "raw"

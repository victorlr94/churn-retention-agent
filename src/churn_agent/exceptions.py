"""Excepciones de dominio del proyecto.

Se usan en lugar de `Exception` genérico para dar contexto y permitir manejo
selectivo (skill: python-code-quality).
"""

from __future__ import annotations


class ChurnAgentError(Exception):
    """Error base del proyecto."""


class DataValidationError(ChurnAgentError):
    """El dataset no cumple el esquema o las invariantes esperadas."""


class LeakageError(ChurnAgentError):
    """Se intentó usar una columna con fuga de información del target."""


class ModelError(ChurnAgentError):
    """Fallo en entrenamiento, calibración o predicción del modelo."""


class EconomicsError(ChurnAgentError):
    """Fallo en el cálculo de la economía de ofertas."""


class GuardrailError(ChurnAgentError):
    """Una salida del agente violó un guardrail (p. ej. oferta fuera de catálogo)."""

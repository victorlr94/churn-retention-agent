"""Contratos del sistema definidos con `typing.Protocol`.

Toda dependencia difícil de cambiar (el LLM, el modelo de propensión) vive
detrás de un Protocol, de modo que se pueda intercambiar la implementación o
inyectar un fake en tests sin tocar el resto del sistema
(skills: ai-project-architecture, testing-strategy).

Las firmas se completan en las fases correspondientes; aquí queda el contrato.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ChurnModel(Protocol):
    """Modelo de propensión al churn calibrado.

    Devuelve una probabilidad real en [0, 1], no un score sin calibrar:
    la economía de ofertas (EV) depende de que sea una probabilidad.
    """

    def predict_proba(self, features: dict[str, object]) -> float: ...


@runtime_checkable
class LLM(Protocol):
    """Proveedor de generación de texto detrás de una interfaz neutral."""

    def generate(self, prompt: str, **opts: object) -> str: ...

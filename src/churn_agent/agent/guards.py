"""Guardrails de seguridad: input guard y output guard.

El input guard bloquea intentos de prompt injection antes de que lleguen al LLM.
El output guard verifica que la oferta generada sea coherente con el catálogo.

Referencia: OWASP LLM Top 10 — LLM01 Prompt Injection.
"""

from __future__ import annotations

import re

from churn_agent.economics.ev import OFFER_TIERS
from churn_agent.exceptions import GuardrailError, SecurityError

# Patrones que indican intento de prompt injection o jailbreak.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+.{0,25}instructions?",
        r"forget\s+(your|all|previous)\s+instructions?",
        r"you\s+are\s+(now\s+)?a\b",
        r"\bact\s+as\b",
        r"pretend\s+(you\s+are|to\s+be)",
        r"\bjailbreak\b",
        r"\bDAN\b",
        r"system\s+prompt",
        r"override\s+(all\s+)?instructions?",
        r"<\s*script",
        r"\beval\s*\(",
    ]
]

# Nombres de tiers válidos para el output guard.
_VALID_TIERS: frozenset[str] = frozenset(t.name for t in OFFER_TIERS)


def check_input_security(text: str) -> None:
    """Lanza SecurityError si `text` contiene patrones de prompt injection.

    Args:
        text: Cualquier cadena de entrada que el usuario controle
              (customer_id, notas libres, etc.).

    Raises:
        SecurityError: Si se detecta un patrón sospechoso.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            raise SecurityError(
                f"Entrada bloqueada: patrón sospechoso detectado — {pattern.pattern!r}"
            )


def check_output_offer(tier_name: str | None) -> None:
    """Lanza GuardrailError si el tier de oferta no pertenece al catálogo.

    Args:
        tier_name: Nombre del tier propuesto por el agente, o None si no hay oferta.

    Raises:
        GuardrailError: Si el tier no está en el catálogo de OFFER_TIERS.
    """
    if tier_name is None:
        return
    if tier_name not in _VALID_TIERS:
        raise GuardrailError(
            f"Tier {tier_name!r} no pertenece al catálogo. "
            f"Valores válidos: {sorted(_VALID_TIERS)}"
        )

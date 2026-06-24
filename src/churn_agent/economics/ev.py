"""Economía de ofertas de retención.

EV = P(churn) × retention_uplift × CLTV − costo_oferta

Donde:
- P(churn)          = propensidad calibrada del modelo (en [0, 1]).
- retention_uplift  = fracción de clientes en riesgo que se retienen con la oferta.
- CLTV              = Customer Lifetime Value en MXN.
- costo_oferta      = costo directo de la oferta (descuento + gestión).

Solo se lanza una oferta cuando EV > 0 (rentable). Para clientes de alto EV
(EV > Settings.hitl_ev_threshold) se requiere aprobación humana (Fase 3).
"""

from __future__ import annotations

from dataclasses import dataclass

from churn_agent.exceptions import EconomicsError


@dataclass(frozen=True)
class OfferTier:
    """Nivel de oferta de retención con su economía asociada."""

    name: str
    cost: float
    retention_uplift: float
    description: str


# Catálogo de ofertas, ordenado de menor a mayor intensidad.
OFFER_TIERS: list[OfferTier] = [
    OfferTier(
        name="LIGHT",
        cost=10.0,
        retention_uplift=0.40,
        description="Descuento 5% durante 3 meses",
    ),
    OfferTier(
        name="STANDARD",
        cost=25.0,
        retention_uplift=0.65,
        description="Descuento 15% más upgrade de plan incluido",
    ),
    OfferTier(
        name="PREMIUM",
        cost=50.0,
        retention_uplift=0.85,
        description="Descuento 25% más loyalty points y upgrade de plan",
    ),
]


def compute_ev(
    propensity: float,
    cltv: float,
    uplift: float,
    offer_cost: float,
) -> float:
    """Valor esperado de lanzar una oferta a un cliente en riesgo.

    Raises:
        EconomicsError: Si algún parámetro está fuera de rango.
    """
    if not (0.0 <= propensity <= 1.0):
        raise EconomicsError(f"propensity debe estar en [0, 1]; recibido: {propensity}")
    if cltv < 0:
        raise EconomicsError(f"CLTV debe ser ≥ 0; recibido: {cltv}")
    if not (0.0 <= uplift <= 1.0):
        raise EconomicsError(f"uplift debe estar en [0, 1]; recibido: {uplift}")
    return propensity * uplift * cltv - offer_cost


def select_best_offer(
    propensity: float,
    cltv: float,
    tiers: list[OfferTier] | None = None,
) -> tuple[OfferTier, float] | None:
    """Tier con mayor EV positivo, o None si todos los EVs son ≤ 0.

    Args:
        propensity: P(churn) calibrado en [0, 1].
        cltv: Customer Lifetime Value en MXN.
        tiers: Catálogo de ofertas; si es None, usa OFFER_TIERS.

    Returns:
        (OfferTier, ev) del mejor tier, o None si no hay oferta rentable.
    """
    effective_tiers = tiers if tiers is not None else OFFER_TIERS
    best: tuple[OfferTier, float] | None = None
    for tier in effective_tiers:
        ev = compute_ev(propensity, cltv, tier.retention_uplift, tier.cost)
        if ev > 0 and (best is None or ev > best[1]):
            best = (tier, ev)
    return best

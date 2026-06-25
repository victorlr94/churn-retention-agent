"""Economía de ofertas: EV = P(churn) × uplift × CLTV − costo."""

from churn_agent.economics.ev import (
    OFFER_TIERS,
    OfferTier,
    compute_ev,
    select_best_offer,
)

__all__ = ["OFFER_TIERS", "OfferTier", "compute_ev", "select_best_offer"]

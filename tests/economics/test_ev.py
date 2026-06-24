"""Tests unitarios de la economía de ofertas."""

from __future__ import annotations

import pytest

from churn_agent.economics.ev import (
    OFFER_TIERS,
    OfferTier,
    compute_ev,
    select_best_offer,
)
from churn_agent.exceptions import EconomicsError

# ---------------------------------------------------------------------------
# compute_ev
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_ev_positive() -> None:
    # p=0.8, uplift=0.65, cltv=3000, cost=25 → EV = 0.8*0.65*3000 - 25 = 1535
    ev = compute_ev(propensity=0.8, cltv=3000.0, uplift=0.65, offer_cost=25.0)
    assert ev == pytest.approx(1535.0)


@pytest.mark.unit
def test_compute_ev_zero_cltv() -> None:
    ev = compute_ev(propensity=0.9, cltv=0.0, uplift=0.85, offer_cost=50.0)
    assert ev == pytest.approx(-50.0)


@pytest.mark.unit
def test_compute_ev_low_propensity_may_be_negative() -> None:
    # propensity muy baja → EV negativo aunque CLTV sea alto
    ev = compute_ev(propensity=0.01, cltv=1000.0, uplift=0.40, offer_cost=10.0)
    assert ev < 0


@pytest.mark.unit
def test_compute_ev_invalid_propensity_above_one() -> None:
    with pytest.raises(EconomicsError, match="propensity"):
        compute_ev(propensity=1.1, cltv=3000.0, uplift=0.65, offer_cost=25.0)


@pytest.mark.unit
def test_compute_ev_invalid_propensity_negative() -> None:
    with pytest.raises(EconomicsError, match="propensity"):
        compute_ev(propensity=-0.1, cltv=3000.0, uplift=0.65, offer_cost=25.0)


@pytest.mark.unit
def test_compute_ev_invalid_cltv_negative() -> None:
    with pytest.raises(EconomicsError, match="CLTV"):
        compute_ev(propensity=0.7, cltv=-100.0, uplift=0.65, offer_cost=25.0)


@pytest.mark.unit
def test_compute_ev_invalid_uplift() -> None:
    with pytest.raises(EconomicsError, match="uplift"):
        compute_ev(propensity=0.7, cltv=3000.0, uplift=1.5, offer_cost=25.0)


# ---------------------------------------------------------------------------
# select_best_offer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_select_best_offer_returns_tier_for_high_risk_high_value() -> None:
    result = select_best_offer(propensity=0.8, cltv=5000.0)
    assert result is not None
    tier, ev = result
    assert ev > 0
    assert tier.name in {"LIGHT", "STANDARD", "PREMIUM"}


@pytest.mark.unit
def test_select_best_offer_none_when_all_ev_negative() -> None:
    # CLTV=0 → todos los EVs son -cost < 0
    result = select_best_offer(propensity=0.5, cltv=0.0)
    assert result is None


@pytest.mark.unit
def test_select_best_offer_premium_for_very_high_value() -> None:
    # CLTV=10000, propensity=0.9 → PREMIUM tiene más EV que STANDARD
    result = select_best_offer(propensity=0.9, cltv=10000.0)
    assert result is not None
    tier, _ = result
    assert tier.name == "PREMIUM"


@pytest.mark.unit
def test_select_best_offer_uses_custom_tiers() -> None:
    cheap = OfferTier(name="CHEAP", cost=1.0, retention_uplift=0.1, description="x")
    expensive = OfferTier(name="EXP", cost=999.0, retention_uplift=0.9, description="y")
    # CHEAP: EV = 0.5*0.1*100 - 1 = 4 > 0 ✓   EXP: EV = 0.5*0.9*100 - 999 < 0
    result = select_best_offer(propensity=0.5, cltv=100.0, tiers=[cheap, expensive])
    assert result is not None
    tier, ev = result
    assert tier.name == "CHEAP"
    assert ev == pytest.approx(4.0)


@pytest.mark.unit
def test_offer_tiers_catalog_has_three_entries() -> None:
    assert len(OFFER_TIERS) == 3
    names = [t.name for t in OFFER_TIERS]
    assert names == ["LIGHT", "STANDARD", "PREMIUM"]

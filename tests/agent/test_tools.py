"""Tests de las tools del agente (con modelo y dataset falsos)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from churn_agent.agent.tools import make_tools
from churn_agent.core.interfaces import ChurnModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeChurnModel:
    """Modelo fake: devuelve siempre la propensidad configurada."""

    def __init__(self, fixed_proba: float = 0.75) -> None:
        self._proba = fixed_proba

    def predict_proba(self, features: dict[str, object]) -> float:
        return self._proba


def _make_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CustomerID": ["AAA-111", "BBB-222", "CCC-333"],
            "CLTV": [3000, 6500, 1200],
            "Tenure Months": [12, 48, 3],
            "Monthly Charges": [70.0, 90.0, 45.0],
            "Contract": ["Month-to-month", "Two year", "Month-to-month"],
            "Churn Value": [1, 0, 1],
        }
    )


FEATURE_NAMES = ["Tenure Months", "Monthly Charges", "Contract"]


@pytest.fixture()
def tools() -> list:  # type: ignore[type-arg]
    model: ChurnModel = _FakeChurnModel(fixed_proba=0.75)
    return make_tools(model, _make_data(), FEATURE_NAMES)


def _get_tool(tools_list: list[Any], name: str) -> Any:
    for t in tools_list:
        if t.name == name:
            return t
    raise KeyError(f"Tool {name!r} not found")


# ---------------------------------------------------------------------------
# query_propensity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_query_propensity_found(tools: list) -> None:  # type: ignore[type-arg]
    tool = _get_tool(tools, "query_propensity")
    result = json.loads(tool.invoke({"customer_id": "AAA-111"}))
    assert result["customer_id"] == "AAA-111"
    assert result["propensity"] == pytest.approx(0.75, abs=1e-3)
    assert result["cltv"] == 3000
    assert isinstance(result["top_drivers"], list)


@pytest.mark.unit
def test_query_propensity_not_found(tools: list) -> None:  # type: ignore[type-arg]
    tool = _get_tool(tools, "query_propensity")
    result = json.loads(tool.invoke({"customer_id": "NONEXISTENT"}))
    assert "error" in result


@pytest.mark.unit
def test_query_propensity_high_value_customer(tools: list) -> None:  # type: ignore[type-arg]
    tool = _get_tool(tools, "query_propensity")
    result = json.loads(tool.invoke({"customer_id": "BBB-222"}))
    assert result["cltv"] == 6500


# ---------------------------------------------------------------------------
# select_best_offer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_select_best_offer_high_value_returns_premium(tools: list) -> None:  # type: ignore[type-arg]
    tool = _get_tool(tools, "select_best_offer")
    # propensity=0.9, cltv=8000 → PREMIUM tiene el mayor EV positivo
    result = json.loads(tool.invoke({"propensity": 0.9, "cltv": 8000}))
    assert result["tier"] == "PREMIUM"
    assert result["ev"] > 0


@pytest.mark.unit
def test_select_best_offer_no_viable_offer(tools: list) -> None:  # type: ignore[type-arg]
    tool = _get_tool(tools, "select_best_offer")
    # CLTV=0 → todos los EVs son negativos
    result = json.loads(tool.invoke({"propensity": 0.5, "cltv": 0}))
    assert result["tier"] is None
    assert result["ev"] == pytest.approx(0.0)


@pytest.mark.unit
def test_select_best_offer_returns_positive_ev(tools: list) -> None:  # type: ignore[type-arg]
    tool = _get_tool(tools, "select_best_offer")
    result = json.loads(tool.invoke({"propensity": 0.75, "cltv": 3000}))
    if result["tier"] is not None:
        assert result["ev"] > 0


@pytest.mark.unit
def test_make_tools_returns_two_tools() -> None:
    model: ChurnModel = _FakeChurnModel()
    tools = make_tools(model, _make_data(), FEATURE_NAMES)
    names = {t.name for t in tools}
    assert names == {"query_propensity", "select_best_offer"}

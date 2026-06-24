"""Tools del agente: funciones que el LLM puede invocar durante el análisis.

Cada tool es una closure que captura las dependencias (modelo, datos) en el momento
de construcción. Esto permite inyectar fakes en tests sin tocar el grafo.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from churn_agent.core.interfaces import ChurnModel
from churn_agent.data.schema import COL_CLTV, COL_CUSTOMER_ID
from churn_agent.economics.ev import select_best_offer as _ev_select_best_offer

logger = logging.getLogger(__name__)


def _top_drivers(
    data: pd.DataFrame,
    row: pd.Series,  # type: ignore[type-arg]  # pd.Series no tiene alias genérico completo
    feature_names: list[str],
    n: int = 3,
) -> list[str]:
    """Top-n features por desviación estandarizada respecto a la población."""
    deviations: list[tuple[str, float]] = []
    for col in feature_names:
        if col not in data.columns or col not in row.index:
            continue
        series = pd.to_numeric(data[col], errors="coerce").dropna()
        val = pd.to_numeric(row[col], errors="coerce")
        if series.empty or pd.isna(val):
            continue
        std = float(series.std())
        if std > 0:
            dev = abs(float(val) - float(series.median())) / std
            deviations.append((col, dev))
    deviations.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in deviations[:n]]


def make_tools(
    model: ChurnModel,
    data: pd.DataFrame,
    feature_names: list[str],
) -> list[Any]:
    """Construye la lista de tools con las dependencias capturadas en closure."""

    @tool
    def query_propensity(customer_id: str) -> str:
        """Get churn propensity, CLTV and top risk drivers for a customer.

        Returns JSON with: customer_id, propensity (0-1), cltv (MXN),
        top_drivers (list of feature names most contributing to risk).
        """
        matches = data[data[COL_CUSTOMER_ID] == customer_id]
        if matches.empty:
            return json.dumps({"error": f"Cliente {customer_id!r} no encontrado."})

        row = matches.iloc[0]
        features: dict[str, object] = {
            col: row[col] for col in feature_names if col in data.columns
        }
        propensity = model.predict_proba(features)
        cltv = int(row[COL_CLTV])
        drivers = _top_drivers(data, row, feature_names)

        logger.debug(
            "query_propensity: customer=%s propensity=%.3f cltv=%d",
            customer_id,
            propensity,
            cltv,
        )
        return json.dumps(
            {
                "customer_id": customer_id,
                "propensity": round(propensity, 4),
                "cltv": cltv,
                "top_drivers": drivers,
            }
        )

    @tool
    def select_best_offer(propensity: float, cltv: int) -> str:
        """Select the optimal retention offer based on churn propensity and CLTV.

        Returns JSON with: tier (LIGHT/STANDARD/PREMIUM/null), ev (expected value
        in MXN), cost (offer cost in MXN), description, propensity, cltv.
        A null tier means no offer is economically viable (all EVs ≤ 0).
        """
        result = _ev_select_best_offer(propensity, float(cltv))
        if result is None:
            return json.dumps(
                {
                    "tier": None,
                    "ev": 0.0,
                    "cost": 0.0,
                    "description": (
                        "No hay oferta viable: EV negativo para todos los niveles."
                    ),
                    "propensity": propensity,
                    "cltv": cltv,
                }
            )
        tier, ev = result
        logger.debug(
            "select_best_offer: tier=%s ev=%.1f propensity=%.3f cltv=%d",
            tier.name,
            ev,
            propensity,
            cltv,
        )
        return json.dumps(
            {
                "tier": tier.name,
                "ev": round(ev, 2),
                "cost": tier.cost,
                "description": tier.description,
                "propensity": propensity,
                "cltv": cltv,
            }
        )

    return [query_propensity, select_best_offer]

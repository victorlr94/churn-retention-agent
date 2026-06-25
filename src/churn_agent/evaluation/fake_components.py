"""Componentes falsos para evaluación determinista del agente.

Permiten ejecutar el grafo LangGraph sin modelo pkl, CSV ni ANTHROPIC_API_KEY:
  - FakeChurnModel: implementa ChurnModel Protocol con propensidad fija.
  - make_fake_data: DataFrame mínimo con CustomerID y CLTV.
  - StatefulFakeLLM: BaseChatModel que simula el loop ReAct tool-calling.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Any

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from churn_agent.data.schema import COL_CLTV, COL_CUSTOMER_ID


class FakeChurnModel:
    """ChurnModel Protocol fake — devuelve propensidad fija sin modelo real."""

    def __init__(self, propensity: float) -> None:
        if not 0.0 <= propensity <= 1.0:
            raise ValueError(f"propensity must be in [0, 1], got {propensity}")
        self._propensity = propensity

    def predict_proba(self, features: dict[str, object]) -> float:
        return self._propensity


def make_fake_data(customer_id: str, cltv: int) -> pd.DataFrame:
    """DataFrame mínimo que satisface make_tools (CustomerID + CLTV).

    Con feature_names=[], query_propensity construye features={} y
    FakeChurnModel ignora el dict — no se necesita ninguna otra columna.
    """
    return pd.DataFrame({COL_CUSTOMER_ID: [customer_id], COL_CLTV: [cltv]})


class StatefulFakeLLM(BaseChatModel):  # type: ignore[misc]
    """LLM determinista para evaluación — simula el ReAct tool-calling del agente.

    Inspecciona el historial de mensajes para decidir qué devolver:
      1. Sin resultados de tool → llama query_propensity con el customer_id.
      2. query_propensity disponible → llama select_best_offer con propensity+cltv.
      3. Ambos disponibles → genera el mensaje final de recomendación.

    No hace ninguna llamada de red. Devuelve resultados 100% deterministas.
    """

    @property
    def _llm_type(self) -> str:
        return "fake-stateful"

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> StatefulFakeLLM:  # type: ignore[override]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_results: dict[str, str] = {}
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name:
                tool_results[msg.name] = str(msg.content)

        if "query_propensity" not in tool_results:
            customer_id = _extract_customer_id(messages)
            tool_call = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "name": "query_propensity",
                "args": {"customer_id": customer_id},
            }
            return _make_result(AIMessage(content="", tool_calls=[tool_call]))

        if "select_best_offer" not in tool_results:
            raw = tool_results["query_propensity"]
            try:
                data = json.loads(raw)
                propensity = float(data.get("propensity", 0.5))
                cltv = int(data.get("cltv", 1000))
            except (json.JSONDecodeError, TypeError, ValueError):
                propensity, cltv = 0.5, 1000

            tool_call = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "name": "select_best_offer",
                "args": {"propensity": propensity, "cltv": cltv},
            }
            return _make_result(AIMessage(content="", tool_calls=[tool_call]))

        raw_offer = tool_results["select_best_offer"]
        try:
            offer = json.loads(raw_offer)
            tier = offer.get("tier", "UNKNOWN")
            ev = float(offer.get("ev", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            tier, ev = "UNKNOWN", 0.0

        final = (
            f"PROPENSIDAD: {_extract_propensity(tool_results):.0%}\n"
            f"OFERTA: {tier} (EV estimado: {ev:.0f} MXN)\n"
            f"RECOMENDACIÓN: Ofrecer {tier} al cliente."
        )
        return _make_result(AIMessage(content=final))


# ── helpers privados ──────────────────────────────────────────────────────────


def _extract_customer_id(messages: list[BaseMessage]) -> str:
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str) and "Analiza el cliente:" in content:
            return content.split("Analiza el cliente:", 1)[-1].strip()
    return "UNKNOWN"


def _extract_propensity(tool_results: dict[str, str]) -> float:
    raw = tool_results.get("query_propensity", "{}")
    try:
        return float(json.loads(raw).get("propensity", 0.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def _make_result(msg: AIMessage) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=msg)])

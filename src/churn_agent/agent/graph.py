"""Grafo LangGraph del agente de retención.

Flujo:
  input_guard → analyst ⇆ tools → human_gate → END
             ↘ blocked → END   (si se detecta injection)

El nodo `analyst` ejecuta el loop ReAct (razonar → llamar tool → observar).
El nodo `human_gate` interrumpe el grafo (interrupt()) cuando EV > hitl_ev_threshold,
cediendo el control al operador para aprobación explícita.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Literal

import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from typing_extensions import TypedDict

from churn_agent.agent.guards import check_input_security, check_output_offer
from churn_agent.agent.tools import make_tools
from churn_agent.config import get_settings
from churn_agent.core.interfaces import ChurnModel
from churn_agent.exceptions import SecurityError

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Eres un especialista en retención de clientes de telecomunicaciones.
Tu tarea: analizar el riesgo de churn de un cliente y recomendar la oferta óptima.

Flujo obligatorio:
1. Llama a `query_propensity` con el customer_id.
2. Llama a `select_best_offer` con la propensión y el CLTV obtenidos.
3. Redacta un resumen conciso (3-5 líneas) con:
   - Propensión de churn (en %)
   - Valor del cliente (CLTV en MXN)
   - Top drivers de riesgo
   - Oferta recomendada y justificación económica (EV)

Sé directo y profesional. Usa el formato:
PROPENSIÓN: X%
CLTV: $X,XXX MXN
DRIVERS: <lista>
OFERTA: <tier> — <descripción> (EV estimado: $X MXN)
RECOMENDACIÓN: <una frase de acción concreta>
"""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: str
    security_blocked: bool
    propensity: float | None
    cltv: int | None
    ev: float | None
    offer_tier: str | None
    human_approved: bool | None


def _extract_offer_from_messages(
    messages: list[AnyMessage],
) -> dict[str, Any]:
    """Lee el último ToolMessage de select_best_offer y extrae sus campos."""
    for msg in reversed(messages):
        if (
            isinstance(msg, ToolMessage)
            and getattr(msg, "name", "") == "select_best_offer"
        ):
            try:
                raw = msg.content
                if not isinstance(raw, str):
                    break
                data = json.loads(raw)
                return {
                    "propensity": data.get("propensity"),
                    "cltv": data.get("cltv"),
                    "ev": data.get("ev"),
                    "offer_tier": data.get("tier"),
                }
            except (json.JSONDecodeError, AttributeError):
                break
    return {"propensity": None, "cltv": None, "ev": None, "offer_tier": None}


def build_graph(
    model: ChurnModel,
    data: pd.DataFrame,
    feature_names: list[str],
    llm_override: Any | None = None,
) -> Any:
    """Construye y compila el grafo LangGraph del agente de retención.

    Args:
        model: Cualquier implementación de ChurnModel (LightGBMChurnModel o fake).
        data: DataFrame con todas las columnas del dataset (CustomerID, CLTV, features).
        feature_names: Lista de features que el modelo espera.
        llm_override: LLM alternativo (para tests); si es None, usa ChatAnthropic.

    Returns:
        Grafo compilado con MemorySaver (soporta interrupt/resume).
    """
    settings = get_settings()
    tools = make_tools(model, data, feature_names)
    tool_node = ToolNode(tools)

    if llm_override is not None:
        llm_with_tools = llm_override.bind_tools(tools)
    else:
        llm = ChatAnthropic(  # type: ignore[call-arg]
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
        )
        llm_with_tools = llm.bind_tools(tools)

    # ── Nodos ────────────────────────────────────────────────────────────────

    def input_guard_node(state: AgentState) -> dict[str, Any]:
        try:
            check_input_security(state["customer_id"])
            return {"security_blocked": False}
        except SecurityError as exc:
            logger.warning("Input bloqueado: %s", exc)
            return {"security_blocked": True}

    def blocked_node(state: AgentState) -> dict[str, Any]:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Solicitud bloqueada por seguridad. "
                        "El customer_id contiene patrones no permitidos."
                    )
                )
            ]
        }

    def analyst_node(state: AgentState) -> dict[str, Any]:
        # Antepone system prompt en cada llamada; los mensajes anteriores
        # ya incluyen el historial de tool calls y resultados.
        msgs: list[AnyMessage] = [
            SystemMessage(content=_SYSTEM_PROMPT),
            *state["messages"],
        ]
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response]}

    def human_gate_node(state: AgentState) -> dict[str, Any]:
        offer_data = _extract_offer_from_messages(state["messages"])
        ev = offer_data.get("ev")
        offer_tier = offer_data.get("offer_tier")

        # Output guard: el tier debe pertenecer al catálogo.
        check_output_offer(offer_tier)

        updates: dict[str, Any] = offer_data

        if ev is not None and ev > settings.hitl_ev_threshold:
            logger.info(
                "HITL activado: EV=%.1f > umbral=%.1f (tier=%s)",
                ev,
                settings.hitl_ev_threshold,
                offer_tier,
            )
            decision: Any = interrupt(
                {
                    "message": (
                        f"Cliente de alto valor — EV estimado: {ev:.0f} MXN. "
                        f"Oferta propuesta: {offer_tier}. ¿Aprobar? "
                        f"(propensity={offer_data.get('propensity')}, "
                        f"cltv={offer_data.get('cltv')})"
                    ),
                    "ev": ev,
                    "offer_tier": offer_tier,
                    "propensity": offer_data.get("propensity"),
                    "cltv": offer_data.get("cltv"),
                }
            )
            approved = (
                decision.get("approved", False)
                if isinstance(decision, dict)
                else bool(decision)
            )
            updates["human_approved"] = approved
        else:
            updates["human_approved"] = True

        return updates

    # ── Enrutamiento ─────────────────────────────────────────────────────────

    def route_after_guard(state: AgentState) -> Literal["analyst", "blocked"]:
        return "blocked" if state["security_blocked"] else "analyst"

    def route_analyst(state: AgentState) -> Literal["tools", "human_gate"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "human_gate"

    # ── Construcción del grafo ────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("input_guard", input_guard_node)
    graph.add_node("blocked", blocked_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("tools", tool_node)
    graph.add_node("human_gate", human_gate_node)

    graph.set_entry_point("input_guard")
    graph.add_conditional_edges(
        "input_guard",
        route_after_guard,
        {"analyst": "analyst", "blocked": "blocked"},
    )
    graph.add_conditional_edges(
        "analyst",
        route_analyst,
        {"tools": "tools", "human_gate": "human_gate"},
    )
    graph.add_edge("tools", "analyst")
    graph.add_edge("blocked", END)
    graph.add_edge("human_gate", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

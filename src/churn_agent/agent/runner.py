"""Entry point del agente de retención.

Orquesta: carga del modelo + dataset → construcción del grafo → ejecución →
manejo del interrupt HITL → resultado final.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from churn_agent.agent.graph import AgentState, build_graph
from churn_agent.config import get_settings
from churn_agent.data.loader import load_raw
from churn_agent.data.schema import EXCLUDED_FROM_FEATURES
from churn_agent.model.lgbm_model import LightGBMChurnModel

logger = logging.getLogger(__name__)


@dataclass
class RetentionDecision:
    """Resultado de una sesión del agente para un cliente."""

    customer_id: str
    propensity: float | None
    cltv: int | None
    ev: float | None
    offer_tier: str | None
    human_approved: bool
    final_message: str
    security_blocked: bool = False
    messages: list[Any] = field(default_factory=list)


def _default_approve_callback(payload: dict[str, Any]) -> bool:
    """Callback interactivo por terminal (se usa cuando no se pasa uno externo)."""
    print(f"\n{'=' * 60}")
    print("  APROBACIÓN HUMANA REQUERIDA")
    print(f"{'=' * 60}")
    print(f"  {payload.get('message', '')}")
    print(f"{'=' * 60}")
    raw = input("  ¿Aprobar oferta? [s/N]: ").strip().lower()
    approved = raw in {"s", "si", "sí", "y", "yes"}
    print(f"  Decisión: {'APROBADA' if approved else 'RECHAZADA'}")
    return approved


def run_retention_agent(
    customer_id: str,
    *,
    model_path: Path | None = None,
    data_path: Path | None = None,
    approve_callback: Callable[[dict[str, Any]], bool] | None = None,
    thread_id: str | None = None,
    llm_override: Any | None = None,
) -> RetentionDecision:
    """Ejecuta el agente de retención para un cliente.

    Args:
        customer_id: ID del cliente a analizar.
        model_path: Ruta al pkl del modelo (default: settings.model_path).
        data_path: Ruta al CSV del dataset (default: settings.data_raw_dir/...).
        approve_callback: Función (payload) → bool para HITL.
                          Si es None, usa prompt interactivo en terminal.
        thread_id: ID de sesión para el checkpointer de LangGraph.
        llm_override: LLM alternativo (tests).

    Returns:
        RetentionDecision con el resultado completo de la sesión.

    Raises:
        FileNotFoundError: Si el modelo o el dataset no están presentes.
    """
    settings = get_settings()
    callback = approve_callback or _default_approve_callback

    # ── Carga de artefactos ──────────────────────────────────────────────────
    effective_model_path = model_path or settings.model_path
    if not effective_model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en {effective_model_path}. "
            "Ejecuta primero: uv run python scripts/train_model.py"
        )

    churn_model = LightGBMChurnModel.load(effective_model_path)
    data = load_raw(data_path)

    cols_to_drop = EXCLUDED_FROM_FEATURES & set(data.columns)
    feature_cols = [c for c in data.columns if c not in cols_to_drop]

    # ── Construcción del grafo ───────────────────────────────────────────────
    compiled = build_graph(
        model=churn_model,
        data=data,
        feature_names=feature_cols,
        llm_override=llm_override,
    )

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id or customer_id}}

    initial_state: AgentState = {
        "messages": [HumanMessage(content=f"Analiza el cliente: {customer_id}")],
        "customer_id": customer_id,
        "security_blocked": False,
        "propensity": None,
        "cltv": None,
        "ev": None,
        "offer_tier": None,
        "human_approved": None,
    }

    # ── Primera ejecución ────────────────────────────────────────────────────
    result: dict[str, Any] = compiled.invoke(initial_state, config=config)

    # ── Manejo de interrupt HITL ─────────────────────────────────────────────
    graph_state = compiled.get_state(config)
    if graph_state.next:
        # El grafo está suspendido esperando aprobación humana.
        interrupts = graph_state.tasks[0].interrupts if graph_state.tasks else []
        payload: dict[str, Any] = interrupts[0].value if interrupts else {}

        approved = callback(payload)

        result = compiled.invoke(
            Command(resume={"approved": approved}),
            config=config,
        )

    # ── Construcción de la respuesta ─────────────────────────────────────────
    messages = result.get("messages", [])
    final_message = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if content and not getattr(msg, "tool_calls", None):
            final_message = content if isinstance(content, str) else str(content)
            break

    return RetentionDecision(
        customer_id=customer_id,
        propensity=result.get("propensity"),
        cltv=result.get("cltv"),
        ev=result.get("ev"),
        offer_tier=result.get("offer_tier"),
        human_approved=bool(result.get("human_approved", True)),
        final_message=final_message,
        security_blocked=bool(result.get("security_blocked", False)),
        messages=messages,
    )

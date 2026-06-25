"""Endpoints del agente de retención: analyze y approve."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from churn_agent.agent.graph import AgentState
from churn_agent.agent.guards import check_input_security
from churn_agent.api.dependencies import get_compiled_graph, get_observability_store
from churn_agent.api.schemas import (
    AnalysisResult,
    AnalyzeResponse,
    ApproveRequest,
    ApproveResponse,
)
from churn_agent.exceptions import SecurityError
from churn_agent.observability.cost import estimate_session_cost
from churn_agent.observability.session_log import (
    ObservabilityStore,
    build_session_record,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["agent"])


def _result_to_schema(result: dict[str, Any], customer_id: str) -> AnalysisResult:
    """Extrae los campos relevantes del estado final del grafo."""
    msgs = result.get("messages", [])
    final_message = ""
    for msg in reversed(msgs):
        content = getattr(msg, "content", "")
        if content and not getattr(msg, "tool_calls", None):
            final_message = content if isinstance(content, str) else str(content)
            break
    return AnalysisResult(
        customer_id=customer_id,
        propensity=result.get("propensity"),
        cltv=result.get("cltv"),
        ev=result.get("ev"),
        offer_tier=result.get("offer_tier"),
        human_approved=bool(result.get("human_approved", True)),
        final_message=final_message,
        security_blocked=bool(result.get("security_blocked", False)),
    )


@router.post(
    "/customers/{customer_id}/analyze",
    response_model=AnalyzeResponse,
    summary="Analizar cliente y generar oferta de retención",
)
async def analyze_customer(
    customer_id: str,
    compiled: Any = Depends(get_compiled_graph),  # noqa: B008
    obs_store: ObservabilityStore = Depends(get_observability_store),  # noqa: B008
) -> AnalyzeResponse:
    """Ejecuta el agente para un cliente.

    - Si el EV de la oferta supera el umbral HITL, devuelve `pending_approval`
      con un `session_id` para aprobar vía `/sessions/{session_id}/approve`.
    - Si el customer_id contiene patrones de injection, devuelve `blocked`.
    """
    try:
        check_input_security(customer_id)
    except SecurityError as exc:
        raise HTTPException(
            status_code=400,
            detail="customer_id contiene patrones no permitidos (prompt injection).",
        ) from exc

    session_id = str(uuid.uuid4())
    config: dict[str, Any] = {"configurable": {"thread_id": session_id}}

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

    t0 = time.monotonic()
    result: dict[str, Any] = compiled.invoke(initial_state, config=config)
    latency_ms = (time.monotonic() - t0) * 1000

    messages: list[Any] = result.get("messages", [])
    input_tokens, output_tokens, cost_usd = estimate_session_cost(messages)

    if result.get("security_blocked"):
        obs_store.record(
            build_session_record(
                session_id=session_id,
                customer_id=customer_id,
                phase="analyze",
                latency_ms=latency_ms,
                result=result,
                hitl_triggered=False,
                messages=messages,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )
        )
        return AnalyzeResponse(
            status="blocked",
            session_id=session_id,
            decision=_result_to_schema(result, customer_id),
        )

    graph_state = compiled.get_state(config)
    hitl_triggered = bool(graph_state.next)

    obs_store.record(
        build_session_record(
            session_id=session_id,
            customer_id=customer_id,
            phase="analyze",
            latency_ms=latency_ms,
            result=result,
            hitl_triggered=hitl_triggered,
            messages=messages,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
    )

    if hitl_triggered:
        interrupts = graph_state.tasks[0].interrupts if graph_state.tasks else []
        payload: dict[str, Any] = interrupts[0].value if interrupts else {}
        logger.info(
            "HITL activado para customer=%s session=%s EV=%s",
            customer_id,
            session_id,
            payload.get("ev"),
        )
        return AnalyzeResponse(
            status="pending_approval",
            session_id=session_id,
            hitl_payload=payload,
        )

    return AnalyzeResponse(
        status="completed",
        session_id=session_id,
        decision=_result_to_schema(result, customer_id),
    )


@router.post(
    "/sessions/{session_id}/approve",
    response_model=ApproveResponse,
    summary="Aprobar o rechazar una oferta pendiente de revisión humana",
)
async def approve_session(
    session_id: str,
    body: ApproveRequest,
    compiled: Any = Depends(get_compiled_graph),  # noqa: B008
    obs_store: ObservabilityStore = Depends(get_observability_store),  # noqa: B008
) -> ApproveResponse:
    """Reanuda el grafo suspendido tras la decisión humana.

    Llama a este endpoint tras recibir `pending_approval` del endpoint de análisis.
    """
    config: dict[str, Any] = {"configurable": {"thread_id": session_id}}

    try:
        graph_state = compiled.get_state(config)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Sesión {session_id!r} no encontrada.",
        ) from exc

    if not graph_state.next:
        raise HTTPException(
            status_code=409,
            detail=f"Sesión {session_id!r} no está pendiente de aprobación.",
        )

    t0 = time.monotonic()
    result: dict[str, Any] = compiled.invoke(
        Command(resume={"approved": body.approved}),
        config=config,
    )
    latency_ms = (time.monotonic() - t0) * 1000

    customer_id: str = str(result.get("customer_id", "unknown"))
    messages: list[Any] = result.get("messages", [])
    input_tokens, output_tokens, cost_usd = estimate_session_cost(messages)

    obs_store.record(
        build_session_record(
            session_id=session_id,
            customer_id=customer_id,
            phase="approve",
            latency_ms=latency_ms,
            result=result,
            hitl_triggered=True,
            messages=messages,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
    )

    logger.info(
        "HITL resuelto: session=%s approved=%s customer=%s",
        session_id,
        body.approved,
        customer_id,
    )

    return ApproveResponse(
        approved=body.approved,
        decision=_result_to_schema(result, customer_id),
    )

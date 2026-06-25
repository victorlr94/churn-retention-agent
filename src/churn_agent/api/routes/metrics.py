"""Endpoint de métricas de observabilidad del agente."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from churn_agent.api.dependencies import get_observability_store
from churn_agent.api.schemas import MetricsResponse
from churn_agent.observability.session_log import ObservabilityStore

router = APIRouter(prefix="/api/v1", tags=["observability"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Métricas agregadas de sesiones del agente",
)
def get_metrics(
    store: ObservabilityStore = Depends(get_observability_store),  # noqa: B008
) -> MetricsResponse:
    """Devuelve métricas agregadas de todas las sesiones registradas.

    Incluye latencia media, tasa de HITL, distribución de tiers y
    coste estimado acumulado (Claude Haiku 4.5).
    """
    m = store.compute_metrics()
    return MetricsResponse(
        total=m.total,
        avg_latency_ms=m.avg_latency_ms,
        hitl_rate=m.hitl_rate,
        block_rate=m.block_rate,
        tier_distribution=m.tier_distribution,
        total_cost_usd_est=m.total_cost_usd_est,
        avg_cost_usd_est=m.avg_cost_usd_est,
    )

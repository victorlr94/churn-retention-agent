"""Esquemas Pydantic de request/response de la API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    """Resultado completo de la sesión del agente para un cliente."""

    customer_id: str
    propensity: float | None = None
    cltv: int | None = None
    ev: float | None = None
    offer_tier: str | None = None
    human_approved: bool = False
    final_message: str = ""
    security_blocked: bool = False


class AnalyzeResponse(BaseModel):
    """Respuesta del endpoint POST /customers/{customer_id}/analyze."""

    status: Literal["completed", "pending_approval", "blocked"]
    session_id: str
    decision: AnalysisResult | None = None
    hitl_payload: dict[str, Any] | None = None


class ApproveRequest(BaseModel):
    """Body del endpoint POST /sessions/{session_id}/approve."""

    approved: bool = Field(
        description="True para aprobar la oferta propuesta; False para rechazarla."
    )


class ApproveResponse(BaseModel):
    """Respuesta del endpoint de aprobación."""

    status: Literal["completed"] = "completed"
    approved: bool
    decision: AnalysisResult


class MetricsResponse(BaseModel):
    """Métricas agregadas de todas las sesiones del agente."""

    total: int
    avg_latency_ms: float
    hitl_rate: float
    block_rate: float
    tier_distribution: dict[str, int]
    total_cost_usd_est: float
    avg_cost_usd_est: float

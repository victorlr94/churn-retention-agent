"""Endpoint de salud de la API."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["ops"])


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    customers_loaded: int | None = None


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Devuelve el estado de la API y si el modelo está cargado."""
    ready = getattr(request.app.state, "graph_ready", False)
    n = getattr(request.app.state, "n_customers", None)
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_loaded=ready,
        customers_loaded=n if ready else None,
    )

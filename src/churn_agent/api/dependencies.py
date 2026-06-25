"""Dependencias inyectadas por FastAPI en cada request."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request

from churn_agent.observability.session_log import ObservabilityStore


def get_compiled_graph(request: Request) -> Any:
    """Devuelve el grafo LangGraph compilado o lanza 503 si no está listo."""
    if not getattr(request.app.state, "graph_ready", False):
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo no disponible. "
                "Ejecuta primero: uv run python scripts/train_model.py"
            ),
        )
    return request.app.state.compiled_graph


def get_observability_store(request: Request) -> ObservabilityStore:
    """Devuelve el store de observabilidad.

    Si no está inicializado (modo degradado o tests) devuelve un store no-op.
    """
    store = getattr(request.app.state, "observability_store", None)
    if not isinstance(store, ObservabilityStore):
        return ObservabilityStore(log_path=None)
    return store

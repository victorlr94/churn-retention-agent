"""Dependencias inyectadas por FastAPI en cada request."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request


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

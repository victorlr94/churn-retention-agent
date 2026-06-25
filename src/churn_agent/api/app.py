"""Aplicación FastAPI del agente de retención.

Uso:
    uv run uvicorn churn_agent.api.app:app --reload --host 0.0.0.0 --port 8000

Prerequisitos:
    1. uv run python scripts/train_model.py   (genera models/lgbm_churn_calibrated.pkl)
    2. ANTHROPIC_API_KEY configurada en .env
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from churn_agent.agent.graph import build_graph
from churn_agent.api.routes.agent import router as agent_router
from churn_agent.api.routes.health import router as health_router
from churn_agent.api.routes.metrics import router as metrics_router
from churn_agent.config import get_settings
from churn_agent.data.loader import load_raw
from churn_agent.data.schema import EXCLUDED_FROM_FEATURES
from churn_agent.model.lgbm_model import LightGBMChurnModel
from churn_agent.observability.session_log import ObservabilityStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Carga el modelo y el dataset una vez al arrancar; libera al apagar."""
    settings = get_settings()
    app.state.graph_ready = False
    app.state.observability_store = ObservabilityStore(
        log_path=settings.session_log_path
    )
    logger.info("Observabilidad activa: %s", settings.session_log_path)

    try:
        logger.info("Cargando modelo desde %s ...", settings.model_path)
        churn_model = LightGBMChurnModel.load(settings.model_path)
        data = load_raw()

        cols_to_drop = EXCLUDED_FROM_FEATURES & set(data.columns)
        feature_cols = [c for c in data.columns if c not in cols_to_drop]

        app.state.compiled_graph = build_graph(
            model=churn_model,
            data=data,
            feature_names=feature_cols,
        )
        app.state.n_customers = len(data)
        app.state.graph_ready = True
        logger.info("Agente listo: %d clientes cargados.", len(data))
    except FileNotFoundError as exc:
        logger.warning(
            "Modelo no encontrado (%s). "
            "La API arranca en modo degradado — ejecuta scripts/train_model.py.",
            exc,
        )
    except Exception as exc:
        logger.error("Error al cargar el modelo: %s", exc)

    yield

    logger.info("API apagada.")


def create_app() -> FastAPI:
    application = FastAPI(
        title="Churn Retention Agent API",
        description=(
            "API REST del agente anti-churn con human-in-the-loop.\n\n"
            "Analiza el riesgo de churn de un cliente, calcula el EV "
            "de la mejor oferta de retención y gestiona la aprobación humana "
            "para clientes de alto valor."
        ),
        version="0.4.0",
        lifespan=lifespan,
    )
    application.include_router(health_router)
    application.include_router(agent_router)
    application.include_router(metrics_router)
    return application


app = create_app()

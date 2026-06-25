"""Capa de servicio para la demo interactiva.

Construye el grafo LangGraph en modo demo (StatefulFakeLLM + modelo de muestra)
y expone analyze() / approve() que espejean la lógica de api/routes/agent.py
pero sin FastAPI, sin API key y con los fixtures commiteados.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from churn_agent.agent.graph import AgentState, build_graph
from churn_agent.agent.guards import check_input_security
from churn_agent.config import get_settings
from churn_agent.data.loader import load_raw
from churn_agent.data.schema import COL_CLTV, COL_CUSTOMER_ID, EXCLUDED_FROM_FEATURES
from churn_agent.evaluation.fake_components import StatefulFakeLLM
from churn_agent.exceptions import SecurityError
from churn_agent.model.lgbm_model import LightGBMChurnModel

logger = logging.getLogger(__name__)


@dataclass
class DemoResult:
    """Resultado de una llamada a analyze() o approve() en modo demo."""

    status: str  # "completed" | "pending_approval" | "blocked"
    thread_id: str
    customer_id: str
    propensity: float | None = None
    cltv: int | None = None
    ev: float | None = None
    offer_tier: str | None = None
    human_approved: bool | None = None
    final_message: str = ""
    security_blocked: bool = False
    hitl_payload: dict[str, Any] = field(default_factory=dict)


def _extract_final_message(messages: list[Any]) -> str:
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if content and not getattr(msg, "tool_calls", None):
            return content if isinstance(content, str) else str(content)
    return ""


def _build_result(
    result: dict[str, Any],
    *,
    status: str,
    thread_id: str,
    customer_id: str,
    hitl_payload: dict[str, Any] | None = None,
) -> DemoResult:
    messages = result.get("messages", [])
    return DemoResult(
        status=status,
        thread_id=thread_id,
        customer_id=customer_id,
        propensity=result.get("propensity"),
        cltv=result.get("cltv"),
        ev=result.get("ev"),
        offer_tier=result.get("offer_tier"),
        human_approved=result.get("human_approved"),
        final_message=_extract_final_message(messages),
        security_blocked=bool(result.get("security_blocked", False)),
        hitl_payload=hitl_payload or {},
    )


class DemoService:
    """Servicio demo: grafo LangGraph con LLM falso y modelo de muestra.

    Uso típico (singleton en Streamlit):
        service = DemoService()
        result = service.analyze("1234-ABCD")
        if result.status == "pending_approval":
            result = service.approve(result.thread_id, approved=True)
    """

    def __init__(self) -> None:
        settings = get_settings()

        if not settings.demo_sample_path.exists():
            raise FileNotFoundError(
                f"Sample no encontrado en {settings.demo_sample_path}. "
                "Ejecuta: uv run python scripts/make_sample.py"
            )
        if not settings.demo_model_path.exists():
            raise FileNotFoundError(
                f"Modelo demo no encontrado en {settings.demo_model_path}. "
                "Ejecuta: uv run python scripts/make_sample.py"
            )

        churn_model = LightGBMChurnModel.load(settings.demo_model_path)
        data = load_raw(settings.demo_sample_path)

        cols_to_drop = EXCLUDED_FROM_FEATURES & set(data.columns)
        feature_cols = [c for c in data.columns if c not in cols_to_drop]

        self._churn_model: LightGBMChurnModel = churn_model
        self._compiled = build_graph(
            model=churn_model,
            data=data,
            feature_names=feature_cols,
            llm_override=StatefulFakeLLM(),
        )
        self._data = data
        self._feature_cols = feature_cols
        logger.info(
            "DemoService inicializado: %d clientes, modelo=%s",
            len(data),
            settings.demo_model_path.name,
        )

    @property
    def customer_ids(self) -> list[str]:
        """IDs de clientes disponibles en el dataset de muestra."""
        return list(self._data["CustomerID"].astype(str))

    def analyze(self, customer_id: str) -> DemoResult:
        """Ejecuta el agente para un cliente.

        Si el EV supera el umbral HITL, devuelve status='pending_approval'
        con thread_id para llamar a approve() después.
        """
        try:
            check_input_security(customer_id)
        except SecurityError:
            thread_id = str(uuid.uuid4())
            config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
            initial_state: AgentState = {
                "messages": [
                    HumanMessage(content=f"Analiza el cliente: {customer_id}")
                ],
                "customer_id": customer_id,
                "security_blocked": True,
                "propensity": None,
                "cltv": None,
                "ev": None,
                "offer_tier": None,
                "human_approved": None,
            }
            result: dict[str, Any] = self._compiled.invoke(initial_state, config=config)
            return _build_result(
                result,
                status="blocked",
                thread_id=thread_id,
                customer_id=customer_id,
            )

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "messages": [HumanMessage(content=f"Analiza el cliente: {customer_id}")],
            "customer_id": customer_id,
            "security_blocked": False,
            "propensity": None,
            "cltv": None,
            "ev": None,
            "offer_tier": None,
            "human_approved": None,
        }

        result = self._compiled.invoke(initial_state, config=config)

        graph_state = self._compiled.get_state(config)
        if graph_state.next:
            interrupts = graph_state.tasks[0].interrupts if graph_state.tasks else []
            payload: dict[str, Any] = interrupts[0].value if interrupts else {}
            logger.info(
                "HITL activado: customer=%s thread=%s EV=%s",
                customer_id,
                thread_id,
                payload.get("ev"),
            )
            return _build_result(
                result,
                status="pending_approval",
                thread_id=thread_id,
                customer_id=customer_id,
                hitl_payload=payload,
            )

        return _build_result(
            result, status="completed", thread_id=thread_id, customer_id=customer_id
        )

    def customer_row(self, customer_id: str) -> dict[str, Any]:
        """Devuelve los datos raw de un cliente del dataset de muestra."""
        matches = self._data[self._data[COL_CUSTOMER_ID] == customer_id]
        if matches.empty:
            return {}
        return dict(matches.iloc[0])

    def risk_drivers(self, customer_id: str, n: int = 5) -> list[dict[str, Any]]:
        """Top-n features por desviación respecto a la mediana de la población.

        Devuelve lista de dicts con: feature, value, deviation, direction.
        """
        matches = self._data[self._data[COL_CUSTOMER_ID] == customer_id]
        if matches.empty:
            return []
        row = matches.iloc[0]
        results: list[tuple[str, float, float, str]] = []
        for col in self._feature_cols:
            if col == COL_CLTV or col not in self._data.columns:
                continue
            series = pd.to_numeric(self._data[col], errors="coerce").dropna()
            raw_val: Any = row.get(col)
            val = pd.to_numeric(raw_val, errors="coerce")  # type: ignore[arg-type]
            if series.empty or pd.isna(val):
                continue
            std = float(series.std())
            if std <= 0:
                continue
            median = float(series.median())
            dev = (float(val) - median) / std
            results.append(
                (col, float(val), abs(dev), "↑ riesgo" if dev > 0 else "↓ riesgo")
            )
        results.sort(key=lambda x: x[2], reverse=True)
        return [
            {"feature": f, "value": v, "deviation": d, "direction": dr}
            for f, v, d, dr in results[:n]
        ]

    def customer_propensity(self, customer_id: str) -> tuple[float, int] | None:
        """Propensión y CLTV sin ejecutar el grafo completo.

        Permite mostrar los datos de análisis en la UI antes de que el
        usuario decida si ejecutar el agente.
        """
        matches = self._data[self._data[COL_CUSTOMER_ID] == customer_id]
        if matches.empty:
            return None
        row = matches.iloc[0]
        features: dict[str, object] = {
            col: row[col] for col in self._feature_cols if col in row.index
        }
        propensity = self._churn_model.predict_proba(features)
        raw_cltv: Any = row.get(COL_CLTV, 2000)
        cltv_val = pd.to_numeric(raw_cltv, errors="coerce")  # type: ignore[arg-type]
        cltv = int(cltv_val) if not pd.isna(cltv_val) else 2000
        return propensity, cltv

    def approve(self, thread_id: str, *, approved: bool) -> DemoResult:
        """Reanuda el grafo suspendido tras la decisión humana."""
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        result: dict[str, Any] = self._compiled.invoke(
            Command(resume={"approved": approved}),
            config=config,
        )
        customer_id = str(result.get("customer_id", "unknown"))
        return _build_result(
            result, status="completed", thread_id=thread_id, customer_id=customer_id
        )

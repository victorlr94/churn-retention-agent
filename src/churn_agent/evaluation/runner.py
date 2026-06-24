"""Runner de evaluación del agente.

Ejecuta cada EvalCase usando FakeChurnModel + StatefulFakeLLM para que la
suite completa sea determinista y no requiera modelo pkl, CSV ni API key.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from churn_agent.agent.graph import AgentState, build_graph
from churn_agent.evaluation.cases import EVAL_SUITE, EvalCase
from churn_agent.evaluation.fake_components import (
    FakeChurnModel,
    StatefulFakeLLM,
    make_fake_data,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Resultado de un caso de evaluación individual."""

    case: EvalCase
    actual_tier: str | None
    actual_hitl: bool
    actual_blocked: bool
    passed: bool
    failure_reason: str = field(default="")


def run_eval_case(case: EvalCase) -> EvalResult:
    """Ejecuta un EvalCase con componentes fake y evalúa el resultado.

    No requiere modelo pkl, CSV ni ANTHROPIC_API_KEY.
    """
    fake_model = FakeChurnModel(propensity=case.propensity)
    fake_data = make_fake_data(case.customer_id, case.cltv)
    fake_llm = StatefulFakeLLM()

    compiled = build_graph(
        model=fake_model,
        data=fake_data,
        feature_names=[],
        llm_override=fake_llm,
    )

    thread_id = str(uuid.uuid4())
    config: dict[str, object] = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "messages": [HumanMessage(content=f"Analiza el cliente: {case.customer_id}")],
        "customer_id": case.customer_id,
        "security_blocked": False,
        "propensity": None,
        "cltv": None,
        "ev": None,
        "offer_tier": None,
        "human_approved": None,
    }

    result: dict[str, object] = compiled.invoke(initial_state, config=config)

    graph_state = compiled.get_state(config)
    hitl_triggered = bool(graph_state.next)

    if hitl_triggered:
        result = compiled.invoke(
            Command(resume={"approved": case.approve_hitl}),
            config=config,
        )

    raw_tier = result.get("offer_tier")
    actual_tier: str | None = str(raw_tier) if isinstance(raw_tier, str) else None
    actual_blocked = bool(result.get("security_blocked", False))

    passed, failure_reason = _evaluate(
        case, actual_tier, hitl_triggered, actual_blocked
    )

    logger.debug(
        "eval case=%s passed=%s tier=%s hitl=%s blocked=%s",
        case.case_id,
        passed,
        actual_tier,
        hitl_triggered,
        actual_blocked,
    )

    return EvalResult(
        case=case,
        actual_tier=actual_tier,
        actual_hitl=hitl_triggered,
        actual_blocked=actual_blocked,
        passed=passed,
        failure_reason=failure_reason,
    )


def run_eval_suite(suite: list[EvalCase] | None = None) -> list[EvalResult]:
    """Ejecuta todos los casos de la suite y devuelve los resultados."""
    cases = suite if suite is not None else EVAL_SUITE
    results: list[EvalResult] = []
    for case in cases:
        result = run_eval_case(case)
        results.append(result)
    return results


def _evaluate(
    case: EvalCase,
    actual_tier: str | None,
    hitl_triggered: bool,
    actual_blocked: bool,
) -> tuple[bool, str]:
    """Compara resultado real con expectativas del caso. Devuelve (passed, reason)."""
    reasons: list[str] = []

    if case.expect_blocked != actual_blocked:
        reasons.append(f"expect_blocked={case.expect_blocked} got={actual_blocked}")

    if not case.expect_blocked:
        if case.expect_hitl != hitl_triggered:
            reasons.append(f"expect_hitl={case.expect_hitl} got={hitl_triggered}")
        if case.expect_tier is not None and actual_tier != case.expect_tier:
            reasons.append(f"expect_tier={case.expect_tier!r} got={actual_tier!r}")

    failure_reason = "; ".join(reasons)
    return (len(reasons) == 0, failure_reason)

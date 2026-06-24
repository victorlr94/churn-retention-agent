"""Tests unitarios de la capa de observabilidad."""

from __future__ import annotations

from pathlib import Path

import pytest

from churn_agent.observability.cost import estimate_session_cost
from churn_agent.observability.session_log import (
    ObservabilityStore,
    SessionRecord,
    build_session_record,
)

# ── helpers ────────────────────────────────────────────────────────────────────


def _make_record(**kwargs: object) -> SessionRecord:
    defaults: dict[str, object] = dict(
        session_id="s1",
        customer_id="CUST-001",
        phase="analyze",
        timestamp_utc="2026-06-23T10:00:00+00:00",
        latency_ms=500.0,
        propensity=0.72,
        cltv=3500,
        ev=250.0,
        offer_tier="STANDARD",
        hitl_triggered=False,
        human_approved=True,
        security_blocked=False,
        n_messages=5,
        input_tokens_est=200,
        output_tokens_est=50,
        cost_usd_est=0.00000036,
    )
    defaults.update(kwargs)
    return SessionRecord(**defaults)  # type: ignore[arg-type]


# ── ObservabilityStore ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_noop_store_record_does_not_raise() -> None:
    store = ObservabilityStore(log_path=None)
    record = _make_record()
    store.record(record)  # no-op, no excepcion


@pytest.mark.unit
def test_noop_store_load_returns_empty() -> None:
    store = ObservabilityStore(log_path=None)
    assert store.load_all() == []


@pytest.mark.unit
def test_noop_store_metrics_returns_zeros() -> None:
    store = ObservabilityStore(log_path=None)
    m = store.compute_metrics()
    assert m.total == 0
    assert m.avg_latency_ms == pytest.approx(0.0)


@pytest.mark.unit
def test_store_writes_and_reads_record(tmp_path: Path) -> None:
    log = tmp_path / "sessions.jsonl"
    store = ObservabilityStore(log_path=log)
    record = _make_record()
    store.record(record)

    records = store.load_all()
    assert len(records) == 1
    r = records[0]
    assert r.session_id == "s1"
    assert r.customer_id == "CUST-001"
    assert r.offer_tier == "STANDARD"
    assert r.latency_ms == pytest.approx(500.0)


@pytest.mark.unit
def test_store_appends_multiple_records(tmp_path: Path) -> None:
    log = tmp_path / "sessions.jsonl"
    store = ObservabilityStore(log_path=log)
    for i in range(3):
        store.record(_make_record(session_id=f"s{i}", customer_id=f"C{i}"))

    records = store.load_all()
    assert len(records) == 3
    assert [r.session_id for r in records] == ["s0", "s1", "s2"]


@pytest.mark.unit
def test_store_creates_parent_dir(tmp_path: Path) -> None:
    log = tmp_path / "subdir" / "deep" / "sessions.jsonl"
    ObservabilityStore(log_path=log)  # debe crear los directorios
    assert log.parent.exists()


@pytest.mark.unit
def test_store_load_empty_file_returns_empty(tmp_path: Path) -> None:
    log = tmp_path / "sessions.jsonl"
    log.touch()
    store = ObservabilityStore(log_path=log)
    assert store.load_all() == []


# ── compute_metrics ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_compute_metrics_single_record(tmp_path: Path) -> None:
    log = tmp_path / "sessions.jsonl"
    store = ObservabilityStore(log_path=log)
    store.record(
        _make_record(latency_ms=1000.0, hitl_triggered=False, security_blocked=False)
    )

    m = store.compute_metrics()
    assert m.total == 1
    assert m.avg_latency_ms == pytest.approx(1000.0)
    assert m.hitl_rate == pytest.approx(0.0)
    assert m.block_rate == pytest.approx(0.0)
    assert m.tier_distribution == {"STANDARD": 1}


@pytest.mark.unit
def test_compute_metrics_hitl_rate(tmp_path: Path) -> None:
    log = tmp_path / "sessions.jsonl"
    store = ObservabilityStore(log_path=log)
    store.record(_make_record(session_id="s1", hitl_triggered=True))
    store.record(_make_record(session_id="s2", hitl_triggered=False))
    store.record(_make_record(session_id="s3", hitl_triggered=False))

    m = store.compute_metrics()
    assert m.hitl_rate == pytest.approx(1 / 3, abs=1e-4)


@pytest.mark.unit
def test_compute_metrics_tier_distribution(tmp_path: Path) -> None:
    log = tmp_path / "sessions.jsonl"
    store = ObservabilityStore(log_path=log)
    store.record(_make_record(session_id="a", offer_tier="LIGHT"))
    store.record(_make_record(session_id="b", offer_tier="STANDARD"))
    store.record(_make_record(session_id="c", offer_tier="STANDARD"))
    store.record(_make_record(session_id="d", offer_tier="PREMIUM"))

    m = store.compute_metrics()
    assert m.tier_distribution == {"LIGHT": 1, "PREMIUM": 1, "STANDARD": 2}
    assert m.total == 4


# ── build_session_record ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_build_session_record_maps_fields() -> None:
    result: dict[str, object] = {
        "propensity": 0.72,
        "cltv": 3500,
        "ev": 250.0,
        "offer_tier": "STANDARD",
        "human_approved": True,
        "security_blocked": False,
        "messages": [],
    }
    record = build_session_record(
        session_id="test-id",
        customer_id="CUST-001",
        phase="analyze",
        latency_ms=750.0,
        result=result,
        hitl_triggered=False,
        messages=[],
        input_tokens=100,
        output_tokens=30,
        cost_usd=0.0002,
    )
    assert record.session_id == "test-id"
    assert record.propensity == pytest.approx(0.72)
    assert record.cltv == 3500
    assert record.offer_tier == "STANDARD"
    assert record.human_approved is True
    assert record.hitl_triggered is False
    assert record.latency_ms == pytest.approx(750.0)


@pytest.mark.unit
def test_build_session_record_handles_none_fields() -> None:
    result: dict[str, object] = {"security_blocked": True}
    record = build_session_record(
        session_id="blocked",
        customer_id="BAD-ID",
        phase="analyze",
        latency_ms=10.0,
        result=result,
        hitl_triggered=False,
        messages=[],
        input_tokens=5,
        output_tokens=5,
        cost_usd=0.00001,
    )
    assert record.propensity is None
    assert record.cltv is None
    assert record.ev is None
    assert record.offer_tier is None
    assert record.security_blocked is True


# ── estimate_session_cost ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_estimate_cost_returns_positive() -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    messages = [
        HumanMessage(content="Analiza el cliente CUST-001"),
        AIMessage(content="Recomendacion: STANDARD con EV=250"),
    ]
    input_tokens, output_tokens, cost = estimate_session_cost(messages)
    assert input_tokens >= 1
    assert output_tokens >= 1
    assert cost > 0.0


@pytest.mark.unit
def test_estimate_cost_empty_messages() -> None:
    input_tokens, output_tokens, cost = estimate_session_cost([])
    assert input_tokens >= 1
    assert output_tokens >= 1
    assert cost > 0.0


@pytest.mark.unit
def test_estimate_cost_ai_message_counts_as_output() -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    only_human = [HumanMessage(content="a" * 400)]
    only_ai = [AIMessage(content="a" * 400)]

    _, out_human, cost_human = estimate_session_cost(only_human)
    in_ai, _, cost_ai = estimate_session_cost(only_ai)

    assert out_human == pytest.approx(1, rel=0.5)
    assert in_ai == pytest.approx(1, rel=0.5)
    assert cost_ai > cost_human

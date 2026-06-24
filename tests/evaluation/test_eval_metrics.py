"""Tests unitarios de las métricas de evaluación y el gate de CI."""

from __future__ import annotations

import pytest

from churn_agent.evaluation.cases import EvalCase
from churn_agent.evaluation.metrics import (
    GATE_THRESHOLDS,
    EvalMetrics,
    compute_metrics,
    gate_passes,
)
from churn_agent.evaluation.runner import EvalResult


def _make_case(
    case_id: str = "c1",
    expect_tier: str | None = "STANDARD",
    expect_hitl: bool = False,
    expect_blocked: bool = False,
) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        customer_id=f"CUST-{case_id}",
        propensity=0.50,
        cltv=1000,
        expect_tier=expect_tier,
        expect_hitl=expect_hitl,
        approve_hitl=True,
        expect_blocked=expect_blocked,
    )


def _make_result(
    case: EvalCase,
    actual_tier: str | None = "STANDARD",
    actual_hitl: bool = False,
    actual_blocked: bool = False,
    passed: bool = True,
) -> EvalResult:
    return EvalResult(
        case=case,
        actual_tier=actual_tier,
        actual_hitl=actual_hitl,
        actual_blocked=actual_blocked,
        passed=passed,
    )


class TestComputeMetrics:
    @pytest.mark.unit
    def test_all_pass_perfect_metrics(self) -> None:
        case_tier = _make_case("t1", expect_tier="STANDARD")
        case_hitl = _make_case("t2", expect_hitl=True)
        case_blocked = _make_case("t3", expect_blocked=True, expect_tier=None)

        results = [
            _make_result(case_tier, actual_tier="STANDARD", passed=True),
            _make_result(case_hitl, actual_hitl=True, passed=True),
            _make_result(case_blocked, actual_blocked=True, passed=True),
        ]
        m = compute_metrics(results)

        assert m.total == 3
        assert m.passed == 3
        assert m.pass_rate == pytest.approx(1.0)
        assert m.tier_accuracy == pytest.approx(1.0)
        assert m.hitl_recall == pytest.approx(1.0)
        assert m.block_rate == pytest.approx(1.0)

    @pytest.mark.unit
    def test_tier_accuracy_partial(self) -> None:
        cases = [_make_case(f"t{i}", expect_tier="STANDARD") for i in range(4)]
        results = [
            _make_result(cases[0], actual_tier="STANDARD", passed=True),
            _make_result(cases[1], actual_tier="STANDARD", passed=True),
            _make_result(cases[2], actual_tier="LIGHT", passed=False),
            _make_result(cases[3], actual_tier="LIGHT", passed=False),
        ]
        m = compute_metrics(results)
        assert m.tier_accuracy == pytest.approx(0.5)
        assert m.pass_rate == pytest.approx(0.5)

    @pytest.mark.unit
    def test_hitl_recall_zero_when_never_triggered(self) -> None:
        case = _make_case("h1", expect_hitl=True)
        results = [_make_result(case, actual_hitl=False, passed=False)]
        m = compute_metrics(results)
        assert m.hitl_recall == pytest.approx(0.0)

    @pytest.mark.unit
    def test_block_rate_zero_when_never_blocked(self) -> None:
        case = _make_case("b1", expect_blocked=True, expect_tier=None)
        results = [_make_result(case, actual_blocked=False, passed=False)]
        m = compute_metrics(results)
        assert m.block_rate == pytest.approx(0.0)

    @pytest.mark.unit
    def test_empty_suite_returns_neutral_metrics(self) -> None:
        m = compute_metrics([])
        assert m.total == 0
        assert m.pass_rate == pytest.approx(0.0)
        assert m.tier_accuracy == pytest.approx(1.0)
        assert m.hitl_recall == pytest.approx(1.0)
        assert m.block_rate == pytest.approx(1.0)

    @pytest.mark.unit
    def test_str_representation(self) -> None:
        m = EvalMetrics(
            total=5,
            passed=4,
            pass_rate=0.8,
            tier_accuracy=1.0,
            hitl_recall=1.0,
            block_rate=1.0,
        )
        text = str(m)
        assert "passed" in text
        assert "80%" in text


class TestGatePasses:
    @pytest.mark.unit
    def test_gate_passes_all_perfect(self) -> None:
        m = EvalMetrics(
            total=5,
            passed=5,
            pass_rate=1.0,
            tier_accuracy=1.0,
            hitl_recall=1.0,
            block_rate=1.0,
        )
        ok, failures = gate_passes(m)
        assert ok is True
        assert failures == []

    @pytest.mark.unit
    def test_gate_fails_on_low_pass_rate(self) -> None:
        m = EvalMetrics(
            total=5,
            passed=3,
            pass_rate=0.60,
            tier_accuracy=1.0,
            hitl_recall=1.0,
            block_rate=1.0,
        )
        ok, failures = gate_passes(m)
        assert ok is False
        assert any("pass_rate" in f for f in failures)

    @pytest.mark.unit
    def test_gate_fails_on_hitl_recall_below_1(self) -> None:
        m = EvalMetrics(
            total=5,
            passed=4,
            pass_rate=0.80,
            tier_accuracy=1.0,
            hitl_recall=0.50,
            block_rate=1.0,
        )
        ok, failures = gate_passes(m)
        assert ok is False
        assert any("hitl_recall" in f for f in failures)

    @pytest.mark.unit
    def test_gate_fails_on_block_rate_below_1(self) -> None:
        m = EvalMetrics(
            total=5,
            passed=4,
            pass_rate=0.80,
            tier_accuracy=1.0,
            hitl_recall=1.0,
            block_rate=0.0,
        )
        ok, failures = gate_passes(m)
        assert ok is False
        assert any("block_rate" in f for f in failures)

    @pytest.mark.unit
    def test_gate_thresholds_are_defined(self) -> None:
        assert "pass_rate" in GATE_THRESHOLDS
        assert "tier_accuracy" in GATE_THRESHOLDS
        assert "hitl_recall" in GATE_THRESHOLDS
        assert "block_rate" in GATE_THRESHOLDS

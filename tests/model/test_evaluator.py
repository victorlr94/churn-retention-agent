"""Tests del evaluador de métricas."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from churn_agent.model.evaluator import compare_models, compute_metrics
from churn_agent.model.trainer import train_lgbm


@pytest.mark.unit
def test_compute_metrics_returns_expected_keys(
    small_X: pd.DataFrame,
    small_y: pd.Series[Any],
) -> None:
    model = train_lgbm(small_X, small_y, params={"n_estimators": 50})
    metrics = compute_metrics(model, small_X, small_y)
    expected = {"auc_roc", "auc_pr", "brier", "recall_decil1", "lift_decil1"}
    assert set(metrics.keys()) == expected


@pytest.mark.unit
def test_metrics_are_in_valid_range(
    small_X: pd.DataFrame,
    small_y: pd.Series[Any],
) -> None:
    model = train_lgbm(small_X, small_y, params={"n_estimators": 50})
    m = compute_metrics(model, small_X, small_y)
    assert 0.0 <= m["auc_roc"] <= 1.0
    assert 0.0 <= m["auc_pr"] <= 1.0
    assert 0.0 <= m["brier"] <= 1.0
    assert 0.0 <= m["recall_decil1"] <= 1.0
    assert m["lift_decil1"] >= 0.0


@pytest.mark.unit
def test_lift_is_ratio_of_recall_to_base_rate(
    small_X: pd.DataFrame,
    small_y: pd.Series[Any],
) -> None:
    model = train_lgbm(small_X, small_y, params={"n_estimators": 50})
    m = compute_metrics(model, small_X, small_y)
    base_rate = float(small_y.mean())
    expected_lift = m["recall_decil1"] / base_rate
    assert abs(m["lift_decil1"] - expected_lift) < 1e-6


@pytest.mark.unit
def test_compare_models_returns_dataframe(
    small_X: pd.DataFrame,
    small_y: pd.Series[Any],
) -> None:
    model_a = train_lgbm(small_X, small_y, params={"n_estimators": 30})
    model_b = train_lgbm(small_X, small_y, params={"n_estimators": 50})
    table = compare_models({"lgbm_30": model_a, "lgbm_50": model_b}, small_X, small_y)
    assert isinstance(table, pd.DataFrame)
    assert set(table.index) == {"lgbm_30", "lgbm_50"}
    assert "auc_pr" in table.columns

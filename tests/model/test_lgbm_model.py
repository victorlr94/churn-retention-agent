"""Tests del adaptador LightGBMChurnModel -> ChurnModel Protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from churn_agent.core.interfaces import ChurnModel
from churn_agent.model.lgbm_model import LightGBMChurnModel
from churn_agent.model.trainer import train_lgbm


def _trained_adapter(X: pd.DataFrame, y: pd.Series[Any]) -> LightGBMChurnModel:
    pipeline = train_lgbm(X, y, params={"n_estimators": 30})
    return LightGBMChurnModel(model=pipeline, feature_names=list(X.columns))


def _row_as_features(X: pd.DataFrame, i: int) -> dict[str, object]:
    return {str(k): v for k, v in X.iloc[i].to_dict().items()}


@pytest.mark.unit
def test_lgbm_model_implements_protocol(
    small_X: pd.DataFrame,
    small_y: pd.Series[Any],
) -> None:
    adapter = _trained_adapter(small_X, small_y)
    assert isinstance(adapter, ChurnModel)


@pytest.mark.unit
def test_predict_proba_single_customer(
    small_X: pd.DataFrame,
    small_y: pd.Series[Any],
) -> None:
    adapter = _trained_adapter(small_X, small_y)
    customer = _row_as_features(small_X, 0)
    prob = adapter.predict_proba(customer)
    assert 0.0 <= prob <= 1.0


@pytest.mark.unit
def test_predict_proba_missing_features_handled(
    small_X: pd.DataFrame,
    small_y: pd.Series[Any],
) -> None:
    adapter = _trained_adapter(small_X, small_y)
    incomplete: dict[str, object] = {"Tenure Months": 12, "Monthly Charges": 75.0}
    prob = adapter.predict_proba(incomplete)
    assert 0.0 <= prob <= 1.0


@pytest.mark.unit
def test_save_and_load_roundtrip(
    tmp_path: Path,
    small_X: pd.DataFrame,
    small_y: pd.Series[Any],
) -> None:
    adapter = _trained_adapter(small_X, small_y)
    path = tmp_path / "model.pkl"
    adapter.save(path)
    loaded = LightGBMChurnModel.load(path)
    customer = _row_as_features(small_X, 0)
    assert abs(adapter.predict_proba(customer) - loaded.predict_proba(customer)) < 1e-9

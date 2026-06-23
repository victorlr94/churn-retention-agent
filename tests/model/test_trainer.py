"""Tests del entrenamiento LightGBM y calibracion."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from churn_agent.model.trainer import calibrate, train_lgbm


@pytest.mark.unit
def test_lgbm_pipeline_fits_and_predicts(
    small_X: pd.DataFrame,
    small_y: pd.Series[Any],
) -> None:
    pipeline = train_lgbm(small_X, small_y)
    probas = pipeline.predict_proba(small_X)
    assert probas.shape == (len(small_X), 2)
    assert ((probas >= 0) & (probas <= 1)).all()


@pytest.mark.unit
def test_lgbm_probas_are_valid_distribution(
    small_X: pd.DataFrame,
    small_y: pd.Series[Any],
) -> None:
    pipeline = train_lgbm(small_X, small_y)
    probas = pipeline.predict_proba(small_X)
    row_sums = probas.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


@pytest.mark.unit
def test_lgbm_accepts_custom_params(
    small_X: pd.DataFrame,
    small_y: pd.Series[Any],
) -> None:
    pipeline = train_lgbm(small_X, small_y, params={"n_estimators": 50})
    assert pipeline.named_steps["clf"].n_estimators == 50


@pytest.mark.unit
def test_calibration_produces_valid_probas(
    train_val_split: tuple[pd.DataFrame, pd.DataFrame, pd.Series[Any], pd.Series[Any]],
) -> None:
    X_train, X_val, y_train, y_val = train_val_split
    pipeline = train_lgbm(X_train, y_train, params={"n_estimators": 50})
    cal = calibrate(pipeline, X_val, y_val, method="isotonic")
    probas = cal.predict_proba(X_val)
    assert probas.shape == (len(X_val), 2)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.unit
def test_calibrated_model_has_predict_proba(
    train_val_split: tuple[pd.DataFrame, pd.DataFrame, pd.Series[Any], pd.Series[Any]],
) -> None:
    X_train, X_val, y_train, y_val = train_val_split
    pipeline = train_lgbm(X_train, y_train, params={"n_estimators": 50})
    cal = calibrate(pipeline, X_val, y_val)
    assert hasattr(cal, "predict_proba")
    out = cal.predict_proba(X_val)
    assert out.shape == (len(X_val), 2)


@pytest.mark.unit
def test_calibrate_rejects_unknown_method(
    train_val_split: tuple[pd.DataFrame, pd.DataFrame, pd.Series[Any], pd.Series[Any]],
) -> None:
    X_train, X_val, y_train, y_val = train_val_split
    pipeline = train_lgbm(X_train, y_train, params={"n_estimators": 50})
    with pytest.raises(ValueError, match="isotonic"):
        calibrate(pipeline, X_val, y_val, method="sigmoid")

"""Tests de los modelos baseline."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from churn_agent.model.baseline import train_logistic_baseline, train_majority_baseline


@pytest.mark.unit
def test_majority_baseline_predicts_proba(
    small_X: pd.DataFrame,
    small_y: pd.Series[Any],
) -> None:
    model = train_majority_baseline(small_y)
    dummy_X = pd.DataFrame({"_": range(len(small_X))})
    probas = model.predict_proba(dummy_X)
    assert probas.shape == (len(small_X), 2)
    # Todas las predicciones son iguales (prior)
    assert (probas[:, 1] == probas[0, 1]).all()


@pytest.mark.unit
def test_majority_baseline_proba_matches_churn_rate(
    small_y: pd.Series[Any],
) -> None:
    model = train_majority_baseline(small_y)
    dummy_X = pd.DataFrame({"_": [0]})
    proba = model.predict_proba(dummy_X)[0, 1]
    assert abs(proba - float(small_y.mean())) < 0.05


@pytest.mark.unit
def test_logistic_baseline_fits_and_predicts(
    small_X: pd.DataFrame,
    small_y: pd.Series[Any],
) -> None:
    model = train_logistic_baseline(small_X, small_y)
    probas = model.predict_proba(small_X)
    assert probas.shape == (len(small_X), 2)
    assert ((probas >= 0) & (probas <= 1)).all()


@pytest.mark.unit
def test_logistic_baseline_produces_varying_probabilities(
    small_X: pd.DataFrame,
    small_y: pd.Series[Any],
) -> None:
    model = train_logistic_baseline(small_X, small_y)
    probas = model.predict_proba(small_X)[:, 1]
    # Logistic debe producir probabilidades variadas (no constantes como majority)
    assert float(probas.std()) > 0.01

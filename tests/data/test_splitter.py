"""Tests del splitter estratificado."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from churn_agent.data.splitter import make_split


def _make_imbalanced_dataset(
    n: int = 500, churn_rate: float = 0.27
) -> tuple[pd.DataFrame, pd.Series[Any]]:
    """Dataset sintético con distribución de churn similar al dataset real (~27%)."""
    rng = np.random.default_rng(0)
    y_vals = rng.choice([0, 1], size=n, p=[1 - churn_rate, churn_rate])
    X = pd.DataFrame({"feature_a": rng.random(n), "feature_b": rng.random(n)})
    y: pd.Series[Any] = pd.Series(y_vals, dtype=int)
    return X, y


@pytest.mark.unit
def test_split_sizes() -> None:
    X, y = _make_imbalanced_dataset()
    X_train, X_test, y_train, y_test = make_split(X, y, test_size=0.20, seed=42)
    assert len(X_train) + len(X_test) == len(X)
    assert abs(len(X_test) / len(X) - 0.20) < 0.01


@pytest.mark.unit
def test_split_is_stratified() -> None:
    """La tasa de churn en train y test debe ser similar a la del dataset original."""
    X, y = _make_imbalanced_dataset(n=1000, churn_rate=0.27)
    _, _, y_train, y_test = make_split(X, y, test_size=0.20, seed=42)
    original_rate = float(y.mean())
    assert abs(float(y_train.mean()) - original_rate) < 0.03
    assert abs(float(y_test.mean()) - original_rate) < 0.03


@pytest.mark.unit
def test_split_is_reproducible() -> None:
    """Dos splits con la misma semilla producen los mismos índices."""
    X, y = _make_imbalanced_dataset()
    X_train_a, X_test_a, _, _ = make_split(X, y, seed=42)
    X_train_b, X_test_b, _, _ = make_split(X, y, seed=42)
    pd.testing.assert_frame_equal(X_train_a, X_train_b)
    pd.testing.assert_frame_equal(X_test_a, X_test_b)


@pytest.mark.unit
def test_split_different_seeds_differ() -> None:
    """Semillas distintas producen splits distintos."""
    X, y = _make_imbalanced_dataset()
    X_train_a, _, _, _ = make_split(X, y, seed=42)
    X_train_b, _, _, _ = make_split(X, y, seed=99)
    assert not X_train_a.index.equals(X_train_b.index)

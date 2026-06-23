"""Tests del preprocesador de features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_agent.model.preprocessor import (
    HIGH_CARDINALITY_THRESHOLD,
    build_preprocessor,
)


@pytest.mark.unit
def test_preprocessor_transforms_without_error(
    small_X: pd.DataFrame,
) -> None:
    pre = build_preprocessor(small_X)
    X_t = pre.fit_transform(small_X)
    assert X_t is not None
    assert X_t.shape[0] == len(small_X)


@pytest.mark.unit
def test_numeric_columns_are_scaled(small_X: pd.DataFrame) -> None:
    pre = build_preprocessor(small_X)
    X_t: np.ndarray = pre.fit_transform(small_X)
    # Columnas numéricas escaladas deben tener media ≈ 0 y std ≈ 1
    num_idx = pre.output_indices_["num"]
    numeric_block = X_t[:, num_idx]
    assert abs(numeric_block.mean()) < 0.5


@pytest.mark.unit
def test_high_cardinality_columns_dropped() -> None:
    # n debe ser > threshold para garantizar que all rows son distintas
    n = HIGH_CARDINALITY_THRESHOLD + 10
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {
            "numeric": rng.random(n),
            "low_card": rng.choice(["a", "b", "c"], n),
            # columna con mas de HIGH_CARDINALITY_THRESHOLD categorias distintas
            "high_card": [f"city_{i}" for i in range(n)],
        }
    )
    assert X["high_card"].nunique() > HIGH_CARDINALITY_THRESHOLD
    pre = build_preprocessor(X)
    pre.fit_transform(X)
    # high_card debería quedar fuera (remainder="drop")
    feature_names = list(pre.get_feature_names_out())
    assert not any("high_card" in f for f in feature_names)


@pytest.mark.unit
def test_missing_values_imputed(small_X: pd.DataFrame) -> None:
    X_with_nan = small_X.copy()
    X_with_nan.loc[0, "Monthly Charges"] = float("nan")
    X_with_nan.loc[1, "Contract"] = None
    pre = build_preprocessor(small_X)
    pre.fit(small_X)
    X_t: np.ndarray = pre.transform(X_with_nan)
    assert not np.isnan(X_t).any()


@pytest.mark.unit
def test_unknown_category_handled(small_X: pd.DataFrame) -> None:
    pre = build_preprocessor(small_X)
    pre.fit(small_X)
    X_unseen = small_X.copy()
    X_unseen.loc[0, "Contract"] = "Quarterly"  # categoría nueva
    X_t: np.ndarray = pre.transform(X_unseen)
    assert not np.isnan(X_t).any()

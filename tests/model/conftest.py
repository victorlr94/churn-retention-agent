"""Fixtures compartidas para los tests del módulo model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest


def _make_X(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Tenure Months": rng.integers(0, 72, n),
            "Monthly Charges": rng.uniform(20, 120, n),
            "Total Charges": rng.uniform(0, 8000, n),
            "CLTV": rng.integers(2000, 8000, n),
            "Satisfaction Score": rng.integers(1, 6, n),
            "Senior Citizen": rng.integers(0, 2, n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "Internet Service": rng.choice(["DSL", "Fiber optic", "No"], n),
            "Payment Method": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
            ),
            "Partner": rng.choice(["Yes", "No"], n),
        }
    )


def _make_y(n: int = 200, churn_rate: float = 0.27, seed: int = 0) -> pd.Series[Any]:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.choice([0, 1], size=n, p=[1 - churn_rate, churn_rate]),
        dtype=int,
    )


@pytest.fixture()
def small_X() -> pd.DataFrame:
    return _make_X(n=200)


@pytest.fixture()
def small_y() -> pd.Series[Any]:
    return _make_y(n=200)


@pytest.fixture()
def train_val_split(
    small_X: pd.DataFrame,
    small_y: pd.Series[Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series[Any], pd.Series[Any]]:
    n = len(small_X)
    split = int(n * 0.75)
    return (
        small_X.iloc[:split].reset_index(drop=True),
        small_X.iloc[split:].reset_index(drop=True),
        small_y.iloc[:split].reset_index(drop=True),
        small_y.iloc[split:].reset_index(drop=True),
    )

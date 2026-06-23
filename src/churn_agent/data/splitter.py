"""Split estratificado y reproducible del dataset.

El split se hace SIEMPRE estratificado por el target (Churn Value) para
preservar la distribución de clase en datasets desbalanceados.
La semilla proviene de Settings para garantizar reproducibilidad entre runs.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_agent.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_TEST_SIZE = 0.20


def make_split(
    X: pd.DataFrame,
    y: pd.Series[Any],
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series[Any], pd.Series[Any]]:
    """Divide X e y en train/test estratificados.

    Args:
        X: Features (sin leakage ni target).
        y: Target binario (Churn Value).
        test_size: Fracción del conjunto de test. Default 0.20.
        seed: Semilla aleatoria. Si es None, usa settings.random_seed.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    if seed is None:
        seed = get_settings().random_seed

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    churn_train = float(y_train.mean()) * 100
    churn_test = float(y_test.mean()) * 100

    logger.info(
        "Split: train=%d (churn %.1f%%) | test=%d (churn %.1f%%)",
        len(X_train),
        churn_train,
        len(X_test),
        churn_test,
    )
    return X_train, X_test, y_train, y_test

"""Entrenamiento del modelo LightGBM y calibración de probabilidades.

Flujo:
  1. build_preprocessor(X_train)  → ColumnTransformer
  2. Pipeline([pre, LGBMClassifier]) → fit sobre train
  3. calibrate(pipeline, X_val, y_val) → _CalibratedModel

La calibración isotónica corrige la tendencia de LGBM a producir scores
que no son probabilidades verdaderas (especialmente con class_weight="balanced").

Nota: CalibratedClassifierCV(cv="prefit") fue eliminado en sklearn ≥1.6.
Se usa un wrapper propio (_CalibratedModel) que aplica la calibración
como una etapa de post-proceso separada, sin re-entrenar el estimador base.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

from churn_agent.model.preprocessor import build_preprocessor

LGBM_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


class _CalibratedModel:
    """Pipeline + calibrador isotónico como una unidad cohesiva."""

    def __init__(self, pipeline: Pipeline, calibrator: IsotonicRegression) -> None:
        self._pipeline = pipeline
        self._calibrator = calibrator

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw: np.ndarray = self._pipeline.predict_proba(X)[:, 1]
        cal: np.ndarray = np.clip(self._calibrator.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series[Any],
    params: dict[str, Any] | None = None,
) -> Pipeline:
    """Entrena un Pipeline [preprocessor → LGBMClassifier] sobre (X_train, y_train)."""
    merged: dict[str, Any] = {**LGBM_DEFAULT_PARAMS, **(params or {})}
    pipeline: Pipeline = Pipeline(
        [
            ("pre", build_preprocessor(X_train)),
            ("clf", LGBMClassifier(**merged)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def calibrate(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series[Any],
    method: str = "isotonic",
) -> _CalibratedModel:
    """Calibra un pipeline ya entrenado usando un conjunto de validación separado.

    Solo soporta method="isotonic" (IsotonicRegression). Se aplica sobre las
    probabilidades crudas del pipeline en X_val, sin re-entrenar el estimador base.
    """
    if method != "isotonic":
        raise ValueError(f"method debe ser 'isotonic', recibido: {method!r}")

    raw_val: np.ndarray = pipeline.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val, y_val)
    return _CalibratedModel(pipeline=pipeline, calibrator=calibrator)

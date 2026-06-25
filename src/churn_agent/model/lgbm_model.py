"""Adaptador LightGBMChurnModel → ChurnModel Protocol.

El Protocol define predict_proba(features: dict) → float para uso
single-customer del agente. Este adaptador envuelve el Pipeline/
CalibratedClassifierCV sklearn y convierte el dict de entrada en un
DataFrame de una fila con las columnas esperadas.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import joblib
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline

    _Model = Pipeline | CalibratedClassifierCV


class LightGBMChurnModel:
    """Adapter: sklearn Pipeline/CalibratedClassifierCV → ChurnModel Protocol."""

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
    ) -> None:
        self._model = model
        self._feature_names = feature_names

    def predict_proba(self, features: dict[str, object]) -> float:
        """Devuelve P(churn=1) para un cliente individual."""
        row = pd.DataFrame([features])
        for col in self._feature_names:
            if col not in row.columns:
                row[col] = np.nan
        row = row[self._feature_names]
        proba: float = float(self._model.predict_proba(row)[0, 1])
        return proba

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> LightGBMChurnModel:
        return cast(LightGBMChurnModel, joblib.load(path))

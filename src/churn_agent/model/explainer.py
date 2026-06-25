"""Drivers SHAP para explicabilidad del modelo LightGBM.

SHAP (SHapley Additive exPlanations) cuantifica la contribución de cada
feature a la predicción individual. Aquí lo usamos para:
- Ranking global de importancia (mean |SHAP|) → reportado como parte del
  reporte de métricas de Fase 2.
- Drivers individuales por cliente → insumo para el agente en Fase 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def get_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Calcula SHAP values del LGBMClassifier dentro del pipeline.

    Devuelve (shap_matrix, feature_names) donde shap_matrix tiene forma
    (n_samples, n_features_transformadas).
    """
    preprocessor = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]

    X_transformed: np.ndarray = preprocessor.transform(X)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)

    # Para clasificación binaria LGBM devuelve list de 2; tomamos clase positiva.
    if isinstance(shap_values, list):
        shap_matrix: np.ndarray = shap_values[1]
    else:
        shap_matrix = shap_values

    feature_names: list[str] = list(preprocessor.get_feature_names_out())
    return shap_matrix, feature_names


def top_drivers(
    pipeline: Pipeline,
    X: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Ranking global de los n features más importantes (mean |SHAP|)."""
    shap_matrix, feature_names = get_shap_values(pipeline, X)
    importance = np.abs(shap_matrix).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
    return (
        df.sort_values("mean_abs_shap", ascending=False).head(n).reset_index(drop=True)
    )

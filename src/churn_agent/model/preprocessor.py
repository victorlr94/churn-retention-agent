"""Preprocesador de features para el modelo de propensión.

Auto-detecta columnas numéricas y categóricas del DataFrame de entrada,
descarta las de alta cardinalidad (p. ej. City con cientos de ciudades)
e imputa valores faltantes antes de encodear.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

HIGH_CARDINALITY_THRESHOLD = 50


def _detect_columns(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Devuelve (num_cols, cat_cols) tras inspeccionar dtypes y cardinalidad."""
    num_cols: list[str] = []
    cat_cols: list[str] = []

    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            num_cols.append(col)
        else:
            coerced = pd.to_numeric(series, errors="coerce")
            if coerced.notna().mean() > 0.5:
                # mayoría de valores se convierten → es numérica mal tipada
                num_cols.append(col)
            elif series.nunique() <= HIGH_CARDINALITY_THRESHOLD:
                cat_cols.append(col)
            # alta cardinalidad → se descarta (remainder="drop")

    return num_cols, cat_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construye y devuelve (sin fit) un ColumnTransformer para X.

    Se llama con X_train; el resultado se encadena en un Pipeline antes
    de hacer fit, lo que garantiza que la transformación no ve el test set.
    """
    num_cols, cat_cols = _detect_columns(X)

    num_pipeline: Pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipeline: Pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )

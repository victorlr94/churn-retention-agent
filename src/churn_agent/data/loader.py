"""Carga y validación del dataset IBM Telco Customer Churn.

Flujo:
  1. Lee el CSV crudo.
  2. Detecta y bloquea columnas con leakage (lanza LeakageError si las encuentra).
  3. Valida el esquema del CSV con pandera.
  4. Devuelve (X, y) limpios, listos para el splitter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from churn_agent.config import get_settings
from churn_agent.exceptions import DataValidationError, LeakageError

from .schema import (
    COL_CHURN_VALUE,
    EXCLUDED_FROM_FEATURES,
    LEAKAGE_COLUMNS,
    TelcoRawSchema,
)

logger = logging.getLogger(__name__)


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Lee el CSV crudo sin ninguna transformación ni validación de leakage.

    Args:
        path: Ruta al CSV. Si es None, usa settings.data_raw_dir / filename.

    Returns:
        DataFrame con todas las columnas originales, incluyendo leakage
        (que se excluye en load_features, no aquí).
    """
    if path is None:
        settings = get_settings()
        path = settings.data_raw_dir / "telco_customer_churn.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado en {path}. "
            "Sigue las instrucciones en scripts/download_data.py."
        )

    df = pd.read_csv(path)
    logger.info(
        "Dataset cargado: %d filas, %d columnas — %s",
        len(df),
        len(df.columns),
        path.name,
    )
    return df


def check_no_leakage(df: pd.DataFrame) -> None:
    """Verifica que el DataFrame no contenga columnas con leakage.

    Se debe llamar ANTES de cualquier preprocesamiento o entrenamiento.

    Args:
        df: DataFrame a inspeccionar (puede ser X o el DataFrame completo).

    Raises:
        LeakageError: Si alguna columna de LEAKAGE_COLUMNS está presente.
    """
    found = LEAKAGE_COLUMNS & set(df.columns)
    if found:
        raise LeakageError(
            f"Leakage detectado: {sorted(found)}. "
            "Estas columnas están derivadas del target y no pueden "
            "usarse como features. Consulta ADR-0002 y docs/DATA_CARD.md."
        )


def load_features(
    path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series[Any]]:
    """Carga el dataset y devuelve (X, y) libres de leakage.

    Pasos:
      1. Carga el CSV crudo.
      2. Valida el esquema mínimo con pandera.
      3. Verifica explícitamente que no hay leakage en X.
      4. Separa features (X) del target (y).

    Args:
        path: Ruta opcional al CSV (ver load_raw).

    Returns:
        X: DataFrame de features (sin columnas de leakage ni target).
        y: Series binaria del target (Churn Value: 0 o 1).
    """
    df = load_raw(path)

    # Valida el esquema del CSV crudo (pandera lanza SchemaErrors con lazy=True)
    try:
        TelcoRawSchema.validate(df, lazy=True)
    except Exception as exc:
        raise DataValidationError(
            f"El CSV no cumple el esquema esperado: {exc}"
        ) from exc

    # Separar target antes de construir X
    if COL_CHURN_VALUE not in df.columns:
        raise DataValidationError(f"Columna target '{COL_CHURN_VALUE}' no encontrada.")

    y: pd.Series[Any] = df[COL_CHURN_VALUE].astype(int)

    # Construir X excluyendo leakage, target e ID
    cols_to_drop = EXCLUDED_FROM_FEATURES & set(df.columns)
    X = df.drop(columns=list(cols_to_drop))

    # Guardia explícita: si leakage sobrevivió a la exclusión, falla ruidosamente
    check_no_leakage(X)

    logger.info(
        "Features listas: %d filas, %d columnas. Churn rate: %.1f%%",
        len(X),
        len(X.columns),
        y.mean() * 100,
    )
    return X, y

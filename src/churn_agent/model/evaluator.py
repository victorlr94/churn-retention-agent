"""Métricas de evaluación del modelo de propensión al churn.

Métricas elegidas para datasets desbalanceados en retención:
- AUC-ROC: discriminación global (referencia).
- AUC-PR: área bajo curva Precision-Recall (más informativa en desbalance).
- Brier score: calibración (qué tan cerca están las probs del resultado real).
- Recall@decil1: de los top-10% clientes ordenados por riesgo, ¿qué % son
  churners reales?
- Lift@decil1: ratio recall@decil1 / tasa_base — cuánto mejor que aleatorio.

Accuracy deliberadamente ausente: con ~27% churn, predecir siempre 0 da 73% accuracy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

MetricsDict = dict[str, float]


def _recall_at_decile(
    y_true: pd.Series[Any],
    y_proba: np.ndarray,
    decile: int = 1,
) -> float:
    n = len(y_true)
    top_k = max(1, int(n * decile / 10))
    idx = np.argsort(y_proba)[::-1][:top_k]
    return float(np.array(y_true)[idx].mean())


def compute_metrics(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series[Any],
) -> MetricsDict:
    """Calcula el conjunto estándar de métricas sobre (X, y)."""
    y_proba: np.ndarray = model.predict_proba(X)[:, 1]
    base_rate = float(y.mean())

    recall_d1 = _recall_at_decile(y, y_proba, decile=1)
    lift_d1 = recall_d1 / base_rate if base_rate > 0 else 0.0

    return {
        "auc_roc": float(roc_auc_score(y, y_proba)),
        "auc_pr": float(average_precision_score(y, y_proba)),
        "brier": float(brier_score_loss(y, y_proba)),
        "recall_decil1": recall_d1,
        "lift_decil1": lift_d1,
    }


def compare_models(
    models: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series[Any],
) -> pd.DataFrame:
    """Tabla comparativa de métricas para un dict {nombre: modelo}."""
    rows = []
    for name, model in models.items():
        metrics = compute_metrics(model, X, y)
        rows.append({"model": name, **metrics})
    return pd.DataFrame(rows).set_index("model").round(4)

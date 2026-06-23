"""Modelos baseline para comparación con LightGBM.

Dos baselines:
- MajorityBaseline: siempre predice la clase mayoritaria (p = churn_rate).
- LogisticBaseline: regresión logística con preprocesador estándar.

Sin baseline sólido no hay forma de saber si el modelo "aprende algo".
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn_agent.model.preprocessor import build_preprocessor


def train_majority_baseline(y_train: pd.Series[Any]) -> DummyClassifier:
    clf = DummyClassifier(strategy="prior", random_state=0)
    clf.fit(pd.DataFrame({"_": range(len(y_train))}), y_train)
    return clf


def train_logistic_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series[Any],
) -> Pipeline:
    pipeline: Pipeline = Pipeline(
        [
            ("pre", build_preprocessor(X_train)),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=0,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline

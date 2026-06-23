"""Modelo de propensión: entrenamiento, calibración y drivers SHAP (Fase 2)."""

from churn_agent.model.baseline import train_logistic_baseline, train_majority_baseline
from churn_agent.model.evaluator import MetricsDict, compare_models, compute_metrics
from churn_agent.model.explainer import top_drivers
from churn_agent.model.lgbm_model import LightGBMChurnModel
from churn_agent.model.preprocessor import build_preprocessor
from churn_agent.model.trainer import calibrate, train_lgbm

__all__ = [
    "MetricsDict",
    "LightGBMChurnModel",
    "build_preprocessor",
    "calibrate",
    "compare_models",
    "compute_metrics",
    "top_drivers",
    "train_lgbm",
    "train_logistic_baseline",
    "train_majority_baseline",
]

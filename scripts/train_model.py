"""Script de entrenamiento end-to-end del modelo de propensión al churn.

Uso:
    uv run python scripts/train_model.py [--data-path PATH] [--output-dir DIR]

Produce:
    models/lgbm_churn_calibrated.pkl  — LightGBMChurnModel listo para el agente
    models/metrics_report.txt         — tabla comparativa de métricas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de propensión al churn."
    )
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Churn Retention Agent — Fase 2: Entrenamiento ===\n")

    # 1. Carga y split -------------------------------------------------------
    print("[1/5] Cargando datos...")
    from churn_agent.data.loader import load_features
    from churn_agent.data.splitter import make_split

    X, y = load_features(args.data_path)
    print(
        f"      {len(X)} clientes · {X.shape[1]} features · churn rate: {y.mean():.1%}"
    )

    # Split train 60% / val 20% / test 20%
    X_trainval, X_test, y_trainval, y_test = make_split(X, y, test_size=0.20)
    X_train, X_val, y_train, y_val = make_split(X_trainval, y_trainval, test_size=0.25)
    print(f"      train={len(X_train)} · val={len(X_val)} · test={len(X_test)}")

    # 2. Baselines -----------------------------------------------------------
    print("\n[2/5] Entrenando baselines...")
    from churn_agent.model.baseline import (
        train_logistic_baseline,
        train_majority_baseline,
    )

    logistic = train_logistic_baseline(X_train, y_train)
    # majority baseline: AUC-ROC = 0.5 por definición (score constante)
    majority_rate = float(
        train_majority_baseline(y_train).predict_proba(
            __import__("pandas").DataFrame({"_": [0]})
        )[0, 1]
    )
    print(f"      Majority churn rate: {majority_rate:.1%} | Logistic: OK")

    # 3. LightGBM + calibración ----------------------------------------------
    print("\n[3/5] Entrenando LightGBM...")
    from churn_agent.model.trainer import calibrate, train_lgbm

    lgbm_pipeline = train_lgbm(X_train, y_train)
    print("      Calibrando (isotonic) sobre val...")
    lgbm_calibrated = calibrate(lgbm_pipeline, X_val, y_val, method="isotonic")

    # 4. Evaluación ----------------------------------------------------------
    print("\n[4/5] Evaluando sobre test set...")
    from churn_agent.model.evaluator import compare_models

    models = {
        "Logistic baseline": logistic,
        "LGBM (sin calibrar)": lgbm_pipeline,
        "LGBM (calibrado)": lgbm_calibrated,
    }
    table = compare_models(models, X_test, y_test)
    print(table.to_string())

    report_path = output_dir / "metrics_report.txt"
    report_path.write_text(table.to_string())
    print(f"\n      Reporte guardado en: {report_path}")

    # 5. Guardar modelo ------------------------------------------------------
    print("\n[5/5] Guardando modelo...")
    from churn_agent.model.lgbm_model import LightGBMChurnModel

    adapter = LightGBMChurnModel(
        model=lgbm_calibrated,
        feature_names=list(X_train.columns),
    )
    model_path = output_dir / "lgbm_churn_calibrated.pkl"
    adapter.save(model_path)
    print(f"      Modelo guardado en: {model_path}")
    print("\n=== Entrenamiento completo ===")


if __name__ == "__main__":
    main()

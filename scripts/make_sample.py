"""Genera el dataset de muestra y el modelo demo para la Fase 7.

Uso (una sola vez en local; los artefactos se commitean):
    uv run python scripts/make_sample.py [--data-path PATH] [--n-rows 500]

Produce:
    data/sample/telco_sample.csv   — 500 filas estratificadas por Churn Value
    models/demo/lgbm_demo.pkl      — LightGBMChurnModel entrenado sobre la muestra
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera el dataset de muestra y el modelo demo."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Ruta al CSV completo (default: data/raw/telco_customer_churn.csv).",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=500,
        help="Número de filas en el sample (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== Churn Retention Agent — Fase 7: Generación de artefactos demo ===\n")

    # 1. Cargar CSV completo ---------------------------------------------------
    print("[1/4] Cargando dataset completo...")
    from churn_agent.data.loader import load_raw

    df_full = load_raw(args.data_path)
    churn_col = "Churn Value"
    print(f"      {len(df_full)} filas · churn rate: {df_full[churn_col].mean():.1%}")

    # 2. Muestra estratificada -------------------------------------------------
    print(f"\n[2/4] Muestreando {args.n_rows} filas (estratificado por {churn_col})...")
    from sklearn.model_selection import train_test_split

    _, df_sample = train_test_split(
        df_full,
        test_size=args.n_rows / len(df_full),
        stratify=df_full[churn_col],
        random_state=args.seed,
    )
    df_sample = df_sample.reset_index(drop=True)
    churn_rate = df_sample[churn_col].mean()
    print(f"      {len(df_sample)} filas · churn rate: {churn_rate:.1%}")

    sample_dir = PROJECT_ROOT / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "telco_sample.csv"
    df_sample.to_csv(sample_path, index=False)
    print(f"      Guardado en: {sample_path}")

    # 3. Entrenar modelo demo --------------------------------------------------
    print("\n[3/4] Entrenando modelo demo sobre la muestra...")
    from churn_agent.data.loader import load_features
    from churn_agent.data.splitter import make_split
    from churn_agent.model.trainer import calibrate, train_lgbm

    X, y = load_features(sample_path)
    print(f"      {len(X)} filas · {X.shape[1]} features · churn rate: {y.mean():.1%}")

    X_train, X_val, y_train, y_val = make_split(X, y, test_size=0.20)
    print(f"      train={len(X_train)} · val={len(X_val)}")

    lgbm_pipeline = train_lgbm(X_train, y_train)
    lgbm_calibrated = calibrate(lgbm_pipeline, X_val, y_val, method="isotonic")

    # 4. Guardar modelo --------------------------------------------------------
    print("\n[4/4] Guardando modelo demo...")
    from churn_agent.model.lgbm_model import LightGBMChurnModel

    demo_dir = PROJECT_ROOT / "models" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    model_path = demo_dir / "lgbm_demo.pkl"

    adapter = LightGBMChurnModel(
        model=lgbm_calibrated,
        feature_names=list(X_train.columns),
    )
    adapter.save(model_path)
    print(f"      Modelo guardado en: {model_path}")

    print("\n=== Artefactos demo generados ===")
    print(f"  CSV:    {sample_path} ({sample_path.stat().st_size // 1024} KB)")
    print(f"  Modelo: {model_path} ({model_path.stat().st_size // 1024} KB)")
    print("\n  Commitea estos archivos (ya incluidos en .gitignore allow-list):")
    print("    git add data/sample/telco_sample.csv models/demo/lgbm_demo.pkl")


if __name__ == "__main__":
    main()

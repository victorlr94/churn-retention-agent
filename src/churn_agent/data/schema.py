"""Esquema y contrato del dataset IBM Telco Customer Churn.

Define:
- Columnas con leakage que NUNCA deben entrar al entrenamiento (ADR-0002).
- Columnas esperadas en el CSV crudo.
- Constantes de nombres de columnas para evitar strings sueltos en el código.

NOTA: este módulo NO usa `from __future__ import annotations` deliberadamente.
Pandera resuelve los tipos de los campos del DataFrameModel con `get_type_hints()`;
las anotaciones como strings (que ese import activa) hacen que la resolución falle.
"""

import pandera.pandas as pa
from pandera.typing import Series

# ---------------------------------------------------------------------------
# Constantes de columnas — evita strings sueltos en el resto del código
# ---------------------------------------------------------------------------

COL_CUSTOMER_ID = "CustomerID"
COL_CHURN_LABEL = "Churn Label"  # target categórico: "Yes" / "No"
COL_CHURN_VALUE = "Churn Value"  # target numérico: 1 / 0
COL_CLTV = "CLTV"  # Customer Lifetime Value (para economics)
COL_SATISFACTION = "Satisfaction Score"

# ---------------------------------------------------------------------------
# Columnas con leakage directo — ADR-0002
# Se excluyen del entrenamiento; algunas sirven solo para evaluar al agente.
# ---------------------------------------------------------------------------
LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "Churn Score",  # propensidad calculada por IBM a partir del resultado
        "Churn Reason",  # solo existe para clientes que ya churnaron
        "Churn Category",  # agregación de Churn Reason
    }
)

# Columnas que se excluyen también del set de features (target + ID)
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        COL_CUSTOMER_ID,
        COL_CHURN_LABEL,
        COL_CHURN_VALUE,
        # Geo de alta cardinalidad sin valor predictivo adicional
        "Count",
        "Country",
        "State",
        "Lat Long",
    }
)

# Todas las columnas que no deben entrar a X
EXCLUDED_FROM_FEATURES: frozenset[str] = LEAKAGE_COLUMNS | NON_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Schema del CSV crudo (validación al cargar)
# Solo define las columnas críticas; las demás se aceptan tal cual.
# ---------------------------------------------------------------------------
class TelcoRawSchema(pa.DataFrameModel):
    """Contrato mínimo del CSV tal como lo devuelve Kaggle."""

    # Identificador
    CustomerID: Series[str] = pa.Field(alias="CustomerID")

    # Target
    ChurnLabel: Series[str] = pa.Field(
        alias="Churn Label",
        isin=["Yes", "No"],
    )
    ChurnValue: Series[int] = pa.Field(
        alias="Churn Value",
        isin=[0, 1],
    )

    # Económico (requerido para las tools de Fase 3)
    CLTV: Series[int] = pa.Field(alias="CLTV", ge=0)

    # Numéricas básicas
    TenureMonths: Series[int] = pa.Field(alias="Tenure Months", ge=0)
    MonthlyCharges: Series[float] = pa.Field(alias="Monthly Charges", ge=0)

    class Config:
        name = "TelcoRawSchema"
        strict = False  # acepta columnas extra (p. ej. geo, otras features)
        coerce = True  # intenta castear tipos antes de fallar

"""Tests de schema, leakage y carga del dataset.

Todos los tests usan DataFrames sintéticos en memoria — no necesitan el CSV
real para correr en CI. El CSV real se valida con scripts/download_data.py.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest

from churn_agent.data.loader import check_no_leakage
from churn_agent.data.schema import (
    EXCLUDED_FROM_FEATURES,
    LEAKAGE_COLUMNS,
    NON_FEATURE_COLUMNS,
    TelcoRawSchema,
)
from churn_agent.exceptions import LeakageError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_raw_df(**overrides: object) -> pd.DataFrame:
    """DataFrame sintético que cumple el esquema mínimo de TelcoRawSchema."""
    base: dict[str, list[object]] = {
        "Customer ID": ["1", "2", "3"],
        "Churn Label": ["Yes", "No", "No"],
        "Churn Value": [1, 0, 0],
        "CLTV": [3000, 5000, 4200],
        "Satisfaction Score": [2, 4, 5],
        "Tenure Months": [1, 24, 36],
        "Monthly Charges": [70.5, 45.0, 89.9],
        # columna extra que no está en el schema mínimo (strict=False debe aceptarla)
        "Contract": ["Month-to-month", "One year", "Two year"],
    }
    base.update(overrides)  # type: ignore[arg-type]
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# Tests de LEAKAGE_COLUMNS
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_leakage_columns_defined() -> None:
    assert "Churn Score" in LEAKAGE_COLUMNS
    assert "Churn Reason" in LEAKAGE_COLUMNS
    assert "Churn Category" in LEAKAGE_COLUMNS


@pytest.mark.unit
def test_leakage_columns_are_excluded_from_features() -> None:
    assert LEAKAGE_COLUMNS.issubset(EXCLUDED_FROM_FEATURES)


@pytest.mark.unit
def test_non_feature_columns_are_excluded_from_features() -> None:
    assert NON_FEATURE_COLUMNS.issubset(EXCLUDED_FROM_FEATURES)


# ---------------------------------------------------------------------------
# Tests de check_no_leakage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_no_leakage_passes_clean_df() -> None:
    df = pd.DataFrame({"Monthly Charges": [70.0], "Contract": ["Month-to-month"]})
    check_no_leakage(df)  # no debe lanzar


@pytest.mark.unit
@pytest.mark.parametrize("leakage_col", sorted(LEAKAGE_COLUMNS))
def test_check_no_leakage_raises_on_each_leakage_column(leakage_col: str) -> None:
    df = pd.DataFrame({"Monthly Charges": [70.0], leakage_col: ["something"]})
    with pytest.raises(LeakageError, match="Leakage detectado"):
        check_no_leakage(df)


# ---------------------------------------------------------------------------
# Tests de TelcoRawSchema
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_telco_raw_schema_validates_correct_df() -> None:
    df = _minimal_raw_df()
    validated = TelcoRawSchema.validate(df, lazy=True)
    assert len(validated) == 3


@pytest.mark.unit
def test_telco_raw_schema_accepts_extra_columns() -> None:
    df = _minimal_raw_df()
    df["Unknown Extra Column"] = "value"
    # strict=False en el schema — columnas extra no fallan
    validated = TelcoRawSchema.validate(df, lazy=True)
    assert "Unknown Extra Column" in validated.columns


@pytest.mark.unit
def test_telco_raw_schema_rejects_invalid_churn_label() -> None:
    df = _minimal_raw_df(**{"Churn Label": ["Yes", "No", "Maybe"]})
    with pytest.raises(pa.errors.SchemaErrors):
        TelcoRawSchema.validate(df, lazy=True)


@pytest.mark.unit
def test_telco_raw_schema_rejects_negative_cltv() -> None:
    df = _minimal_raw_df(**{"CLTV": [3000, -1, 4200]})
    with pytest.raises(pa.errors.SchemaErrors):
        TelcoRawSchema.validate(df, lazy=True)


@pytest.mark.unit
def test_telco_raw_schema_rejects_satisfaction_out_of_range() -> None:
    df = _minimal_raw_df(**{"Satisfaction Score": [1, 6, 3]})
    with pytest.raises(pa.errors.SchemaErrors):
        TelcoRawSchema.validate(df, lazy=True)

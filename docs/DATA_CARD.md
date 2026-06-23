# Data Card — IBM Telco Customer Churn

## Origen

- **Dataset**: IBM Telco Customer Churn (versión "rica" / completa).
- **Naturaleza**: datos **sintéticos** de una telco ficticia en California.
  No contiene datos de personas reales → cero riesgo de exponer PII real.
- **Tamaño**: 7.043 clientes.
- **Distribución**: vía Kaggle / "Accelerator: Telco customer churn" de IBM.
- **Obtención**: ver `scripts/download_data.py` (Fase 1). El CSV **no** se
  versiona; el repo guarda el script y un checksum del archivo.

## Licencia

Redistribuido bajo licencia permisiva (Apache 2.0 en la redistribución de
*Agents for Data*). Al ser sintético y de licencia permisiva, es seguro de usar
en un portafolio público. Verifica los términos de la fuente concreta de
descarga antes de redistribuir el CSV.

## Columnas relevantes para el proyecto

- **Features candidatas**: demográficas, de contrato, servicios contratados,
  cargos mensuales/totales, tenure, etc.
- **Target**: `Churn` (sí/no).
- **Para economía de ofertas**: `CLTV` (Customer Lifetime Value), `Satisfaction
  Score`.
- **Solo para evaluación, NUNCA como feature**: `Churn Reason`, `Churn Category`.

## ⚠️ Leakage conocido (lo más importante de esta tarjeta)

El dataset trae columnas **derivadas del propio churn**. Usarlas como features
es leakage directo y arruina la credibilidad del modelo.

| Columna | Por qué es leakage | Cómo se maneja |
|---------|--------------------|----------------|
| `Churn Score` | Score de propensión calculado por IBM a partir del resultado | **Excluida** del entrenamiento |
| `Churn Reason` | Solo existe para quien ya churnó | Solo para **evaluar** la decisión del agente; nunca input |
| `Churn Category` | Agregación de la razón de churn | **Excluida** del entrenamiento |

La exclusión se hace explícita y testeada en la capa de datos (Fase 1): si una
de estas columnas entra al set de features, la validación lanza `LeakageError`.

## Limitaciones del dataset

| Limitación | Implicación | Manejo |
|------------|-------------|--------|
| Snapshot **point-in-time** (sin cuándo churnó cada cliente) | No permite time-to-event ni ventanas reales | Se enmarca como **scoring de propensión sobre snapshot**, declarado en el README |
| Datos sintéticos y "irrealmente limpios" | El modelo se ve mejor que en producción | Se reporta como limitación; opcional: inyectar ruido/missing para realismo (stretch) |
| Columnas derivadas del target | Leakage si se usan como features | Exclusión explícita y testeada (ver arriba) |

## Splits

Train/test estratificado por `Churn` (Fase 1), con semilla fija
(`CHURN_RANDOM_SEED=42`) para reproducibilidad.

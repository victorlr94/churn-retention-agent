# ADR-0002: Exclusión de columnas con leakage del entrenamiento

## Estado
Aceptado

## Contexto
El dataset IBM Telco Customer Churn incluye columnas **derivadas del propio
resultado de churn**: `Churn Score` (propensión calculada por IBM a partir del
resultado), `Churn Reason` y `Churn Category` (solo existen para clientes que ya
churnaron). Usarlas como features produce leakage: el modelo "aprende" el
resultado y reporta métricas irrealmente altas que colapsan en producción.

## Decisión
**Excluir** `Churn Score`, `Churn Reason` y `Churn Category` del conjunto de
features. `Churn Reason` se conserva únicamente como señal de **evaluación** de la
decisión del agente, nunca como input del modelo. La exclusión se hace explícita
y **testeada** en la capa de datos: si alguna entra al set de features, la
validación lanza `LeakageError`.

## Alternativas consideradas
- **Usarlas como features**: maximiza métricas en papel, pero es leakage directo;
  descalifica el proyecto ante cualquier revisor técnico.
- **Excluir solo en silencio**: funcional, pero pierde la oportunidad de
  demostrar criterio; sin test, el leakage puede reintroducirse por accidente.

## Consecuencias
+ Métricas honestas y defendibles; credibilidad ante un evaluador.
+ Un test impide la reintroducción accidental del leakage.
+ `Churn Reason` queda disponible como ground-truth para evaluar al agente.
- Métricas más modestas que las de notebooks que (erróneamente) usan estas
  columnas; se documenta como decisión deliberada en el README y la Data Card.

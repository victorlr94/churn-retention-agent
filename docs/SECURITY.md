# Security — Threat Model y Mitigaciones

La higiene de seguridad es lo que separa este proyecto de un demo de juguete,
sobre todo para un dominio tipo banca/telco. Este documento es un threat model
mínimo pero explícito. Se actualiza a medida que avanzan las fases.

## Principio rector

El sistema **prefiere rechazar antes que arriesgar** una acción incorrecta o una
fuga. Las tools son read-only y toda acción de cara al cliente queda tras una
compuerta humana.

## Modelo de amenazas y mitigaciones

| Riesgo | Mitigación | Fase |
|--------|-----------|------|
| **Secretos en el código** (API keys de LLM) | `.env` gitignored, `.env.example` solo con nombres, **gitleaks** + `detect-private-key` en pre-commit y CI. Nunca hardcodear | 0 ✅ |
| **Prompt injection** | Las notas/razones del cliente son **input no confiable**: se delimitan, se escapan y el system prompt instruye ignorar instrucciones embebidas | 4 |
| **Alucinación de ofertas** | El agente solo ofrece ítems del catálogo. Validación post-respuesta: oferta fuera de catálogo → `GuardrailError` | 3-4 |
| **Fuga de PII / datos internos** | Aunque el dataset es sintético, se trata como real: solo `customer_id` al exterior; nunca scores internos ni features crudas en el mensaje. Output filtering | 4-6 |
| **Acciones irreversibles** | Tools read-only; la "ejecución" es mock tras la compuerta humana (least privilege) | 3-4 |
| **Abuso de la API** | FastAPI con validación de payload (pydantic) y rate limiting | 6 |
| **Dependencias vulnerables** | `pip-audit`/Dependabot en CI; deps pineadas en `uv.lock` | 0-5 |
| **Falta de auditabilidad** | Cada decisión se loguea (input, tools, EV, oferta, latencia, costo) para auditoría y supervisión humana | 4-7 |

## Estado actual (Fase 0)

- ✅ Secretos fuera del repo: `.gitignore` cubre `.env*`; `.env.example` sin
  valores; gitleaks en pre-commit.
- ✅ Lockfile pineado (`uv.lock`) → superficie de dependencias controlada.
- ✅ `check-added-large-files` evita commitear binarios/datos por error.

## Pendiente por fase

Prompt injection, guardrail de ofertas, output filtering, rate limiting y
trazabilidad se implementan y testean (incl. casos adversariales) en las fases
indicadas arriba.

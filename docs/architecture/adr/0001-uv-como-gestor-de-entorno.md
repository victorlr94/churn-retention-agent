# ADR-0001: uv como gestor de entorno

## Estado
Aceptado

## Contexto
El proyecto es Python puro (ML clásico + agente LLM + API), sin binarios
científicos pesados no empaquetables por pip. Se necesita reproducibilidad desde
el día 1: que un tercero clone el repo y obtenga un entorno idéntico con dos
comandos, y que CI use exactamente las mismas versiones.

## Decisión
Usar **uv** con lockfile (`uv.lock`) versionado y la versión de Python fijada en
`.python-version` (3.12).

## Alternativas consideradas
- **conda/pixi**: justificado solo con binarios pesados (GDAL, CUDA exótica). No
  es el caso; añade peso y lentitud de resolución sin beneficio aquí.
- **pip + requirements.txt + venv**: sin lockfile determinista nativo ni gestión
  de versión de Python; reproducibilidad frágil.
- **Poetry**: válido, pero uv resuelve más rápido, instala Python por sí mismo y
  tiene un flujo lock/sync más simple.

## Consecuencias
+ Lockfile determinista cross-platform; `uv sync --frozen` garantiza el entorno.
+ uv instala la versión de Python correcta sin pasos manuales.
+ Separación nativa runtime / grupo `dev`.
- Herramienta relativamente nueva; se mitiga pineando su versión en el build
  system y documentando el flujo en el README.

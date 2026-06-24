"""Script de evaluación del agente de retención.

Ejecuta la suite completa de EvalCases con componentes fake (sin modelo pkl,
sin CSV, sin ANTHROPIC_API_KEY) y reporta las métricas de calidad.

Uso:
    uv run python scripts/evaluate_agent.py
    uv run python scripts/evaluate_agent.py --exit-code   # falla si gate no pasa

El flag --exit-code se usa en CI para bloquear merges si la calidad cae.
"""

from __future__ import annotations

import argparse
import logging
import sys

from churn_agent.evaluation.cases import EVAL_SUITE
from churn_agent.evaluation.metrics import EvalMetrics, compute_metrics, gate_passes
from churn_agent.evaluation.runner import EvalResult, run_eval_suite


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _print_report(
    results: list[EvalResult], metrics: EvalMetrics, failures: list[str]
) -> None:
    print("\n=== Evaluacion del Agente de Retencion ===\n")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        detail = f" -- {r.failure_reason}" if r.failure_reason else ""
        print(f"  [{status}] {r.case.case_id:30s}  {r.case.label}{detail}")

    print(f"\n--- Metricas ---\n{metrics}\n")

    if failures:
        print("Gate FALLIDO:")
        for f in failures:
            print(f"  x {f}")
    else:
        print("Gate OK: todas las metricas superan los umbrales.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalua el agente de retencion.")
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Salir con codigo 1 si el gate de calidad no pasa (para CI).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    print("Ejecutando suite de evaluacion (sin modelo real ni API key)...")
    results = run_eval_suite(EVAL_SUITE)
    metrics = compute_metrics(results)
    ok, failures = gate_passes(metrics)

    _print_report(results, metrics, failures)

    if args.exit_code and not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

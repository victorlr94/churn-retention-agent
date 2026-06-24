"""CLI para ejecutar el agente de retención en un cliente real.

Uso:
    uv run python scripts/run_agent.py --customer-id 3668-QPYBK
    uv run python scripts/run_agent.py --customer-id 3668-QPYBK --no-hitl

Prerequisitos:
    1. Tener el CSV del dataset en data/raw/telco_customer_churn.csv
    2. Haber entrenado el modelo: uv run python scripts/train_model.py
    3. Haber configurado ANTHROPIC_API_KEY en .env
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Asegura que el paquete sea importable aunque el script se llame directamente.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn_agent.agent.runner import RetentionDecision, run_retention_agent
from churn_agent.config import get_settings

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _print_decision(decision: RetentionDecision) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  AGENTE DE RETENCIÓN — Cliente: {decision.customer_id}")
    print(sep)

    if decision.security_blocked:
        print("  ⚠️  SOLICITUD BLOQUEADA POR SEGURIDAD")
        print(sep)
        return

    if decision.propensity is not None:
        print(f"  Propensidad de churn : {decision.propensity * 100:.1f}%")
    if decision.cltv is not None:
        print(f"  CLTV                 : ${decision.cltv:,} MXN")
    if decision.ev is not None:
        print(f"  EV de la oferta      : ${decision.ev:,.0f} MXN")
    if decision.offer_tier:
        print(f"  Tier recomendado     : {decision.offer_tier}")

    approved_str = (
        "APROBADA (automático)"
        if decision.human_approved
        and decision.ev is not None
        and decision.ev <= get_settings().hitl_ev_threshold
        else "APROBADA (humano)"
        if decision.human_approved
        else "RECHAZADA (humano)"
    )
    print(f"  Estado               : {approved_str}")
    print(sep)
    print("\n  RECOMENDACIÓN DEL AGENTE:\n")
    print(f"  {decision.final_message}")
    print(f"\n{sep}\n")


def _auto_approve(payload: dict[str, Any]) -> bool:
    """Aprueba automáticamente (para demo sin interacción humana)."""
    print(
        f"\n  [--no-hitl] Auto-aprobando oferta: {payload.get('offer_tier')} "
        f"(EV={payload.get('ev', 0):.0f})"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agente de retención anti-churn con human-in-the-loop"
    )
    parser.add_argument(
        "--customer-id",
        required=True,
        help="ID del cliente a analizar (p. ej. 3668-QPYBK)",
    )
    parser.add_argument(
        "--no-hitl",
        action="store_true",
        help="Omitir aprobación humana (modo demo automático)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Habilitar logging DEBUG",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("churn_agent").setLevel(logging.DEBUG)

    settings = get_settings()
    if not settings.anthropic_api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY no configurada.\n"
            "  Agrega la clave a .env:\n"
            "  ANTHROPIC_API_KEY=sk-ant-...\n"
        )
        sys.exit(1)

    approve_callback = _auto_approve if args.no_hitl else None

    try:
        decision = run_retention_agent(
            args.customer_id,
            approve_callback=approve_callback,
        )
        _print_decision(decision)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR inesperado: {exc}\n")
        raise


if __name__ == "__main__":
    main()

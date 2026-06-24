"""Visor de sesiones del agente de retencion.

Lee el log JSONL de sesiones y muestra un resumen de actividad y coste.

Uso:
    uv run python scripts/show_sessions.py
    uv run python scripts/show_sessions.py --last 20
    uv run python scripts/show_sessions.py --summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from churn_agent.config import get_settings
from churn_agent.observability.session_log import ObservabilityStore, SessionRecord


def _fmt_tier(tier: str | None) -> str:
    return (tier or "---").ljust(8)


def _fmt_ev(ev: float | None) -> str:
    return f"{ev:>7.1f}" if ev is not None else "    ---"


def _fmt_approved(approved: bool | None, hitl: bool) -> str:
    if not hitl:
        return "auto"
    if approved is None:
        return "pending"
    return "YES" if approved else "NO"


def _print_records(records: list[SessionRecord], n: int) -> None:
    shown = records[-n:] if n > 0 else records
    print(
        f"\n{'Timestamp':20s}  {'Customer':12s}  {'Phase':7s}  "
        f"{'Tier':8s}  {'EV':>7s}  {'Latency':>8s}  {'HITL':7s}  {'Cost':>10s}"
    )
    print("-" * 92)
    for r in shown:
        ts = r.timestamp_utc[:19].replace("T", " ")
        latency = f"{r.latency_ms:>6.0f}ms"
        hitl_col = _fmt_approved(r.human_approved, r.hitl_triggered)
        cost_col = f"${r.cost_usd_est:.6f}"
        blocked = " [BLOCKED]" if r.security_blocked else ""
        print(
            f"{ts}  {r.customer_id[:12]:12s}  {r.phase[:7]:7s}  "
            f"{_fmt_tier(r.offer_tier)}  {_fmt_ev(r.ev)}  "
            f"{latency}  {hitl_col:7s}  {cost_col}{blocked}"
        )


def _print_summary(store: ObservabilityStore) -> None:
    m = store.compute_metrics()
    if m.total == 0:
        print("Sin sesiones registradas todavia.")
        return
    tiers_str = "  ".join(f"{k}={v}" for k, v in m.tier_distribution.items())
    print(f"\n{'Resumen':}")
    print(f"  Total sesiones:     {m.total}")
    print(f"  Latencia media:     {m.avg_latency_ms:.0f} ms")
    print(f"  Tasa HITL:          {m.hitl_rate:.0%}")
    print(f"  Tasa bloqueados:    {m.block_rate:.0%}")
    print(f"  Tiers:              {tiers_str or 'ninguno'}")
    print(f"  Coste total est.:   ${m.total_cost_usd_est:.6f} USD")
    print(f"  Coste medio est.:   ${m.avg_cost_usd_est:.6f} USD/sesion")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visor de sesiones del agente.")
    parser.add_argument(
        "--last", type=int, default=20, help="Mostrar las N ultimas sesiones (0=todas)"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Mostrar solo el resumen"
    )
    parser.add_argument(
        "--log", type=Path, help="Ruta al log JSONL (default: settings)"
    )
    args = parser.parse_args()

    settings = get_settings()
    log_path: Path = args.log or settings.session_log_path
    store = ObservabilityStore(log_path=log_path)

    print(f"Sessions log: {log_path}")
    records = store.load_all()

    if not records:
        print(
            "Sin sesiones registradas todavia. Ejecuta la API y analiza algunos clientes."
        )
        return 0

    print(f"  {len(records)} sesiones registradas\n")
    _print_summary(store)

    if not args.summary:
        n = args.last
        label = f"ultimas {n}" if n > 0 else "todas"
        print(f"\n--- Sesiones ({label}) ---")
        _print_records(records, n)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Registro JSONL de sesiones del agente con metricas agregadas.

Cada sesion se escribe como una linea JSON en logs/sessions.jsonl (gitignoreado).
El store opera en modo no-op si log_path es None (para tests o modo degradado).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SessionRecord:
    """Registro completo de una llamada al agente."""

    session_id: str
    customer_id: str
    phase: str  # "analyze" | "approve"
    timestamp_utc: str  # ISO 8601
    latency_ms: float
    propensity: float | None
    cltv: int | None
    ev: float | None
    offer_tier: str | None
    hitl_triggered: bool
    human_approved: bool | None
    security_blocked: bool
    n_messages: int
    input_tokens_est: int
    output_tokens_est: int
    cost_usd_est: float


@dataclass
class SessionMetrics:
    """Metricas agregadas de todas las sesiones registradas."""

    total: int
    avg_latency_ms: float
    hitl_rate: float
    block_rate: float
    tier_distribution: dict[str, int]
    total_cost_usd_est: float
    avg_cost_usd_est: float


class ObservabilityStore:
    """Almacen JSONL append-only de sesiones del agente.

    Si log_path es None opera en modo no-op (tests y modo degradado).
    """

    def __init__(self, log_path: Path | None = None) -> None:
        self._path = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path | None:
        return self._path

    def record(self, session: SessionRecord) -> None:
        """Escribe una sesion al log JSONL (no-op si log_path es None)."""
        if self._path is None:
            return
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(session)) + "\n")
        except OSError as exc:
            logger.warning(
                "No se pudo registrar la sesion %s: %s", session.session_id, exc
            )

    def load_all(self) -> list[SessionRecord]:
        """Carga todas las sesiones del log JSONL."""
        if self._path is None or not self._path.exists():
            return []
        records: list[SessionRecord] = []
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(SessionRecord(**json.loads(line)))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Error leyendo log de sesiones: %s", exc)
        return records

    def compute_metrics(self) -> SessionMetrics:
        """Calcula metricas agregadas de todas las sesiones."""
        records = self.load_all()
        if not records:
            return SessionMetrics(
                total=0,
                avg_latency_ms=0.0,
                hitl_rate=0.0,
                block_rate=0.0,
                tier_distribution={},
                total_cost_usd_est=0.0,
                avg_cost_usd_est=0.0,
            )

        n = len(records)
        avg_latency = sum(r.latency_ms for r in records) / n
        hitl_rate = sum(1 for r in records if r.hitl_triggered) / n
        block_rate = sum(1 for r in records if r.security_blocked) / n

        tiers: dict[str, int] = {}
        for r in records:
            if r.offer_tier:
                tiers[r.offer_tier] = tiers.get(r.offer_tier, 0) + 1

        total_cost = sum(r.cost_usd_est for r in records)

        return SessionMetrics(
            total=n,
            avg_latency_ms=round(avg_latency, 1),
            hitl_rate=round(hitl_rate, 4),
            block_rate=round(block_rate, 4),
            tier_distribution=dict(sorted(tiers.items())),
            total_cost_usd_est=round(total_cost, 8),
            avg_cost_usd_est=round(total_cost / n, 8),
        )


def build_session_record(
    *,
    session_id: str,
    customer_id: str,
    phase: str,
    latency_ms: float,
    result: dict[str, Any],
    hitl_triggered: bool,
    messages: list[Any],
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> SessionRecord:
    """Construye un SessionRecord a partir del estado final del grafo."""
    raw_propensity = result.get("propensity")
    raw_cltv = result.get("cltv")
    raw_ev = result.get("ev")
    raw_tier = result.get("offer_tier")
    raw_approved = result.get("human_approved")

    return SessionRecord(
        session_id=session_id,
        customer_id=customer_id,
        phase=phase,
        timestamp_utc=datetime.now(UTC).isoformat(),
        latency_ms=round(latency_ms, 1),
        propensity=float(raw_propensity)
        if isinstance(raw_propensity, int | float)
        else None,
        cltv=int(raw_cltv) if isinstance(raw_cltv, int | float) else None,
        ev=float(raw_ev) if isinstance(raw_ev, int | float) else None,
        offer_tier=str(raw_tier) if isinstance(raw_tier, str) else None,
        hitl_triggered=hitl_triggered,
        human_approved=bool(raw_approved) if raw_approved is not None else None,
        security_blocked=bool(result.get("security_blocked", False)),
        n_messages=len(messages),
        input_tokens_est=input_tokens,
        output_tokens_est=output_tokens,
        cost_usd_est=cost_usd,
    )

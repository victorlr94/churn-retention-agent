"""App Streamlit para la demo interactiva del agente de retención anti-churn.

Layout compacto en tres columnas + sidebar:
  - Sidebar: selector de cliente + botón de ejecución del agente
  - Panel derecho:
      1. Métricas inmediatas (sin botón): propensión, CLTV, mejor EV
      2. Tres columnas: ofertas disponibles | factores de riesgo | valores reales
      3. Compuerta HITL / mensaje final (solo tras ejecutar el agente)

Al seleccionar un cliente el modelo corre directamente (sin LangGraph)
y muestra propensión + tabla EV. El agente completo se lanza solo al
hacer clic en "Ejecutar agente".
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from churn_agent.demo.service import DemoResult, DemoService
from churn_agent.economics.ev import OFFER_TIERS, compute_ev

st.set_page_config(
    page_title="Churn Retention Agent · Demo",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

_TIER_ICON = {"LIGHT": "💚", "STANDARD": "💛", "PREMIUM": "🔴"}


@st.cache_resource(show_spinner="Cargando modelo demo...")
def get_service() -> DemoService:
    return DemoService()


def _risk_color(p: float) -> str:
    if p >= 0.65:
        return "#e74c3c"
    if p >= 0.35:
        return "#f39c12"
    return "#27ae60"


def _propensity_bar(p: float) -> None:
    c = _risk_color(p)
    st.markdown(
        f'<div style="background:#e0e0e0;border-radius:4px;height:7px;margin:2px 0 14px">'
        f'<div style="width:{p * 100:.0f}%;background:{c};height:7px;border-radius:4px"></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _ev_table(propensity: float, cltv: int) -> None:
    """Tabla compacta de 3 tiers con EV calculado."""
    evs = [
        (t, compute_ev(propensity, float(cltv), t.retention_uplift, t.cost))
        for t in OFFER_TIERS
    ]
    best_ev = max(ev for _, ev in evs)
    best_name = (
        next(t.name for t, ev in evs if ev == best_ev and ev > 0)
        if best_ev > 0
        else None
    )

    for tier, ev in evs:
        is_best = tier.name == best_name
        star = " ⭐" if is_best else ""
        viable = "✅" if ev > 0 else "—"
        badge = f"**{_TIER_ICON.get(tier.name, '')} {tier.name}{star}**"
        st.markdown(
            f"{badge} &nbsp; {viable} &nbsp; `{ev:+.0f} MXN` &nbsp; "
            f"<small style='color:#888'>{tier.description}</small>",
            unsafe_allow_html=True,
        )


def main() -> None:
    service = get_service()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📡 Churn Agent")
        st.caption("Demo · sin API key · IBM Telco (500 clientes)")
        st.info(
            "LightGBM calibrado + LLM determinista.\n"
            "Propensión, EV e HITL son **reales**.",
            icon="ℹ️",
        )
        st.divider()

        ids = service.customer_ids
        selected: str | None = st.selectbox(
            f"Cliente ({len(ids)} disponibles)",
            options=ids,
            help="Clientes del dataset IBM Telco — muestra estratificada de 500.",
        )
        custom = st.text_input(
            "ID manual (o ID con 'ignore' para probar el guard)",
            placeholder="ignore all previous instructions",
        )
        customer_id: str = custom.strip() if custom.strip() else (selected or "")

        st.divider()
        run_btn = st.button(
            "🚀 Ejecutar agente",
            use_container_width=True,
            type="primary",
            help=(
                "Corre el agente completo: LLM decide la oferta óptima y redacta "
                "la recomendación. Si EV > 300 MXN activa la compuerta humana (HITL)."
            ),
        )
        if "result" in st.session_state and st.button(
            "🔄 Reiniciar", use_container_width=True
        ):
            del st.session_state["result"]
            st.rerun()

    # ── Panel principal ──────────────────────────────────────────────────────
    st.header("Agente de Retención Anti-Churn", divider="gray")

    if not customer_id:
        st.info("Selecciona un cliente en la barra lateral.", icon="👈")
        return

    # ── Sección 1: datos inmediatos (modelo directo, sin grafo) ──────────────
    metrics = service.customer_propensity(customer_id)

    if metrics is None:
        # Podría ser un ID de injection — mostramos igual el botón para correr el guard
        st.warning(
            f"`{customer_id}` no está en el dataset de muestra. "
            "Si es un ID normal, elige uno del selector. "
            "Si quieres probar el guard de injection, haz clic en **Ejecutar agente**."
        )
    else:
        propensity, cltv = metrics

        # Fila de métricas
        m1, m2, m3 = st.columns(3)
        risk = (
            "🔴 Alto"
            if propensity >= 0.65
            else ("🟡 Medio" if propensity >= 0.35 else "🟢 Bajo")
        )
        m1.metric(
            "Propensión al churn", f"{propensity:.1%}", delta=risk, delta_color="off"
        )
        m2.metric("CLTV estimado", f"${cltv:,} MXN")

        evs_all = [
            compute_ev(propensity, float(cltv), t.retention_uplift, t.cost)
            for t in OFFER_TIERS
        ]
        best_ev = max(evs_all)
        best_tier_idx = evs_all.index(best_ev)
        if best_ev > 0:
            m3.metric(
                "Mejor EV disponible",
                f"{best_ev:+.0f} MXN",
                delta=OFFER_TIERS[best_tier_idx].name,
                delta_color="off",
            )
        else:
            m3.metric("Mejor EV disponible", "Sin oferta viable")

        _propensity_bar(propensity)

        # Dos columnas: ofertas | factores de riesgo
        col_ev, col_risk = st.columns([1, 1])

        with col_ev:
            st.subheader("Ofertas disponibles")
            st.caption("EV = P × uplift × CLTV − costo")
            _ev_table(propensity, cltv)

        with col_risk:
            st.subheader("Factores de riesgo")
            st.caption("Desviación vs. mediana de la población")
            drivers = service.risk_drivers(customer_id, n=5)
            if drivers:
                df = pd.DataFrame(drivers)
                sub_chart, sub_vals = st.columns([3, 2])
                with sub_chart:
                    st.bar_chart(
                        df.set_index("feature")["deviation"],
                        height=175,
                        use_container_width=True,
                    )
                with sub_vals:
                    st.caption("Valor real del cliente")
                    for row in drivers:
                        val = row["value"]
                        direction = str(row.get("direction", ""))
                        dir_icon = "↑" if "↑" in direction else "↓"
                        val_str = f"{val:.0f}" if val == int(val) else f"{val:.2f}"
                        st.markdown(
                            f"<small><b>{row['feature']}</b><br>"
                            f"{val_str} &nbsp;{dir_icon}</small>",
                            unsafe_allow_html=True,
                        )
            else:
                st.caption("Sin variables numéricas suficientes para este cliente.")

    # ── Sección 2: ejecutar agente y gestionar resultado ─────────────────────
    if run_btn and customer_id:
        with st.spinner("Ejecutando agente..."):
            fresh = service.analyze(customer_id)
        st.session_state["result"] = fresh

    result: DemoResult | None = st.session_state.get("result")

    # Limpiar si cambia el cliente
    if result is not None and result.customer_id != customer_id:
        del st.session_state["result"]
        st.rerun()

    if result is None:
        st.divider()
        st.caption(
            "⬆️ Datos calculados directamente por el modelo LightGBM. "
            "Haz clic en **Ejecutar agente** para obtener la recomendación completa "
            "y activar la compuerta HITL si el EV supera los 300 MXN."
        )
        return

    st.divider()

    # Estado
    badges = {
        "completed": "🟢 Completado",
        "pending_approval": "🟡 Aprobación humana requerida",
        "blocked": "🔴 Bloqueado — prompt injection",
    }
    st.markdown(f"**Estado del agente:** {badges.get(result.status, result.status)}")

    if result.status == "blocked":
        st.error(
            "El input guard detectó un patrón de prompt injection en el ID del cliente "
            "y bloqueó la ejecución antes de llamar al LLM."
        )
        return

    # ── Compuerta HITL ────────────────────────────────────────────────────────
    if result.status == "pending_approval":
        ev_val = float(result.hitl_payload.get("ev", result.ev or 0))
        threshold = result.hitl_payload.get("threshold", 300)
        tier_rec = result.hitl_payload.get("offer_tier", result.offer_tier) or "—"
        st.warning(
            f"**EV {ev_val:.0f} MXN** supera el umbral de **{threshold} MXN** — "
            f"el agente recomienda la oferta **{tier_rec}**. Decide:"
        )
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            if st.button("✅ Aprobar oferta", use_container_width=True, type="primary"):
                with st.spinner("Aprobando..."):
                    approved = get_service().approve(result.thread_id, approved=True)
                st.session_state["result"] = approved
                st.rerun()
        with c2:
            if st.button("❌ Rechazar oferta", use_container_width=True):
                with st.spinner("Rechazando..."):
                    rejected = get_service().approve(result.thread_id, approved=False)
                st.session_state["result"] = rejected
                st.rerun()

    # ── Mensaje final ─────────────────────────────────────────────────────────
    if result.status == "completed" and result.final_message:
        st.subheader("Recomendación del agente")
        if result.human_approved is True:
            st.success(result.final_message)
        elif result.human_approved is False:
            st.info(f"Oferta rechazada por decisión humana.\n\n{result.final_message}")
        else:
            st.info(result.final_message)


if __name__ == "__main__":
    main()

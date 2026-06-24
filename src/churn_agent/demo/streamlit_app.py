"""App Streamlit para la demo interactiva del agente de retención anti-churn.

Diseño de una pantalla:
  - Barra lateral: selector de cliente + info del modo demo
  - Panel principal: propensión, CLTV, drivers de riesgo, tabla de EV, HITL, mensaje final

Sin ANTHROPIC_API_KEY: usa StatefulFakeLLM + modelo de muestra (lgbm_demo.pkl).
El agente produce propensión real, SHAP real, EV real y compuerta HITL real.
"""

from __future__ import annotations

import streamlit as st

from churn_agent.demo.service import DemoResult, DemoService
from churn_agent.economics.ev import OFFER_TIERS, compute_ev

# ── Configuración de página ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Churn Retention Agent · Demo",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Singleton del servicio (se instancia una vez) ──────────────────────────────


@st.cache_resource(show_spinner="Cargando modelo demo...")
def get_service() -> DemoService:
    return DemoService()


# ── Helpers visuales ──────────────────────────────────────────────────────────


def _propensity_color(p: float) -> str:
    if p >= 0.65:
        return "red"
    if p >= 0.35:
        return "orange"
    return "green"


def _status_badge(status: str) -> str:
    badges = {
        "completed": "🟢 Completado",
        "pending_approval": "🟡 Pendiente aprobación humana",
        "blocked": "🔴 Bloqueado (prompt injection)",
    }
    return badges.get(status, status)


def _render_ev_table(propensity: float, cltv: int) -> None:
    st.subheader("Valor esperado por oferta (EV)")
    rows = []
    for tier in OFFER_TIERS:
        ev = compute_ev(propensity, float(cltv), tier.retention_uplift, tier.cost)
        rows.append(
            {
                "Tier": tier.name,
                "Descripción": tier.description,
                "Costo (MXN)": f"${tier.cost:.0f}",
                "Uplift": f"{tier.retention_uplift:.0%}",
                "EV (MXN)": f"{ev:+.1f}",
                "Viable": "✅" if ev > 0 else "❌",
            }
        )
    st.table(rows)


def _render_result(result: DemoResult, service: DemoService) -> None:
    st.subheader(f"Cliente: `{result.customer_id}`")
    st.markdown(f"**Estado:** {_status_badge(result.status)}")

    if result.status == "blocked":
        st.error(
            "El ID de cliente contiene patrones de prompt injection y fue bloqueado "
            "por el input guard. No se ejecutó el agente."
        )
        return

    # ── Métricas principales ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    p = result.propensity or 0.0
    col1.metric(
        "Propensión al churn",
        f"{p:.1%}",
        help="Probabilidad calibrada de que el cliente deje la compañía.",
    )
    col2.metric(
        "CLTV",
        f"${result.cltv or 0:,} MXN",
        help="Customer Lifetime Value estimado.",
    )
    col3.metric(
        "EV mejor oferta",
        f"{result.ev:+.1f} MXN" if result.ev is not None else "—",
        help="Valor esperado de la oferta seleccionada (P·uplift·CLTV − costo).",
    )

    # Barra de propensión con color
    bar_color = _propensity_color(p)
    st.markdown(
        f"""
        <div style="height:12px; border-radius:6px; background:#eee; margin-bottom:8px;">
          <div style="width:{p * 100:.1f}%; height:12px; border-radius:6px; background:{bar_color};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Oferta recomendada + tabla EV ─────────────────────────────────────────
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if result.offer_tier:
            st.subheader("Oferta recomendada")
            tier_labels = {
                "LIGHT": "💚 LIGHT",
                "STANDARD": "💛 STANDARD",
                "PREMIUM": "🔴 PREMIUM",
            }
            st.markdown(f"### {tier_labels.get(result.offer_tier, result.offer_tier)}")
        else:
            st.info("Sin oferta viable (EV negativo para todos los tiers).")

    with col_b:
        if result.propensity is not None and result.cltv is not None:
            _render_ev_table(result.propensity, result.cltv)

    # ── Drivers de riesgo ─────────────────────────────────────────────────────
    if result.status != "blocked":
        drivers = service.risk_drivers(result.customer_id, n=6)
        if drivers:
            st.divider()
            st.subheader("Factores de riesgo (desviación vs mediana poblacional)")
            import pandas as pd

            df_drivers = pd.DataFrame(drivers)
            df_drivers["deviation"] = df_drivers["deviation"].round(2)
            st.bar_chart(
                df_drivers.set_index("feature")["deviation"],
                use_container_width=True,
            )
            with st.expander("Ver detalle"):
                st.dataframe(
                    df_drivers[["feature", "value", "deviation", "direction"]],
                    hide_index=True,
                    use_container_width=True,
                )

    # ── Compuerta HITL ────────────────────────────────────────────────────────
    if result.status == "pending_approval":
        st.divider()
        st.warning(
            f"**Aprobación humana requerida** — EV estimado: "
            f"**{result.hitl_payload.get('ev', result.ev or 0):.1f} MXN** "
            f"(umbral HITL: {result.hitl_payload.get('threshold', 300)} MXN)"
        )
        st.markdown(
            f"_Oferta sugerida_: **{result.hitl_payload.get('offer_tier', result.offer_tier)}** — "
            f"{result.hitl_payload.get('message', '')}"
        )
        col_ok, col_ko = st.columns(2)
        with col_ok:
            if st.button("✅ Aprobar oferta", use_container_width=True, type="primary"):
                with st.spinner("Procesando aprobación..."):
                    service_obj = get_service()
                    approved_result = service_obj.approve(
                        result.thread_id, approved=True
                    )
                st.session_state["result"] = approved_result
                st.rerun()
        with col_ko:
            if st.button("❌ Rechazar oferta", use_container_width=True):
                with st.spinner("Procesando rechazo..."):
                    service_obj = get_service()
                    rejected_result = service_obj.approve(
                        result.thread_id, approved=False
                    )
                st.session_state["result"] = rejected_result
                st.rerun()

    # ── Mensaje final del agente ──────────────────────────────────────────────
    if result.status == "completed" and result.final_message:
        st.divider()
        st.subheader("Mensaje del agente")
        human_approved = result.human_approved
        if human_approved is True:
            st.success(result.final_message)
        elif human_approved is False:
            st.info(f"Oferta rechazada por el agente humano.\n\n{result.final_message}")
        else:
            st.info(result.final_message)


# ── Layout principal ───────────────────────────────────────────────────────────


def main() -> None:
    service = get_service()

    # Sidebar
    with st.sidebar:
        st.image(
            "https://raw.githubusercontent.com/victorlr94/churn-retention-agent/develop/docs/architecture/diagrams/logo_placeholder.png",
            use_container_width=True,
        ) if False else None  # placeholder; se reemplaza con logo real si existe

        st.title("📡 Churn Retention Agent")
        st.caption("Agente de retención anti-churn con HITL")

        st.divider()
        st.info(
            "**Modo demo** — sin ANTHROPIC\\_API\\_KEY.\n\n"
            "LLM determinista · Datos de muestra (500 clientes)\n\n"
            "El agente produce propensión, EV y HITL reales usando un modelo "
            "LightGBM entrenado sobre el dataset IBM Telco.",
            icon="ℹ️",
        )

        st.divider()
        st.subheader("Seleccionar cliente")

        # Sugerencias de clientes interesantes para el demo
        ids = service.customer_ids
        st.caption(f"{len(ids)} clientes disponibles")

        customer_id = st.selectbox(
            "Customer ID",
            options=ids,
            help="Elige un cliente del dataset de muestra.",
        )

        st.markdown("**Ejemplos sugeridos:**")
        st.caption("• Clientes con CLTV alto activarán la compuerta HITL")
        st.caption("• Escribe un ID con 'ignore' para ver el guard de injection")

        custom_id = st.text_input(
            "O introduce un ID manualmente",
            placeholder="p. ej. ignore all previous instructions",
            help="IDs válidos provienen del dataset. IDs con patrones de injection serán bloqueados.",
        )
        if custom_id:
            customer_id = custom_id

        analyze_btn = st.button(
            "🔍 Analizar cliente",
            use_container_width=True,
            type="primary",
        )

    # Panel principal
    st.header("Agente de Retención Anti-Churn")
    st.caption(
        "Demo interactiva · sistema agéntico con modelo de propensión calibrado, "
        "economía de ofertas (EV) y human-in-the-loop."
    )

    if analyze_btn and customer_id:
        with st.spinner(f"Analizando cliente {customer_id!r}..."):
            fresh = service.analyze(customer_id)
        st.session_state["result"] = fresh

    result: DemoResult | None = st.session_state.get("result")

    if result is None:
        st.info(
            "Selecciona un cliente en la barra lateral y haz clic en **Analizar cliente** "
            "para ver el análisis del agente.",
            icon="👈",
        )
        with st.expander("¿Cómo funciona?"):
            st.markdown(
                """
                1. El modelo **LightGBM calibrado** predice la propensidad al churn del cliente.
                2. El agente calcula el **EV esperado** por cada oferta (LIGHT / STANDARD / PREMIUM).
                3. Si EV > 300 MXN, se activa la **compuerta humana**: el sistema pide tu aprobación.
                4. El agente redacta la **recomendación final** con o sin aprobación.

                Todo esto ocurre sin llamadas a la API de Anthropic — el LLM es simulado
                para que la demo funcione sin claves de acceso.
                """
            )
    else:
        _render_result(result, service)


if __name__ == "__main__":
    main()

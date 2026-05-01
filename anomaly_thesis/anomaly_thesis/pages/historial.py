"""
pages/historial.py — Gestión del historial
"""
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.almacenamiento import cargar_historico, cargar_log, borrar_historico


def render():
    st.markdown('<div class="section-title">📋 Historial de Análisis</div>', unsafe_allow_html=True)

    df_log = cargar_log()
    df_hist = cargar_historico()

    if df_log is None:
        st.info("No hay historial todavía.")
        return

    # Resumen por carga
    st.subheader("📑 Resumen por carga")
    st.dataframe(df_log, use_container_width=True)

    # Filtrar por carga específica
    st.subheader("🔎 Explorar una carga")
    id_sel = st.selectbox("Selecciona ID de carga", df_log["id_carga"].tolist())

    if df_hist is not None and id_sel:
        df_filtrado = df_hist[df_hist["id_carga"] == id_sel]
        col_anomalia = "anomaly_ensemble" if "anomaly_ensemble" in df_filtrado.columns else "anomaly"

        st.markdown(f"**Total registros:** {len(df_filtrado):,}   |   **Anomalías:** {df_filtrado[col_anomalia].sum():,}")

        vista = st.radio("Ver", ["Todos", "Solo anomalías", "Solo normales"], horizontal=True)
        if vista == "Solo anomalías":
            df_filtrado = df_filtrado[df_filtrado[col_anomalia] == True]
        elif vista == "Solo normales":
            df_filtrado = df_filtrado[df_filtrado[col_anomalia] == False]

        st.dataframe(df_filtrado, use_container_width=True)

        csv = df_filtrado.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar esta vista", csv, f"historial_{id_sel}.csv", "text/csv")

    st.divider()
    if st.button("🗑️ Borrar TODO el historial", type="secondary"):
        borrar_historico()
        st.success("Historial eliminado correctamente.")
        st.rerun()

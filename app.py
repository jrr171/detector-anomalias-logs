"""
pages/comparacion.py — Comparación entre cargas históricas
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.almacenamiento import cargar_log

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,32,53,0.8)",
    font=dict(color="#e2e8f0", family="Space Grotesk"),
    margin=dict(l=20, r=20, t=40, b=20),
)

def render():
    st.markdown('<div class="section-title">📈 Comparación entre Análisis</div>', unsafe_allow_html=True)

    df_log = cargar_log()

    if df_log is None or df_log.empty:
        st.info("Aún no hay análisis registrados. Ve a **Análisis** y procesa un archivo primero.")
        return

    st.subheader("📋 Log de todos los análisis")
    st.dataframe(df_log, use_container_width=True)

    # Gráfica evolución anomalías
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_log["fecha"], y=df_log["pct_anomalias"],
        mode="lines+markers+text",
        text=df_log["archivo"],
        textposition="top center",
        line=dict(color="#60a5fa", width=2),
        marker=dict(size=8, color="#a78bfa")
    ))
    fig1.update_layout(
        title="Evolución del % de anomalías por análisis",
        xaxis_title="Fecha", yaxis_title="% Anomalías",
        **PLOT_LAYOUT
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Gráfica registros vs anomalías
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df_log["id_carga"], y=df_log["registros"],
        name="Total", marker_color="#60a5fa"
    ))
    fig2.add_trace(go.Bar(
        x=df_log["id_carga"], y=df_log["anomalias"],
        name="Anomalías", marker_color="#f87171"
    ))
    fig2.update_layout(
        barmode="overlay",
        title="Registros totales vs anomalías por carga",
        xaxis_title="ID Carga", yaxis_title="Cantidad",
        **PLOT_LAYOUT
    )
    st.plotly_chart(fig2, use_container_width=True)

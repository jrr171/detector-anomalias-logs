"""
pages/analisis.py — Página principal de análisis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import ejecutar_deteccion
from core.metricas import resumen_estadistico, comparar_resultados, perfil_anomalia
from core.almacenamiento import guardar_resultado, registrar_log, nuevo_id, exportar_anomalias


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

COLORES = {
    "fondo": "#0a0e1a",
    "panel": "#1a2035",
    "azul": "#60a5fa",
    "violeta": "#a78bfa",
    "rojo": "#f87171",
    "verde": "#4ade80",
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,32,53,0.8)",
    font=dict(color="#e2e8f0", family="Space Grotesk"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def leer_csv(archivo):
    for sep, enc in [(None, "utf-8"), (";", "latin1"), (",", "latin1")]:
        try:
            kwargs = dict(engine="python", encoding=enc)
            if sep:
                kwargs["sep"] = sep
            else:
                kwargs["sep"] = None
            return pd.read_csv(archivo, **kwargs, on_bad_lines="skip")
        except Exception:
            continue
    raise ValueError("No se pudo leer el CSV con ningún formato conocido.")


def detectar_fechas(df):
    meses = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
             7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
    for col in df.columns:
        if "fecha" in col.lower() or "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df["anio"] = df[col].dt.year
                df["mes"] = df[col].dt.month
                df["dia"] = df[col].dt.day
                df["mes_nombre"] = df["mes"].map(meses)
                return col
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────
# GRÁFICAS
# ─────────────────────────────────────────────

def grafica_distribucion_scores(scores, umbral, nombre_algo):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores, nbinsx=50,
        marker_color=COLORES["azul"], opacity=0.7,
        name="Score de anomalía"
    ))
    fig.add_vline(x=umbral, line_dash="dash", line_color=COLORES["rojo"],
                  annotation_text=f"Umbral ({umbral:.3f})",
                  annotation_font_color=COLORES["rojo"])
    fig.update_layout(
        title=f"Distribución de scores — {nombre_algo}",
        xaxis_title="Score de anomalía",
        yaxis_title="Frecuencia",
        **PLOT_LAYOUT
    )
    return fig


def grafica_scatter_anomalias(df, col_x, col_y, mascara):
    colores = np.where(mascara, COLORES["rojo"], COLORES["azul"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[col_x][~mascara], y=df[col_y][~mascara],
        mode="markers", name="Normal",
        marker=dict(color=COLORES["azul"], size=5, opacity=0.6)
    ))
    fig.add_trace(go.Scatter(
        x=df[col_x][mascara], y=df[col_y][mascara],
        mode="markers", name="Anomalía",
        marker=dict(color=COLORES["rojo"], size=8, opacity=0.9,
                    symbol="x", line=dict(width=1))
    ))
    fig.update_layout(
        title=f"Scatter: {col_x} vs {col_y}",
        xaxis_title=col_x, yaxis_title=col_y,
        **PLOT_LAYOUT
    )
    return fig


def grafica_votos_ensemble(votos):
    conteo = pd.Series(votos).value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=[f"{int(v)} votos" for v in conteo.index],
        y=conteo.values,
        marker_color=[COLORES["rojo"] if v > 1 else COLORES["azul"] for v in conteo.index],
        text=conteo.values, textposition="outside"
    ))
    fig.update_layout(
        title="Distribución de votos del Ensemble",
        xaxis_title="Número de algoritmos que marcaron anomalía",
        yaxis_title="Cantidad de registros",
        **PLOT_LAYOUT
    )
    return fig


def grafica_comparacion_barras(tabla_comp):
    fig = go.Figure(go.Bar(
        x=tabla_comp["Algoritmo"],
        y=tabla_comp["Anomalías detectadas"],
        marker=dict(
            color=tabla_comp["Anomalías detectadas"],
            colorscale=[[0, COLORES["azul"]], [1, COLORES["rojo"]]],
        ),
        text=tabla_comp["% del total"],
        textposition="outside"
    ))
    fig.update_layout(
        title="Anomalías detectadas por algoritmo",
        xaxis_title="Algoritmo",
        yaxis_title="N° anomalías",
        **PLOT_LAYOUT
    )
    return fig


def grafica_heatmap_correlacion(corr_matrix):
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu_r",
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont_size=10
    ))
    fig.update_layout(title="Mapa de correlación", **PLOT_LAYOUT)
    return fig


def grafica_boxplot_anomalias(df, columnas, mascara):
    fig = make_subplots(rows=1, cols=len(columnas[:4]),
                        subplot_titles=columnas[:4])
    for i, col in enumerate(columnas[:4], 1):
        fig.add_trace(go.Box(
            y=df[col][~mascara], name="Normal",
            marker_color=COLORES["azul"], showlegend=(i == 1)
        ), row=1, col=i)
        fig.add_trace(go.Box(
            y=df[col][mascara], name="Anomalía",
            marker_color=COLORES["rojo"], showlegend=(i == 1)
        ), row=1, col=i)
    fig.update_layout(title="Distribución por columna: Normal vs Anomalía", **PLOT_LAYOUT)
    return fig


# ─────────────────────────────────────────────
# RENDER PRINCIPAL
# ─────────────────────────────────────────────

def render(umbral_percentil, max_iter, algoritmos):
    st.markdown('<div class="section-title">📊 Módulo de Análisis</div>', unsafe_allow_html=True)

    archivo = st.file_uploader("📂 Sube tu archivo CSV", type=["csv"])

    if archivo is None:
        st.info("👆 Sube un archivo CSV para comenzar el análisis.")
        return

    df = leer_csv(archivo)
    col_fecha = detectar_fechas(df)

    if col_fecha:
        st.success(f"✅ Columna de fecha detectada: **{col_fecha}**")

    st.markdown('<div class="section-title">Vista previa del dataset</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", f"{len(df):,}")
    c2.metric("Columnas", f"{df.shape[1]}")
    c3.metric("Valores nulos", f"{df.isnull().sum().sum():,}")

    st.dataframe(df.head(10), use_container_width=True)

    if not algoritmos:
        st.warning("⚠️ Selecciona al menos un algoritmo en el panel lateral.")
        return

    st.markdown("---")
    boton = st.button("🚀 Ejecutar Análisis", use_container_width=True)

    if not boton and not st.session_state.analizado:
        return

    if boton:
        with st.spinner("🔄 Ejecutando algoritmos de detección..."):
            resultados, X_original, columnas = ejecutar_deteccion(
                df, algoritmos, umbral_percentil, max_iter
            )

        # ── Ensemble como columna principal
        ensemble = resultados.get("_ensemble", {})
        mascara_ensemble = ensemble.get("anomalias", np.zeros(len(df), dtype=bool))
        votos = ensemble.get("votos", np.zeros(len(df)))

        df["anomaly_ensemble"] = mascara_ensemble
        df["votos_algoritmos"] = votos.astype(int)

        # Añadir columna por cada algoritmo
        for nombre, r in resultados.items():
            if not nombre.startswith("_") and "anomalias" in r:
                col_safe = nombre.replace(" ", "_").replace("(", "").replace(")", "")
                df[f"anomaly_{col_safe}"] = r["anomalias"]

        # Guardar
        id_carga = nuevo_id()
        guardar_resultado(df, archivo.name, id_carga, ", ".join(algoritmos))
        registrar_log(id_carga, archivo.name, len(df), int(mascara_ensemble.sum()), algoritmos)

        st.session_state.analizado = True
        st.session_state.ultima_carga = id_carga
        st.session_state.df_resultado = df
        st.session_state.resultados_raw = resultados
        st.session_state.columnas = columnas
        st.session_state.X_original = X_original

    # ──────────────────────────────────────────
    # MOSTRAR RESULTADOS
    # ──────────────────────────────────────────
    if not st.session_state.analizado or st.session_state.df_resultado is None:
        return

    df = st.session_state.df_resultado
    resultados = st.session_state.resultados_raw
    columnas = st.session_state.columnas

    mascara = df["anomaly_ensemble"].values
    votos = df["votos_algoritmos"].values
    n_anomalias = int(mascara.sum())

    st.markdown("---")
    st.markdown('<div class="section-title">📌 Resultados del Análisis</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total registros", f"{len(df):,}")
    m2.metric("Anomalías (ensemble)", f"{n_anomalias:,}")
    m3.metric("% anomalías", f"{n_anomalias/len(df)*100:.1f}%")
    m4.metric("Registros normales", f"{len(df)-n_anomalias:,}")

    # Tabs de resultados
    tabs = st.tabs(["🔎 Datos", "📊 Visualizaciones", "📐 Estadísticas", "🤖 Por algoritmo", "📥 Exportar"])

    # ── Tab 1: Datos
    with tabs[0]:
        st.subheader("Todos los registros")
        st.dataframe(df, use_container_width=True)

        st.subheader("🚨 Solo anomalías (Ensemble)")
        df_anomalos = df[df["anomaly_ensemble"] == True]
        st.dataframe(df_anomalos, use_container_width=True)

    # ── Tab 2: Visualizaciones
    with tabs[1]:
        # Histograma del primer algoritmo
        primer_algo = [k for k in resultados if not k.startswith("_")]
        if primer_algo:
            r0 = resultados[primer_algo[0]]
            if "scores" in r0:
                st.plotly_chart(
                    grafica_distribucion_scores(r0["scores"], r0["umbral"], r0["nombre"]),
                    use_container_width=True
                )

        # Votos ensemble
        if "_ensemble" in resultados:
            st.plotly_chart(grafica_votos_ensemble(votos), use_container_width=True)

        # Scatter si hay ≥ 2 columnas numéricas
        if len(columnas) >= 2:
            col_x = st.selectbox("Eje X", columnas, index=0)
            col_y = st.selectbox("Eje Y", columnas, index=min(1, len(columnas)-1))
            st.plotly_chart(grafica_scatter_anomalias(df, col_x, col_y, mascara), use_container_width=True)

        # Boxplot
        if columnas:
            st.plotly_chart(grafica_boxplot_anomalias(df, columnas, mascara), use_container_width=True)

    # ── Tab 3: Estadísticas
    with tabs[2]:
        st.subheader("📋 Resumen estadístico")
        st.dataframe(resumen_estadistico(df, columnas), use_container_width=True)

        st.subheader("🔍 Perfil de anomalías vs normales")
        perfil = perfil_anomalia(df, mascara, columnas)
        st.dataframe(perfil, use_container_width=True)

        if len(columnas) >= 2:
            from core.metricas import calcular_correlaciones
            corr = calcular_correlaciones(df, columnas)
            st.plotly_chart(grafica_heatmap_correlacion(corr), use_container_width=True)

    # ── Tab 4: Por algoritmo
    with tabs[3]:
        tabla_comp = comparar_resultados(resultados, len(df))
        st.dataframe(tabla_comp, use_container_width=True)
        st.plotly_chart(grafica_comparacion_barras(tabla_comp), use_container_width=True)

        for nombre, r in resultados.items():
            if nombre.startswith("_") or "error" in r:
                continue
            with st.expander(f"{r.get('icono','🔹')} {r['nombre']} — {r['n_anomalias']} anomalías"):
                st.write(r.get("descripcion", ""))
                if "scores" in r:
                    st.plotly_chart(
                        grafica_distribucion_scores(r["scores"], r["umbral"], r["nombre"]),
                        use_container_width=True
                    )

    # ── Tab 5: Exportar
    with tabs[4]:
        st.subheader("📥 Descargar resultados")

        csv_completo = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar dataset completo con etiquetas",
            csv_completo, f"resultado_completo_{st.session_state.ultima_carga}.csv", "text/csv"
        )

        df_anomalos = df[df["anomaly_ensemble"] == True]
        csv_anomalias = df_anomalos.to_csv(index=False).encode("utf-8")
        st.download_button(
            "🚨 Descargar solo anomalías (Ensemble)",
            csv_anomalias, f"anomalias_{st.session_state.ultima_carga}.csv", "text/csv"
        )

        st.info(f"**ID de carga:** `{st.session_state.ultima_carga}` — guárdalo para referencia en tu tesis.")

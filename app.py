import streamlit as st
import pandas as pd
import re
from sklearn.ensemble import IsolationForest

# =========================
# 🎯 CONFIGURACIÓN
# =========================
st.set_page_config(page_title="Detector de Anomalías en Logs", layout="wide")

st.title("🔍 Detector de Anomalías en Logs")
st.write("Sube un archivo CSV para analizar posibles anomalías usando Machine Learning.")

# =========================
# 📂 SUBIDA DE ARCHIVO
# =========================
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

# =========================
# 🧠 FUNCIONES
# =========================

def extraer_features_texto(texto):
    texto = str(texto).lower()
    return {
        "tiene_error": int("error" in texto),
        "tiene_fail": int("fail" in texto),
        "tiene_timeout": int("timeout" in texto),
        "num_numeros": len(re.findall(r'\d+', texto))
    }

# =========================
# 🚀 PROCESAMIENTO
# =========================

if archivo is not None:
    df = pd.read_csv(archivo)

    st.subheader("📊 Vista previa de los datos")
    st.dataframe(df.head())

    if "analizar" not in st.session_state:
        st.session_state.analizar = False

    if st.button("Analizar"):
        st.session_state.analizar = True

    if st.session_state.analizar:

        # =========================
        # 🔧 CREACIÓN DE FEATURES
        # =========================

        # Longitud del mensaje (última columna como fallback)
        df["longitud_evento"] = df.iloc[:, -1].astype(str).apply(len)

        # Procesamiento de texto
        features_texto = df.iloc[:, -1].apply(extraer_features_texto)
        df_texto = pd.DataFrame(features_texto.tolist())
        df = pd.concat([df, df_texto], axis=1)

        # Manejo de tiempo
        if "Time" in df.columns:
            df["hora"] = pd.to_datetime(df["Time"], errors='coerce').dt.hour
            df["hora"] = df["hora"].fillna(0)
        else:
            df["hora"] = range(len(df))

        # =========================
        # ⚙️ CONFIGURACIÓN DEL MODELO
        # =========================

        st.sidebar.header("⚙️ Configuración")

        contaminacion = st.sidebar.slider(
            "Nivel de anomalías esperado",
            min_value=0.01,
            max_value=0.2,
            value=0.05
        )

        # Variables para ML
        X = df[
            [
                "longitud_evento",
                "hora",
                "tiene_error",
                "tiene_fail",
                "tiene_timeout",
                "num_numeros"
            ]
        ]

        # =========================
        # 🤖 MODELO ML
        # =========================

        model = IsolationForest(
            contamination=contaminacion,
            random_state=42
        )

        df["anomaly"] = model.fit_predict(X)

        # =========================
        # 📊 RESULTADOS
        # =========================

        normales = df[df["anomaly"] == 1]
        anomalias = df[df["anomaly"] == -1]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Datos normales")
            st.dataframe(normales.head(100))

        with col2:
            st.subheader("⚠️ Anomalías detectadas")
            st.dataframe(anomalias.head(100))

        # =========================
        # 📈 MÉTRICAS
        # =========================

        st.subheader("📈 Resumen")
        st.write(f"Total registros: {len(df)}")
        st.write(f"Anomalías detectadas: {len(anomalias)}")

        # =========================
        # 📊 GRÁFICO
        # =========================

        st.subheader("📊 Distribución de anomalías")
        st.bar_chart(df["anomaly"].value_counts())

        # =========================
        # 💾 DESCARGA
        # =========================

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="⬇️ Descargar resultados",
            data=csv,
            file_name="resultado_analisis.csv",
            mime="text/csv"
        )
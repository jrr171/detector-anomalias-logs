import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# 🎯 CONFIG
# =========================
st.set_page_config(page_title="Detector de Anomalías en Logs - Tesis", layout="wide")

st.title("🔍 Detector de Anomalías en Logs (Nivel Tesis)")
st.write("Análisis de anomalías usando múltiples modelos de Machine Learning")

# =========================
# 📂 UPLOAD
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

def preparar_datos(df):
    df["longitud_evento"] = df.iloc[:, -1].astype(str).apply(len)

    # texto
    features_texto = df.iloc[:, -1].apply(extraer_features_texto)
    df_texto = pd.DataFrame(features_texto.tolist())
    df = pd.concat([df, df_texto], axis=1)

    # tiempo
    if "Time" in df.columns:
        df["hora"] = pd.to_datetime(df["Time"], errors='coerce').dt.hour
        df["hora"] = df["hora"].fillna(0)
    else:
        df["hora"] = range(len(df))

    return df

def vectorizar_texto(df):
    vectorizer = TfidfVectorizer(max_features=50)
    X_text = vectorizer.fit_transform(df.iloc[:, -1].astype(str)).toarray()
    return X_text

# =========================
# 🚀 MAIN
# =========================

if archivo is not None:

    df = pd.read_csv(archivo)

    st.subheader("📊 Vista previa")
    st.dataframe(df.head())

    if "analizar" not in st.session_state:
        st.session_state.analizar = False

    if st.button("Analizar"):
        st.session_state.analizar = True

    if st.session_state.analizar:

        # =========================
        # ⚙️ SIDEBAR
        # =========================
        st.sidebar.header("⚙️ Configuración")

        contaminacion = st.sidebar.slider("Contaminación", 0.01, 0.2, 0.05)
        n_estimators = st.sidebar.slider("Árboles (IF)", 50, 300, 100)

        # =========================
        # 🔧 PREPARACIÓN
        # =========================
        df = preparar_datos(df)

        X_base = df[
            [
                "longitud_evento",
                "hora",
                "tiene_error",
                "tiene_fail",
                "tiene_timeout",
                "num_numeros"
            ]
        ].values

        # TF-IDF
        X_text = vectorizar_texto(df)

        # Combinar
        X = np.hstack((X_base, X_text))

        st.write("Dimensión del dataset:", X.shape)

        # =========================
        # 🤖 MODELOS
        # =========================

        # Isolation Forest
        model_if = IsolationForest(
            n_estimators=n_estimators,
            contamination=contaminacion,
            random_state=42
        )
        df["anomaly_if"] = model_if.fit_predict(X)
        df["score_if"] = model_if.decision_function(X)

        # LOF
        model_lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contaminacion
        )
        df["anomaly_lof"] = model_lof.fit_predict(X)

        # One-Class SVM
        model_svm = OneClassSVM(nu=contaminacion)
        df["anomaly_svm"] = model_svm.fit_predict(X)

        # =========================
        # 📊 RESULTADOS
        # =========================

        st.subheader("📊 Resultados (primeros 100)")
        st.dataframe(df.head(100))

        # Conteo
        st.subheader("📈 Comparación de modelos")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("Isolation Forest")
            st.write(df["anomaly_if"].value_counts())

        with col2:
            st.write("LOF")
            st.write(df["anomaly_lof"].value_counts())

        with col3:
            st.write("One-Class SVM")
            st.write(df["anomaly_svm"].value_counts())

        # =========================
        # 📉 VISUALIZACIÓN
        # =========================

        st.subheader("📉 Visualización de anomalías (IF)")

        fig, ax = plt.subplots()
        ax.scatter(df["hora"], df["longitud_evento"], c=df["anomaly_if"])
        ax.set_xlabel("Hora")
        ax.set_ylabel("Longitud evento")

        st.pyplot(fig)

        # =========================
        # 📊 MÉTRICAS
        # =========================

        st.subheader("📊 Métricas")

        total = len(df)
        anom_if = len(df[df["anomaly_if"] == -1])

        st.write(f"Total registros: {total}")
        st.write(f"Anomalías (IF): {anom_if}")
        st.write(f"% Anomalías: {anom_if / total:.2%}")

        # =========================
        # 💾 DESCARGAS
        # =========================

        st.subheader("💾 Descargas")

        csv_all = df.to_csv(index=False).encode("utf-8")
        csv_anom = df[df["anomaly_if"] == -1].to_csv(index=False).encode("utf-8")

        st.download_button("Descargar todo", csv_all, "resultado_completo.csv")
        st.download_button("Descargar anomalías", csv_anom, "anomalias.csv")

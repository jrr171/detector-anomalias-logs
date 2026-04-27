# ================================
# IMPORTACIONES
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

# ================================
# CONFIG
# ================================
archivo_historico = "historico.csv"

# ================================
# FUNCIÓN CSV
# ================================
def leer_csv(archivo):
    try:
        return pd.read_csv(archivo, sep=None, engine='python', encoding='utf-8')
    except:
        try:
            return pd.read_csv(archivo, sep=";", encoding="latin1")
        except:
            return pd.read_csv(archivo, sep=",", encoding="latin1", on_bad_lines='skip')

# ================================
# INTERFAZ
# ================================
st.title("Análisis de Anomalías")

if st.button("🗑️ Borrar histórico"):
    if os.path.exists(archivo_historico):
        os.remove(archivo_historico)
        st.success("Histórico eliminado")
    else:
        st.warning("No hay histórico")

# ================================
# MOSTRAR HISTÓRICO
# ================================
if os.path.exists(archivo_historico):
    df_hist = pd.read_csv(archivo_historico)
    st.subheader("📊 Histórico acumulado")
    st.dataframe(df_hist)

    df_hist["fecha_evaluacion"] = pd.to_datetime(df_hist["fecha_evaluacion"])
    resumen = df_hist.groupby(
        df_hist["fecha_evaluacion"].dt.strftime("%Y-%m-%d %H:%M")
    ).size().reset_index(name="registros")
    st.subheader("📈 Resumen por carga")
    st.dataframe(resumen)
else:
    st.info("Aún no hay datos en el histórico")

# ================================
# SUBIDA DE ARCHIVO
# ================================
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

# ================================
# PROCESAR
# ================================
if archivo is not None:
    df = leer_csv(archivo)
    st.subheader("Vista previa")
    st.dataframe(df.head())

    if st.button("Analizar"):

        # NUMÉRICOS
        X = df.select_dtypes(include=["int64", "float64"])
        if X.shape[1] == 0:
            df["valor"] = range(len(df))
            X = df[["valor"]]

        # ESCALAR
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # MODELO (sklearn en lugar de tensorflow)
        autoencoder = MLPRegressor(
            hidden_layer_sizes=(16, 8, 4, 8, 16),
            activation="relu",
            solver="adam",
            max_iter=100,
            random_state=42
        )

        with st.spinner("Entrenando modelo..."):
            autoencoder.fit(X_scaled, X_scaled)

        # DETECCIÓN
        reconstructions = autoencoder.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        threshold = np.percentile(mse, 97)

        df["anomaly"] = mse > threshold
        df["fecha_evaluacion"] = datetime.now()

        # GUARDAR HISTÓRICO
        if os.path.exists(archivo_historico):
            df_old = pd.read_csv(archivo_historico)
            df_total = pd.concat([df_old, df], ignore_index=True)
        else:
            df_total = df

        df_total.to_csv(archivo_historico, index=False)

        # RESULTADO ACTUAL
        st.subheader("📌 Resultado de esta carga")
        st.dataframe(df)

        # GRÁFICO
        st.subheader("Gráfico de error")
        fig, ax = plt.subplots()
        ax.plot(mse, label="Error (MSE)")
        ax.axhline(y=threshold, linestyle='--', label="Umbral")
        anomaly_points = np.where(mse > threshold)[0]
        ax.scatter(anomaly_points, mse[anomaly_points], color="red", label="Anomalía")
        ax.set_title("Detección de anomalías")
        ax.legend()
        st.pyplot(fig)

        # RESUMEN
        st.write(f"Total registros: {len(df)}")
        st.write(f"Anomalías detectadas: {df['anomaly'].sum()}")

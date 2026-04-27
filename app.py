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
# FUNCIÓN PARA LEER CSV
# ================================
def leer_csv(archivo):
    try:
        df = pd.read_csv(archivo, sep=None, engine='python', encoding='utf-8')
    except:
        try:
            df = pd.read_csv(archivo, sep=";", encoding="latin1")
        except:
            df = pd.read_csv(archivo, sep=",", encoding="latin1", on_bad_lines='skip')
    return df


# ================================
# CONFIG HISTÓRICO
# ================================
archivo_historico = "historico.csv"


# ================================
# INTERFAZ
# ================================
st.title("Análisis de Anomalías")

if st.button("🗑️ Borrar histórico"):
    if os.path.exists(archivo_historico):
        os.remove(archivo_historico)
        st.success("Histórico eliminado correctamente")
    else:
        st.warning("No existe histórico aún")

archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])


# ================================
# PROCESO PRINCIPAL
# ================================
if archivo is not None:

    df = leer_csv(archivo)

    st.subheader("Vista previa")
    st.dataframe(df.head())

    if st.button("Analizar"):

        # 1. SELECCIÓN DE DATOS
        X = df.select_dtypes(include=["int64", "float64"])

        if X.shape[1] == 0:
            df["valor"] = range(len(df))
            X = df[["valor"]]

        # 2. ESCALADO
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. AUTOENCODER con sklearn
        autoencoder = MLPRegressor(
            hidden_layer_sizes=(16, 8, 4, 8, 16),
            activation="relu",
            solver="adam",
            max_iter=100,
            random_state=42
        )

        # 4. ENTRENAMIENTO
        with st.spinner("Entrenando modelo..."):
            autoencoder.fit(X_scaled, X_scaled)

        # 5. DETECCIÓN
        reconstructions = autoencoder.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        threshold = np.percentile(mse, 97)

        df["anomaly"] = mse > threshold
        df["fecha_evaluacion"] = datetime.now()

        # 6. GUARDAR HISTÓRICO
        if os.path.exists(archivo_historico):
            df_existente = pd.read_csv(archivo_historico)
            df_total = pd.concat([df_existente, df], ignore_index=True)
            df_total = df_total.drop_duplicates()
        else:
            df_total = df

        df_total.to_csv(archivo_historico, index=False)

        # 7. MOSTRAR ESTA CARGA
        st.subheader("📌 Datos de esta carga")
        st.dataframe(df)

        # 8. GRÁFICO
        st.subheader("Gráfico de error")

        fig, ax = plt.subplots()
        ax.plot(mse, label="Error (MSE)")
        ax.axhline(y=threshold, linestyle='--', label="Umbral")

        anomaly_points = np.where(mse > threshold)[0]
        ax.scatter(anomaly_points, mse[anomaly_points], color="red", label="Anomalía")

        ax.set_title("Detección de anomalías")
        ax.legend()
        st.pyplot(fig)

        # 9. RESUMEN
        st.write(f"Total registros: {len(df)}")

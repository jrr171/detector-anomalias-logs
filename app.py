# ================================
# IMPORTACIONES
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor


# ================================
# INTERFAZ
# ================================
st.title("Detector de Anomalías con Deep Learning (Autoencoder)")

archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])


# ================================
# PROCESO PRINCIPAL
# ================================
if archivo is not None:

    try:
        df = pd.read_csv(archivo, sep=None, engine="python", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(archivo, sep=None, engine="python", encoding="latin-1")
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        st.stop()

    st.subheader("Vista previa de datos")
    st.dataframe(df.head())

    if st.button("Analizar"):

        # ================================
        # 1. SELECCIÓN DE DATOS
        # ================================
        X = df.select_dtypes(include=["int64", "float64"])

        if X.shape[1] == 0:
            df["valor"] = range(len(df))
            X = df[["valor"]]

        # ================================
        # 2. ESCALADO
        # ================================
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # ================================
        # 3. AUTOENCODER con sklearn
        # ================================
        input_dim = X_scaled.shape[1]

        # MLPRegressor como autoencoder: entrada → capas reducidas → salida igual a entrada
        autoencoder = MLPRegressor(
            hidden_layer_sizes=(16, 8, 4, 8, 16),
            activation="relu",
            solver="adam",
            max_iter=100,
            random_state=42
        )

        # ================================
        # 4. ENTRENAMIENTO
        # ================================
        with st.spinner("Entrenando modelo..."):
            autoencoder.fit(X_scaled, X_scaled)

        # ================================
        # 5. DETECCIÓN
        # ================================
        reconstructions = autoencoder.predict(X_scaled)

        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)

        threshold = np.percentile(mse, 97)

        df["anomaly"] = mse > threshold

        # ================================
        # 6. GRÁFICO DE ERROR
        # ================================
        st.subheader("Gráfico de error de reconstrucción")

        fig, ax = plt.subplots()

        ax.plot(mse, label="Error (MSE)")
        ax.axhline(y=threshold, linestyle='--', label="Umbral")

        anomaly_points = np.where(mse > threshold)[0]
        ax.scatter(anomaly_points, mse[anomaly_points], color="red", label="Anomalía")

        ax.set_title("Detección de anomalías")
        ax.set_xlabel("Registros")
        ax.set_ylabel("Error")
        ax.legend()

        st.pyplot(fig)

        # ================================
        # 7. RESULTADOS
        # ================================
        normales  = df[df["anomaly"] == False]
        anomalias = df[df["anomaly"] == True]

        st.subheader("Datos normales")
        st.dataframe(normales)

        st.subheader("⚠️ Anomalías detectadas")
        st.dataframe(anomalias)

        # ================================
        # 8. RESUMEN
        # ================================
        st.subheader("Resumen")
        st.write(f"Total registros: {len(df)}")
        st.write(f"Anomalías detectadas: {len(anomalias)}")

        # ================================
        # 9. DISTRIBUCIÓN
        # ================================
        st.subheader("Distribución de anomalías")
        st.bar_chart(df["anomaly"].value_counts())

        # ================================
        # 10. DESCARGA
        # ================================
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Descargar resultados",
            data=csv,
            file_name="resultado_analisis.csv",
            mime="text/csv"
        )

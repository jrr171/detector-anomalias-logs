# ================================
# IMPORTACIONES
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import uuid

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor


# ================================
# CONFIG
# ================================
archivo_historico = "historico.csv"


# ================================
# FUNCIÓN PARA LEER CSV
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
st.title("Sistema de Detección de Anomalías")

if st.button("🗑️ Borrar histórico"):
    if os.path.exists(archivo_historico):
        os.remove(archivo_historico)
        st.success("Histórico eliminado")
    else:
        st.warning("No hay histórico")


# ================================
# HISTÓRICO (RESUMEN + FILTRO)
# ================================
if os.path.exists(archivo_historico):

    df_hist = pd.read_csv(archivo_historico)

    df_hist["fecha_evaluacion"] = pd.to_datetime(df_hist["fecha_evaluacion"], errors="coerce")
    df_hist["fecha_evaluacion"] = df_hist["fecha_evaluacion"].dt.strftime("%Y-%m-%d %H:%M:%S")

    st.subheader("📈 Resumen por carga")

    resumen = df_hist.groupby(
        ["id_carga", "archivo_nombre", "fecha_evaluacion"]
    ).size().reset_index(name="registros")

    st.dataframe(resumen)

    st.subheader("🔎 Ver anomalías por carga")

    cargas = resumen["id_carga"].unique()
    carga_sel = st.selectbox("Selecciona una carga", cargas)

    df_filtrado = df_hist[df_hist["id_carga"] == carga_sel]
    anomalias = df_filtrado[df_filtrado["anomaly"] == True]

    st.subheader("🚨 Anomalías de la carga seleccionada")
    st.dataframe(anomalias)
    st.write(f"Total anomalías: {len(anomalias)}")

else:
    st.info("Aún no hay datos en el histórico")


# ================================
# SUBIR ARCHIVO
# ================================
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])


# ================================
# PROCESAR ARCHIVO
# ================================
if archivo is not None:

    df = leer_csv(archivo)

    st.subheader("Vista previa")
    st.dataframe(df.head())

    if st.button("Analizar"):

        # DETECTAR COLUMNA FECHA
        meses_dict = {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
            5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
            9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
        }

        columna_fecha = None
        for col in df.columns:
            if "fecha" in col.lower():
                columna_fecha = col
                break

        if columna_fecha:
            try:
                df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors="coerce")
                df["anio"] = df[columna_fecha].dt.year
                df["mes"] = df[columna_fecha].dt.month
                df["dia"] = df[columna_fecha].dt.day
                df["mes_nombre"] = df["mes"].map(meses_dict)
                st.success(f"Columna de fecha detectada: {columna_fecha}")
            except:
                st.warning("No se pudo procesar la columna de fecha")
        else:
            st.info("No se encontró columna de fecha")

        # DATOS NUMÉRICOS
        X = df.select_dtypes(include=["int64", "float64"])

        if X.shape[1] == 0:
            df["valor"] = range(len(df))
            X = df[["valor"]]

        # ESCALADO
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # AUTOENCODER con sklearn
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
        df["id_carga"] = str(uuid.uuid4())
        df["archivo_nombre"] = archivo.name
        df["fecha_evaluacion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

        st.write(f"Total registros: {len(df)}")
        st.write(f"Anomalías detectadas: {df['anomaly'].sum()}")

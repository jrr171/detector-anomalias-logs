# Importar librerías principales
import streamlit as st              # Framework para crear la interfaz web
import pandas as pd                # Manejo de datos (CSV)
import numpy as np                 # Operaciones matemáticas

# Librerías de Deep Learning (Autoencoder)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Librería para escalar datos (muy importante en redes neuronales)
from sklearn.preprocessing import MinMaxScaler


# Título de la aplicación
st.title("Detector de Anomalías con Deep Learning (Autoencoder)")


# Subir archivo CSV desde la interfaz
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])


# Verificar si el usuario subió un archivo
if archivo is not None:

    # Leer el archivo CSV
    df = pd.read_csv(archivo)

    # Mostrar vista previa
    st.subheader("Vista previa de datos")
    st.dataframe(df.head())


    # Botón para iniciar análisis
    if st.button("Analizar"):

        # ================================
        # 1. SELECCIÓN DE DATOS
        # ================================

        # Seleccionar solo columnas numéricas
        X = df.select_dtypes(include=["int64", "float64"])

        # Si no hay columnas numéricas, crear una artificial
        if X.shape[1] == 0:
            df["valor"] = range(len(df))
            X = df[["valor"]]


        # ================================
        # 2. ESCALADO DE DATOS
        # ================================

        # Escalar los datos entre 0 y 1
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)


        # ================================
        # 3. CREACIÓN DEL AUTOENCODER
        # ================================

        # Número de variables de entrada
        input_dim = X_scaled.shape[1]

        # Capa de entrada
        input_layer = Input(shape=(input_dim,))

        # Codificador (reduce información)
        encoded = Dense(8, activation="relu")(input_layer)
        encoded = Dense(4, activation="relu")(encoded)

        # Decodificador (reconstruye datos)
        decoded = Dense(8, activation="relu")(encoded)
        decoded = Dense(input_dim, activation="sigmoid")(decoded)

        # Definir modelo completo
        autoencoder = Model(inputs=input_layer, outputs=decoded)

        # Compilar modelo (optimizador + función de error)
        autoencoder.compile(optimizer="adam", loss="mse")


        # ================================
        # 4. ENTRENAMIENTO
        # ================================

        # Mostrar mensaje mientras entrena
        with st.spinner("Entrenando modelo..."):
            autoencoder.fit(
                X_scaled, X_scaled,   # entrada = salida (reconstrucción)
                epochs=20,
                batch_size=32,
                verbose=0
            )


        # ================================
        # 5. DETECCIÓN DE ANOMALÍAS
        # ================================

        # Reconstruir datos
        reconstructions = autoencoder.predict(X_scaled)

        # Calcular error de reconstrucción (MSE)
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)

        # Definir umbral (percentil 95)
        threshold = np.percentile(mse, 95)

        # Clasificar anomalías
        df["anomaly"] = mse > threshold


        # ================================
        # 6. RESULTADOS
        # ================================

        # Separar datos normales y anómalos
        normales = df[df["anomaly"] == False]
        anomalias = df[df["anomaly"] == True]

        # Mostrar resultados
        st.subheader("Datos normales")
        st.dataframe(normales)

        st.subheader("⚠️ Anomalías detectadas")
        st.dataframe(anomalias)


        # ================================
        # 7. RESUMEN
        # ================================

        st.subheader("Resumen")
        st.write(f"Total registros: {len(df)}")
        st.write(f"Anomalías detectadas: {len(anomalias)}")


        # ================================
        # 8. GRÁFICO
        # ================================

        st.subheader("Distribución de anomalías")
        st.bar_chart(df["anomaly"].value_counts())


        # ================================
        # 9. DESCARGA DE RESULTADOS
        # ================================

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Descargar resultados",
            data=csv,
            file_name="resultado_analisis.csv",
            mime="text/csv"
        )

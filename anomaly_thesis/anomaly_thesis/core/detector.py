"""
core/detector.py
================
Motor principal de detección de anomalías.
Implementa múltiples algoritmos y genera métricas comparativas.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def preparar_datos(df: pd.DataFrame):
    """
    Extrae columnas numéricas y aplica escalado robusto.
    Retorna (X_original, X_scaled, columnas_usadas)
    """
    X = df.select_dtypes(include=["int64", "float64"]).copy()

    # Eliminar columnas internas si existen
    cols_excluir = ["anio", "mes", "dia"]
    X = X.drop(columns=[c for c in cols_excluir if c in X.columns], errors="ignore")

    if X.shape[1] == 0:
        X = pd.DataFrame({"indice": range(len(df))})

    # Imputar NaN con mediana
    X = X.fillna(X.median())

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, list(X.columns)


# ─────────────────────────────────────────────
# ALGORITMOS
# ─────────────────────────────────────────────

def detectar_autoencoder(X_scaled, percentil=97, max_iter=100):
    """
    Autoencoder implementado con MLPRegressor.
    Arquitectura tipo encoder-decoder simétrico.
    """
    n_features = X_scaled.shape[1]
    dim_cuello = max(2, n_features // 4)

    capas = (
        min(64, n_features * 8),
        min(32, n_features * 4),
        dim_cuello,
        min(32, n_features * 4),
        min(64, n_features * 8)
    )

    modelo = MLPRegressor(
        hidden_layer_sizes=capas,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False
    )

    modelo.fit(X_scaled, X_scaled)
    reconstrucciones = modelo.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstrucciones, 2), axis=1)

    umbral = np.percentile(mse, percentil)
    anomalias = mse > umbral

    return {
        "nombre": "Autoencoder (MLP)",
        "scores": mse,
        "anomalias": anomalias,
        "umbral": umbral,
        "n_anomalias": int(anomalias.sum()),
        "descripcion": f"Red neuronal encoder-decoder. Cuello de botella: {dim_cuello} neuronas.",
        "icono": "🧠"
    }


def detectar_isolation_forest(X_scaled, percentil=97):
    """
    Isolation Forest: aísla anomalías con árboles aleatorios.
    """
    contaminacion = (100 - percentil) / 100

    modelo = IsolationForest(
        contamination=contaminacion,
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    preds = modelo.fit_predict(X_scaled)
    scores = -modelo.score_samples(X_scaled)
    anomalias = preds == -1

    return {
        "nombre": "Isolation Forest",
        "scores": scores,
        "anomalias": anomalias,
        "umbral": np.percentile(scores, percentil),
        "n_anomalias": int(anomalias.sum()),
        "descripcion": "Aísla outliers usando árboles de decisión aleatorios. Muy eficiente en alta dimensionalidad.",
        "icono": "🌲"
    }


def detectar_lof(X_scaled, percentil=97):
    """
    Local Outlier Factor: compara densidad local de cada punto.
    """
    contaminacion = (100 - percentil) / 100
    n_vecinos = min(20, max(5, len(X_scaled) // 10))

    modelo = LocalOutlierFactor(
        n_neighbors=n_vecinos,
        contamination=contaminacion,
        novelty=False,
        n_jobs=-1
    )

    preds = modelo.fit_predict(X_scaled)
    scores = -modelo.negative_outlier_factor_
    anomalias = preds == -1

    return {
        "nombre": "LOF",
        "scores": scores,
        "anomalias": anomalias,
        "umbral": np.percentile(scores, percentil),
        "n_anomalias": int(anomalias.sum()),
        "descripcion": f"Compara densidad local con {n_vecinos} vecinos. Ideal para anomalías contextuales.",
        "icono": "📍"
    }


def detectar_zscore(df_original, X, percentil=97):
    """
    Z-Score: detecta valores estadísticamente extremos por columna.
    """
    z_scores = np.abs(stats.zscore(X, nan_policy="omit"))
    score_max = np.nanmax(z_scores, axis=1)

    umbral = np.percentile(score_max, percentil)
    anomalias = score_max > umbral

    return {
        "nombre": "Z-Score",
        "scores": score_max,
        "anomalias": anomalias,
        "umbral": umbral,
        "n_anomalias": int(anomalias.sum()),
        "descripcion": "Mide desviaciones estándar respecto a la media. Simple pero efectivo para distribuciones normales.",
        "icono": "📊"
    }


def detectar_iqr(X, percentil=97):
    """
    IQR (Rango Intercuartílico): detecta outliers por columna.
    """
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Score = cuántas columnas son outliers por fila
    es_outlier = (X < lower) | (X > upper)
    score = es_outlier.sum(axis=1).astype(float)

    umbral = np.percentile(score, percentil)
    anomalias = score > umbral

    return {
        "nombre": "IQR",
        "scores": score,
        "anomalias": anomalias,
        "umbral": umbral,
        "n_anomalias": int(anomalias.sum()),
        "descripcion": "Detecta valores fuera del rango intercuartílico. Robusto ante distribuciones asimétricas.",
        "icono": "📐"
    }


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────

ALGORITMOS_DISPONIBLES = {
    "Autoencoder (MLP)": detectar_autoencoder,
    "Isolation Forest": detectar_isolation_forest,
    "LOF": detectar_lof,
    "Z-Score": detectar_zscore,
    "IQR": detectar_iqr,
}


def ejecutar_deteccion(df: pd.DataFrame, algoritmos: list, percentil=97, max_iter=100):
    """
    Ejecuta los algoritmos seleccionados y retorna resultados consolidados.
    """
    X_original, X_scaled, columnas = preparar_datos(df)
    resultados = {}

    for nombre in algoritmos:
        try:
            if nombre == "Autoencoder (MLP)":
                r = detectar_autoencoder(X_scaled, percentil, max_iter)
            elif nombre == "Isolation Forest":
                r = detectar_isolation_forest(X_scaled, percentil)
            elif nombre == "LOF":
                r = detectar_lof(X_scaled, percentil)
            elif nombre == "Z-Score":
                r = detectar_zscore(df, X_original.values, percentil)
            elif nombre == "IQR":
                r = detectar_iqr(X_original.values, percentil)
            else:
                continue
            resultados[nombre] = r
        except Exception as e:
            resultados[nombre] = {"error": str(e), "nombre": nombre}

    # Votación por consenso (ensemble)
    if len(resultados) > 1:
        votos = np.zeros(len(df))
        for r in resultados.values():
            if "anomalias" in r:
                votos += r["anomalias"].astype(int)

        mayoria = len(resultados) // 2 + 1
        ensemble_anomalias = votos >= mayoria

        resultados["_ensemble"] = {
            "nombre": "Consenso (Ensemble)",
            "anomalias": ensemble_anomalias,
            "votos": votos,
            "n_anomalias": int(ensemble_anomalias.sum()),
        }

    return resultados, X_original, columnas

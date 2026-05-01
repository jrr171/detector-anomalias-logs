"""
core/metricas.py
================
Cálculo de métricas de evaluación y estadísticas descriptivas.
"""

import numpy as np
import pandas as pd
from scipy import stats


def resumen_estadistico(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """
    Genera tabla de estadísticas descriptivas avanzadas.
    """
    X = df[columnas] if all(c in df.columns for c in columnas) else df.select_dtypes(include="number")

    resumen = pd.DataFrame({
        "Media": X.mean(),
        "Mediana": X.median(),
        "Std": X.std(),
        "Min": X.min(),
        "Max": X.max(),
        "Asimetría": X.skew(),
        "Curtosis": X.kurtosis(),
        "Nulos": X.isnull().sum(),
        "% Nulos": (X.isnull().mean() * 100).round(2),
    })

    return resumen.round(4)


def calcular_correlaciones(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """
    Matriz de correlación de Pearson.
    """
    X = df[columnas] if all(c in df.columns for c in columnas) else df.select_dtypes(include="number")
    return X.corr().round(3)


def comparar_resultados(resultados: dict, n_total: int) -> pd.DataFrame:
    """
    Tabla comparativa de todos los algoritmos ejecutados.
    """
    filas = []
    for nombre, r in resultados.items():
        if nombre.startswith("_") or "error" in r:
            continue
        pct = (r["n_anomalias"] / n_total * 100) if n_total > 0 else 0
        filas.append({
            "Algoritmo": r.get("icono", "") + " " + r["nombre"],
            "Anomalías detectadas": r["n_anomalias"],
            "% del total": f"{pct:.1f}%",
            "Descripción": r.get("descripcion", "—"),
        })

    return pd.DataFrame(filas)


def perfil_anomalia(df: pd.DataFrame, mascara_anomalia: np.ndarray, columnas: list) -> pd.DataFrame:
    """
    Compara media de normales vs anomalías por columna.
    """
    X = df[columnas] if all(c in df.columns for c in columnas) else df.select_dtypes(include="number")

    normales = X[~mascara_anomalia]
    anomalos = X[mascara_anomalia]

    perfil = pd.DataFrame({
        "Media (normal)": normales.mean(),
        "Media (anomalía)": anomalos.mean(),
        "Diferencia %": ((anomalos.mean() - normales.mean()) / (normales.mean().abs() + 1e-9) * 100).round(1),
        "p-valor (t-test)": [
            round(stats.ttest_ind(normales[c].dropna(), anomalos[c].dropna()).pvalue, 4)
            if len(anomalos[c].dropna()) > 1 else None
            for c in X.columns
        ]
    })

    perfil["Significativo"] = perfil["p-valor (t-test)"].apply(
        lambda p: "✅ Sí" if p is not None and p < 0.05 else "❌ No"
    )

    return perfil.round(4)


def indice_novedad(scores: np.ndarray) -> dict:
    """
    Calcula métricas sobre la distribución de scores de anomalía.
    """
    return {
        "score_medio": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
        "score_p95": float(np.percentile(scores, 95)),
        "score_p99": float(np.percentile(scores, 99)),
        "coef_variacion": float(np.std(scores) / (np.mean(scores) + 1e-9)),
    }

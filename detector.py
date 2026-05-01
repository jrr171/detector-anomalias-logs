"""
core/almacenamiento.py
======================
Gestión del histórico de análisis en CSV local.
"""

import pandas as pd
import os
import uuid
from datetime import datetime
import pytz

ARCHIVO_HISTORICO = "historico.csv"
ARCHIVO_LOG = "log_analisis.csv"
ZONA_PERU = pytz.timezone("America/Lima")


def timestamp_peru() -> str:
    return datetime.now(ZONA_PERU).strftime("%Y-%m-%d %H:%M:%S")


def nuevo_id() -> str:
    return str(uuid.uuid4())[:8].upper()


def guardar_resultado(df: pd.DataFrame, nombre_archivo: str, id_carga: str, algoritmo_principal: str):
    """
    Guarda el DataFrame con resultados al histórico.
    """
    df_guardar = df.copy()
    df_guardar["id_carga"] = id_carga
    df_guardar["archivo_nombre"] = nombre_archivo
    df_guardar["fecha_evaluacion"] = timestamp_peru()
    df_guardar["algoritmo_principal"] = algoritmo_principal

    if os.path.exists(ARCHIVO_HISTORICO):
        df_old = pd.read_csv(ARCHIVO_HISTORICO)
        df_total = pd.concat([df_old, df_guardar], ignore_index=True)
    else:
        df_total = df_guardar

    df_total.to_csv(ARCHIVO_HISTORICO, index=False)
    return df_guardar


def registrar_log(id_carga: str, nombre_archivo: str, n_registros: int,
                  n_anomalias: int, algoritmos: list):
    """
    Guarda un registro resumido de cada análisis en el log.
    """
    entrada = {
        "id_carga": id_carga,
        "archivo": nombre_archivo,
        "fecha": timestamp_peru(),
        "registros": n_registros,
        "anomalias": n_anomalias,
        "pct_anomalias": round(n_anomalias / n_registros * 100, 2) if n_registros > 0 else 0,
        "algoritmos": ", ".join(algoritmos),
    }

    df_entrada = pd.DataFrame([entrada])

    if os.path.exists(ARCHIVO_LOG):
        df_log = pd.read_csv(ARCHIVO_LOG)
        df_log = pd.concat([df_log, df_entrada], ignore_index=True)
    else:
        df_log = df_entrada

    df_log.to_csv(ARCHIVO_LOG, index=False)


def cargar_historico() -> pd.DataFrame | None:
    if os.path.exists(ARCHIVO_HISTORICO):
        return pd.read_csv(ARCHIVO_HISTORICO)
    return None


def cargar_log() -> pd.DataFrame | None:
    if os.path.exists(ARCHIVO_LOG):
        return pd.read_csv(ARCHIVO_LOG)
    return None


def borrar_historico():
    for f in [ARCHIVO_HISTORICO, ARCHIVO_LOG]:
        if os.path.exists(f):
            os.remove(f)


def exportar_anomalias(df_resultado: pd.DataFrame, id_carga: str) -> bytes:
    """
    Retorna CSV de anomalías de una carga específica como bytes.
    """
    df_hist = cargar_historico()
    if df_hist is None:
        return b""

    filtrado = df_hist[df_hist["id_carga"] == id_carga]
    anomalias = filtrado[filtrado.get("anomaly_ensemble", filtrado.get("anomaly", False)) == True]
    return anomalias.to_csv(index=False).encode("utf-8")

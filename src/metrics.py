"""Persona B: metricas de deteccion."""
import pandas as pd


def calcular_metricas(df: pd.DataFrame) -> dict:
    """Tiempo de deteccion y tasa de falsos positivos.

    Requiere columnas: t, sensor_id, es_ataque, alerta.
    """
    atacadas = df[df["es_ataque"]]
    limpias = df[~df["es_ataque"]]

    # Deteccion: alguna alerta durante el ataque
    detectado = bool(atacadas["alerta"].any()) if len(atacadas) else False

    # Retraso: pasos desde el inicio del ataque hasta la primera alerta
    retraso = None
    if detectado:
        t_inicio = int(atacadas["t"].min())
        t_alerta = int(atacadas.loc[atacadas["alerta"], "t"].min())
        retraso = t_alerta - t_inicio

    # Recall: fraccion de lecturas atacadas marcadas
    recall = float(atacadas["alerta"].mean()) if len(atacadas) else 0.0

    # Falsos positivos: alertas sobre lecturas limpias
    tasa_fp = float(limpias["alerta"].mean()) if len(limpias) else 0.0

    # Sensores limpios con alguna alerta (convertidos a int nativo de Python)
    sensores_atacados = set(atacadas["sensor_id"].unique())
    limpios_con_alerta = sorted(
        int(s) for s in df.loc[df["alerta"], "sensor_id"].unique()
        if s not in sensores_atacados
    )

    return {
        "detectado": detectado,
        "retraso_deteccion_pasos": retraso,
        "recall": round(recall, 4),
        "tasa_falsos_positivos": round(tasa_fp, 4),
        "sensores_limpios_con_alerta": limpios_con_alerta,
    }
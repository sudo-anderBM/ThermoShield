import pandas as pd
from src.metrics import calcular_metricas


def _df(alertas_atacado, alertas_limpio):
    n = len(alertas_atacado)
    atacado = pd.DataFrame({"t": range(n), "sensor_id": 7,
                            "es_ataque": [t >= 5 for t in range(n)],
                            "alerta": alertas_atacado})
    limpio = pd.DataFrame({"t": range(n), "sensor_id": 0,
                           "es_ataque": False, "alerta": alertas_limpio})
    return pd.concat([atacado, limpio], ignore_index=True)


def test_deteccion_con_retraso():
    a = [False] * 8 + [True] * 2          # ataque desde t=5, alerta en t=8
    df = _df(a, [False] * 10)
    m = calcular_metricas(df)
    assert m["detectado"] is True
    assert m["retraso_deteccion_pasos"] == 3
    assert m["tasa_falsos_positivos"] == 0.0


def test_no_detectado():
    df = _df([False] * 10, [False] * 10)
    m = calcular_metricas(df)
    assert m["detectado"] is False
    assert m["retraso_deteccion_pasos"] is None


def test_falsos_positivos():
    df = _df([False] * 10, [True] + [False] * 9)
    m = calcular_metricas(df)
    assert m["tasa_falsos_positivos"] > 0
    assert m["sensores_limpios_con_alerta"] == [0]
"""Pruebas de la persona A: physics.simular y detector.detect."""
import numpy as np
import pandas as pd

from src.detector import detect, resumen
from src.physics import ALPHA, DX, simular, stable_dt
from src.sensors import sample_sensors


def test_forma_tiempo_y_x():
    malla = simular(50)
    assert malla.ndim == 3
    assert malla.shape[0] == 51


def test_paso_estable():
    assert ALPHA * stable_dt() / DX ** 2 <= 0.25


def test_calor_se_difunde_desde_frio():
    malla = simular(800, estacionario=False)
    assert np.allclose(malla[0], malla[0].flat[0])        # arranca uniforme
    assert malla[-1].max() > malla[0].max() + 5           # los racks calientan
    assert np.allclose(malla[-1][0, :], malla[0][0, 0])   # pared fija


def test_estacionario_no_cambia():
    malla = simular(100)
    assert np.abs(malla[-1] - malla[0]).max() < 1e-3


def test_contrato_detect():
    df = sample_sensors(simular(300))
    res = detect(df)
    assert {"residual", "alerta"} <= set(res.columns)
    assert len(res) == len(df) and (res.index == df.index).all()
    assert set(df.columns) <= set(res.columns)


def test_sin_ataque_sin_sospechosos():
    res = detect(sample_sensors(simular(500)))
    assert not resumen(res)["sospechoso"].any()


def test_detecta_deriva_y_pico():
    df = sample_sensors(simular(500))
    m = (df.sensor_id == 7) & (df.t >= 200)
    df.loc[m, "temp_medida"] += 0.01 * (df.loc[m, "t"] - 200)
    m = (df.sensor_id == 3) & df.t.between(350, 355)
    df.loc[m, "temp_medida"] += 3.0
    sosp = resumen(detect(df)).query("sospechoso").sensor_id
    assert set(sosp) == {3, 7}


def test_modo_malla():
    malla = simular(300)
    res = detect(sample_sensors(malla), malla=malla)
    assert res["residual"].abs().mean() < 0.5

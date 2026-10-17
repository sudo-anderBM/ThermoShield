"""
Persona A: detector basado en residuales físicos.

Idea: el calor se difunde de forma suave, así que la lectura de un sensor debe
ser consistente con lo que dicen sus vecinos (o con lo que predice la
simulación). Si no lo es, está mintiendo.

    residual = medido - predicho
    z        = (residual - media_calibración) / σ_calibración
    alerta 3σ: |z| > 3                         (ataques bruscos)
    CUSUM    : acumula z para ver derivas       (ataques lentos)

Contrato (congelado):
    detect(df) -> df + residual, alerta

Entrada: DataFrame largo con columnas  t, sensor_id, x, y, temp_medida
         (las columnas extra, como es_ataque, se conservan sin tocar).

Columnas que agrega:
    predicho      temperatura esperada en el sensor
    residual      temp_medida - predicho
    z             residual estandarizado
    alerta_3sigma |z| > umbral
    cusum_pos, cusum_neg, alerta_cusum
    alerta        alerta_3sigma OR alerta_cusum      (columna del contrato)
    aislado       (modo vecinos) sensor descartado como vecino por ser falso

    python -m src.detector     # demo con un ataque de deriva; guarda figuras/detector.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

T, SID, X, Y, MEDIDA = "t", "sensor_id", "x", "y", "temp_medida"


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
def predecir_con_malla(df: pd.DataFrame, malla: np.ndarray) -> np.ndarray:
    """Predicción física: valor de la simulación [t, y, x] en la posición del sensor."""
    t = df[T].to_numpy().clip(0, malla.shape[0] - 1)
    return malla[t, df[Y].to_numpy(), df[X].to_numpy()]


def predecir_con_vecinos(df: pd.DataFrame, k: int = 4, potencia: float = 2.0,
                         excluir=()) -> np.ndarray:
    """
    Predicción por vecinos (leave-one-out): para cada sensor y tiempo, promedio
    ponderado por 1/d^potencia de sus k vecinos más cercanos, sin usarse a sí
    mismo. Funciona porque el campo de calor es suave. El sesgo fijo de la
    interpolación se elimina restando la media de calibración del residual.
    `excluir`: sensores que no se usan como vecinos (ya detectados como falsos).
    """
    sensores = df.groupby(SID)[[X, Y]].first()
    ids = sensores.index.to_numpy()
    pos = sensores.to_numpy(dtype=float)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    d[:, np.isin(ids, list(excluir))] = np.inf

    W = np.zeros_like(d)
    for i in range(len(ids)):
        nn = np.argsort(d[i])[:k]
        nn = nn[np.isfinite(d[i, nn])]
        W[i, nn] = 1.0 / np.maximum(d[i, nn], 1e-9) ** potencia
    W /= W.sum(axis=1, keepdims=True)

    ancho = _ancho(df)[ids]
    return _a_largo(df, ancho.to_numpy() @ W.T, ancho)


def _ancho(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla [t, sensor_id] de temperaturas medidas."""
    return df.pivot_table(index=T, columns=SID, values=MEDIDA).sort_index(axis=1)


def _a_largo(df: pd.DataFrame, pred: np.ndarray, ancho: pd.DataFrame) -> np.ndarray:
    p = pd.DataFrame(pred, index=pd.Index(ancho.index, name=T),
                     columns=pd.Index(ancho.columns, name=SID))
    largo = p.stack().rename("_pred").reset_index()
    return df[[T, SID]].merge(largo, on=[T, SID], how="left")["_pred"].to_numpy()


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------
def cusum(z: np.ndarray, k: float = 0.5, h: float = 10.0):
    """
    CUSUM bilateral sobre una serie estandarizada z.
    k: holgura (en σ), h: umbral de decisión (en σ).
    Devuelve (S_pos, S_neg, alarma). Tras una alarma el acumulador se reinicia.
    """
    s_pos = np.zeros(len(z))
    s_neg = np.zeros(len(z))
    alarma = np.zeros(len(z), dtype=bool)
    sp = sn = 0.0
    for i, zi in enumerate(z):
        if np.isnan(zi):
            zi = 0.0
        sp = max(0.0, sp + zi - k)
        sn = max(0.0, sn - zi - k)
        s_pos[i], s_neg[i] = sp, sn
        if sp > h or sn > h:
            alarma[i] = True
            sp = sn = 0.0
    return s_pos, s_neg, alarma


MIN_ALARMAS_CUSUM = 2   # una sola alarma puede ser error de calibración
Z_EXTREMO = 5.0         # |z| > 5: probabilidad ~6e-7 por lectura con ruido


def es_sospechoso(n_cusum, n_3sigma, n_total, z_max, umbral=3.0, alfa=1e-4):
    """Veredicto por sensor (acepta escalares o arrays)."""
    return ((np.asarray(n_cusum) >= MIN_ALARMAS_CUSUM)
            | demasiadas_alertas(n_3sigma, n_total, umbral, alfa)
            | (np.asarray(z_max) > Z_EXTREMO))


def demasiadas_alertas(n_alertas, n_total, umbral: float = 3.0, alfa: float = 1e-4):
    """
    ¿Son demasiadas alertas 3σ para ser solo ruido? Prueba binomial: con ruido
    gaussiano cada lectura supera el umbral con probabilidad p = 2·(1-Φ(umbral)).
    Devuelve True si P(X >= n_alertas) < alfa.
    """
    p = 2 * norm.sf(umbral)
    return binom.sf(np.asarray(n_alertas) - 1, np.asarray(n_total), p) < alfa


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
def _puntuar(out: pd.DataFrame, t_cal, umbral, cusum_k, cusum_h) -> pd.DataFrame:
    """Calcula residual, z, alerta 3σ y CUSUM a partir de la columna `predicho`."""
    out = out.copy()
    out["residual"] = out[MEDIDA] - out["predicho"]

    n = len(out)
    z, s_pos, s_neg = np.empty(n), np.empty(n), np.empty(n)
    a_cus = np.empty(n, dtype=bool)

    for _, idx in out.groupby(SID).groups.items():
        idx = np.asarray(idx)
        r = out.loc[idx, "residual"].to_numpy()
        cal = r[out.loc[idx, T].to_numpy() < t_cal]
        # σ: MAD de las diferencias consecutivas sobre TODA la serie. Para ruido
        # independiente std(diff) = √2·σ; una deriva lenta casi no cambia las
        # diferencias y un pico solo afecta unas pocas, así que es robusta y
        # usa muchos más datos que la ventana de calibración.
        dif = np.diff(r)
        sigma = max(1.4826 * np.median(np.abs(dif - np.median(dif))) / np.sqrt(2), 1e-6)
        # μ: nivel base del residual (sesgo fijo del predictor), de la ventana
        # de calibración, descartando atípicos.
        med = np.median(cal)
        cerca = np.abs(cal - med) < 4 * sigma
        mu = cal[cerca].mean() if cerca.any() else med
        zi = (r - mu) / sigma
        z[idx] = zi
        s_pos[idx], s_neg[idx], a_cus[idx] = cusum(zi, cusum_k, cusum_h)

    out["z"] = z
    out["alerta_3sigma"] = np.abs(z) > umbral
    out["cusum_pos"] = s_pos
    out["cusum_neg"] = s_neg
    out["alerta_cusum"] = a_cus
    out["alerta"] = out["alerta_3sigma"] | out["alerta_cusum"]
    return out


def detect(
    df: pd.DataFrame,
    malla: np.ndarray | None = None,
    umbral: float = 3.0,
    pasos_calibracion: int | None = None,
    cusum_k: float = 0.5,
    cusum_h: float = 10.0,
    max_aislar: int = 5,
) -> pd.DataFrame:
    """
    Detecta lecturas falsas. Devuelve df + residual, alerta (y columnas extra).

    Predicción usada:
      * si se pasa `malla` (salida de physics.simular): la simulación;
      * si no: los sensores vecinos (no necesita conocer los racks ni la malla).

    umbral: alerta instantánea en σ (por defecto 3σ).
    cusum_k, cusum_h: holgura y umbral del CUSUM en σ. Con k=0.5 y h=10, un
        sensor limpio da ~1 falsa alarma cada 70 000 pasos.
    pasos_calibracion: primeros pasos que se suponen limpios para estimar la
        media y σ del residual de cada sensor (por defecto el primer 20 %).
    max_aislar (modo vecinos): un sensor falso contamina la predicción de sus
        vecinos; se aísla iterativamente al sensor más anómalo (deja de usarse
        como vecino) y se recalcula, hasta max_aislar veces.
    """
    orden_original = df.index
    out = df.sort_values([SID, T]).reset_index().rename(columns={"index": "_idx"})
    if pasos_calibracion is None:
        pasos_calibracion = max(10, int(0.2 * (out[T].max() - out[T].min() + 1)))
    t_cal = out[T].min() + pasos_calibracion
    args = (t_cal, umbral, cusum_k, cusum_h)

    if malla is not None:
        out["predicho"] = predecir_con_malla(out, malla)
        res = _puntuar(out, *args)
    else:
        aislados: list = []
        while True:
            out["predicho"] = predecir_con_vecinos(out, excluir=aislados)
            res = _puntuar(out, *args)
            por_sensor = res.groupby(SID).agg(
                cus=("alerta_cusum", "sum"), n3=("alerta_3sigma", "sum"),
                n=("alerta_3sigma", "size"), zmax=("z", lambda v: np.abs(v).max()))
            por_sensor = por_sensor.drop(index=aislados, errors="ignore")
            malos = por_sensor[es_sospechoso(por_sensor.cus, por_sensor.n3,
                                             por_sensor.n, por_sensor.zmax, umbral)]
            if malos.empty or len(aislados) >= max_aislar:
                break
            aislados.append(malos.sort_values(["cus", "zmax"], ascending=False).index[0])
        res["aislado"] = res[SID].isin(aislados)

    return res.set_index("_idx").loc[orden_original].rename_axis(df.index.name)


def resumen(res: pd.DataFrame, umbral: float = 3.0, alfa: float = 1e-4) -> pd.DataFrame:
    """
    Resumen por sensor: nº de alertas, primer instante de cada tipo y veredicto.
    Un sensor es `sospechoso` si el CUSUM disparó al menos MIN_ALARMAS_CUSUM
    veces, si tiene demasiadas alertas 3σ para ser ruido (prueba binomial, ver
    `demasiadas_alertas`) o si alguna lectura tiene |z| > Z_EXTREMO.
    """
    def primero(g, col):
        t = g.loc[g[col], T]
        return int(t.min()) if len(t) else None

    filas = []
    for sid, g in res.groupby(SID):
        filas.append({
            "sensor_id": sid,
            "alertas_3sigma": int(g["alerta_3sigma"].sum()),
            "alertas_cusum": int(g["alerta_cusum"].sum()),
            "primera_3sigma": primero(g, "alerta_3sigma"),
            "primera_cusum": primero(g, "alerta_cusum"),
            "max_abs_z": round(float(g["z"].abs().max()), 2),
            "sospechoso": bool(es_sospechoso(
                g["alerta_cusum"].sum(), g["alerta_3sigma"].sum(), len(g),
                g["z"].abs().max(), umbral, alfa)),
        })
    return pd.DataFrame(filas).sort_values(
        ["sospechoso", "alertas_cusum", "max_abs_z"], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.physics import simular
    from src.sensors import sample_sensors

    malla = simular(500)
    df = sample_sensors(malla)
    # Ataque de prueba local (el oficial lo implementa B en attacks.py):
    # deriva de +0.01 °C/paso en el sensor 7 desde t=200.
    m = (df.sensor_id == 7) & (df.t >= 200)
    df.loc[m, "temp_medida"] += 0.01 * (df.loc[m, "t"] - 200)

    res = detect(df)
    print(resumen(res).to_string(index=False))

    os.makedirs("figuras", exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for sid, color in [(7, "tab:red"), (0, "tab:gray")]:
        g = res[res.sensor_id == sid]
        ax[0].plot(g.t, g.residual, color=color, lw=0.8, label=f"sensor {sid}")
        ax[1].plot(g.t, g.z, color=color, lw=0.8)
        ax[2].plot(g.t, g.cusum_pos, color=color, lw=0.8)
    ax[1].axhline(3, ls="--", c="k", lw=0.8); ax[1].axhline(-3, ls="--", c="k", lw=0.8)
    ax[2].axhline(10, ls="--", c="k", lw=0.8)
    ax[0].set_ylabel("residual (°C)"); ax[1].set_ylabel("z"); ax[2].set_ylabel("CUSUM+")
    ax[2].set_xlabel("t (paso)"); ax[0].legend(loc="upper left")
    ax[0].set_title("Detector: residual, umbral 3σ y CUSUM (ataque de deriva en t=200)")
    fig.savefig("figuras/detector.png", dpi=130)
    print("Figura guardada en figuras/detector.png")

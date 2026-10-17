"""Persona B: inyecta ataques a las lecturas."""
import pandas as pd


def inject_attack(df: pd.DataFrame, tipo: str, sensor_id: int,
                  t_inicio: int, duracion_replay: int = 50) -> pd.DataFrame:
    """tipo: 'fijo' | 'deriva' | 'replay'. Devuelve df con columna es_ataque."""
    df = df.copy()
    df["es_ataque"] = False

    mask = (df["sensor_id"] == sensor_id) & (df["t"] >= t_inicio)
    df.loc[mask, "es_ataque"] = True

    if tipo == "fijo":
        # el sensor se congela en un valor falso
        df.loc[mask, "temp_medida"] = 18.0

    elif tipo == "deriva":
        # se aleja lentamente de la realidad (el mas dificil de detectar)
        df.loc[mask, "temp_medida"] -= 0.05 * (df.loc[mask, "t"] - t_inicio)

    elif tipo == "replay":
        # repite lecturas viejas en bucle
        viejo = df[(df["sensor_id"] == sensor_id) &
                   (df["t"] >= t_inicio - duracion_replay) &
                   (df["t"] < t_inicio)]["temp_medida"].to_numpy()
        n = mask.sum()
        df.loc[mask, "temp_medida"] = [viejo[i % len(viejo)] for i in range(n)]

    else:
        raise ValueError(f"tipo desconocido: {tipo}")

    return df


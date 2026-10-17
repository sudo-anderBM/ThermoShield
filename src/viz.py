"""Módulo de visualización y animación para ThermoShield."""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.physics import simular
from src.sensors import sample_sensors
from src.attacks import inject_attack
from src.detector import detect

def generar_animacion(df: pd.DataFrame, malla: np.ndarray, sensor_objetivo: int, t_inicio: int, output_path="figuras/animacion_ataque.gif"):
    """Genera y guarda una animación de 2 paneles: Mapa de calor espacial y evolución del sensor."""
    pasos = malla.shape[0]
    fig, (ax_map, ax_ts) = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={'height_ratios': [2, 1]})
    plt.tight_layout(pad=3.0)

    # Coordenadas estáticas de sensores para el scatter
    df_sensores_coords = df[['sensor_id', 'x', 'y']].drop_duplicates().set_index('sensor_id')

    def actualizar(t):
        ax_map.clear()
        ax_ts.clear()

        # 1. Panel superior: Mapa de calor de la malla
        im = ax_map.imshow(malla[t], cmap='hot', origin='lower', vmin=malla.min(), vmax=malla.max())
        ax_map.set_title(f"ThermoShield - Tiempo: {t}s (Ataque en sensor {sensor_objetivo} desde t={t_inicio})")
        ax_map.set_xlabel("X (m)")
        ax_map.set_ylabel("Y (m)")

        # Dibujar sensores en el mapa
        df_t = df[df['t'] == t]
        for sid, row in df_sensores_coords.iterrows():
            # Estado del sensor en este instante
            fila_s = df_t[df_t['sensor_id'] == sid]
            es_atacado = (sid == sensor_objetivo) and (t >= t_inicio)
            alerta = bool(fila_s['alerta'].values[0]) if not fila_s.empty else False

            if es_atacado:
                color, marker, size = 'blue', 'X', 120
            elif alerta:
                color, marker, size = 'red', 'o', 80
            else:
                color, marker, size = 'cyan', 'o', 50

            ax_map.scatter(row['x'] * 10, row['y'] * 10, color=color, marker=marker, s=size, edgecolors='black') # Ajustado a escala de malla si dx=0.1

        # 2. Panel inferior: Residual/Z del sensor objetivo en el tiempo
        df_sensor = df[df['sensor_id'] == sensor_objetivo]
        df_pasado = df_sensor[df_sensor['t'] <= t]
        
        if not df_pasado.empty and 'z' in df_pasado.columns:
            ax_ts.plot(df_pasado['t'], df_pasado['z'], label=f'Sensor {sensor_objetivo} (Z-score)', color='orange')
            ax_ts.axhline(3.0, color='red', linestyle='--', label='Umbral 3σ')
            ax_ts.axvline(t_inicio, color='blue', linestyle=':', label='Inicio ataque')
            ax_ts.set_xlim(0, pasos)
            ax_ts.set_ylim(-1, max(10, df_sensor['z'].max() if not df_sensor['z'].isna().all() else 10))
            ax_ts.set_xlabel("Tiempo (s)")
            ax_ts.set_ylabel("Z-score / Residual")
            ax_ts.legend(loc='upper left', fontsize=8)
            ax_ts.grid(True)

    anim = animation.FuncAnimation(fig, actualizar, frames=range(0, pasos, 5), interval=50)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Intentar guardar como GIF (requiere pillow)
    try:
        anim.save(output_path, writer='pillow', fps=20)
        print(f"[OK] Animacion guardada en: {output_path}")
    except Exception as e:
        print(f"[WARN] No se pudo guardar gif ({e}), guardando imagen estatica de respaldo.")
        plt.savefig("figuras/detector.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Simulacion y visualizacion ThermoShield")
    parser.add_argument("--ataque", type=str, default="deriva", choices=["fijo", "deriva", "replay"])
    parser.add_argument("--sensor", type=int, default=7)
    parser.add_argument("--t_inicio", type=int, default=200)
    parser.add_argument("--carga", type=float, default=0.3)
    args = parser.parse_args()

    print(f"[INFO] Ejecutando simulación: ataque={args.ataque}, sensor={args.sensor}, t={args.t_inicio}, carga={args.carga}")
    malla = simular(500, variacion_carga=args.carga)
    df = sample_sensors(malla)
    df = inject_attack(df, args.ataque, args.sensor, args.t_inicio)
    df = detect(df, malla=malla)

    generar_animacion(df, malla, args.sensor, args.t_inicio)

if __name__ == "__main__":
    main()
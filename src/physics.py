"""
Persona A: simulador de calor 2D (diferencias finitas).

physics.py — Simulación de la ecuación de calor 2D en un cuarto de servidores.

    dT/dt = alpha * (d2T/dx2 + d2T/dy2) + Q(x, y) - k_cool * (T - T_amb)

  * Diferencias finitas explícitas (FTCS) con numpy.
  * Los racks son fuentes de calor Q (°C/s) en rectángulos de la malla.
  * k_cool modela el aire acondicionado (enfriamiento de Newton hacia T_amb).
  * Paredes con temperatura fija T_amb (condición de Dirichlet).

Contrato (congelado):
    malla = simular(pasos)           # array [tiempo, y, x]

Por defecto el cuarto arranca en estado estacionario (el centro de datos ya
está operando). Con estacionario=False arranca frío, a T_amb, y se ve el
calentamiento.

    python -m src.physics            # guarda figuras/heatmap.png
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Parámetros por defecto del cuarto
# ---------------------------------------------------------------------------
NY, NX = 40, 60          # celdas de la malla (y, x)
DX = 0.1                 # tamaño de celda en metros  -> cuarto de 4 m x 6 m
ALPHA = 2e-3             # difusividad térmica efectiva (m^2/s), aire con mezcla
T_AMB = 20.0             # temperatura ambiente / del aire acondicionado (°C)
K_COOL = 0.01            # coeficiente de enfriamiento (1/s)
VARIACION_CARGA = 0.0    # amplitud relativa de la variación de carga (0 = constante)
PERIODOS_CARGA = (180, 67)  # dos periodos (pasos) inconmensurables: el campo no se repite

# Racks: (y0, x0, alto, ancho, potencia en °C/s)
DEFAULT_RACKS = [
    (8, 8, 6, 3, 0.35),
    (8, 25, 6, 3, 0.50),
    (8, 42, 6, 3, 0.30),
    (26, 8, 6, 3, 0.40),
    (26, 25, 6, 3, 0.25),
    (26, 42, 6, 3, 0.55),
]


def heat_sources(ny: int = NY, nx: int = NX, racks=DEFAULT_RACKS) -> np.ndarray:
    """Mapa Q[y, x] con la potencia de cada rack (0 fuera de los racks)."""
    return rack_masks(ny, nx, racks).sum(axis=0)


def rack_masks(ny: int = NY, nx: int = NX, racks=DEFAULT_RACKS) -> np.ndarray:
    """Array [rack, y, x] con la potencia nominal de cada rack por separado."""
    M = np.zeros((len(racks), ny, nx))
    for i, (y0, x0, h, w, p) in enumerate(racks):
        M[i, y0:y0 + h, x0:x0 + w] = p
    return M


def stable_dt(dx: float = DX, alpha: float = ALPHA, safety: float = 0.9) -> float:
    """Paso de tiempo máximo estable para FTCS 2D: dt <= dx^2 / (4 alpha)."""
    return safety * dx ** 2 / (4 * alpha)


def laplacian(T: np.ndarray, dx: float = DX) -> np.ndarray:
    """Laplaciano con diferencias centrales en el interior (bordes = 0)."""
    L = np.zeros_like(T)
    L[1:-1, 1:-1] = (
        T[2:, 1:-1] + T[:-2, 1:-1] + T[1:-1, 2:] + T[1:-1, :-2] - 4 * T[1:-1, 1:-1]
    ) / dx ** 2
    return L


def simular(
    pasos: int,
    ny: int = NY,
    nx: int = NX,
    dx: float = DX,
    alpha: float = ALPHA,
    dt: float | None = None,
    racks=DEFAULT_RACKS,
    t_amb: float = T_AMB,
    k_cool: float = K_COOL,
    T0: np.ndarray | None = None,
    save_every: int = 1,
    estacionario: bool = True,
    variacion_carga: float = VARIACION_CARGA,
    periodos_carga=PERIODOS_CARGA,
    seed: int = 0,
) -> np.ndarray:
    """
    Simula la difusión de calor durante `pasos` pasos.

    Devuelve un array de forma [tiempo, y, x] con pasos // save_every + 1
    fotogramas (el primero es la condición inicial). Con save_every=1
    (por defecto) la forma es [pasos + 1, ny, nx].

    estacionario: si es True y no se pasa T0, primero calienta el cuarto hasta
    su régimen de operación y la simulación arranca desde ahí.

    variacion_carga: la potencia de cada rack oscila como
        p · (1 + variacion_carga · [0.6·sin(2πn/P1 + φ1) + 0.4·sin(2πn/P2 + φ2)]),
    con periodos P1, P2 = periodos_carga y fases aleatorias por rack (fijas
    por `seed`). Por defecto es 0 (carga constante). Con carga variable el
    campo cambia en el tiempo y un ataque de replay (repetir lecturas viejas)
    se vuelve físicamente detectable; con carga constante no lo es, porque la
    temperatura real no cambia.
    """
    if dt is None:
        dt = stable_dt(dx, alpha)
    if alpha * dt / dx ** 2 > 0.25:
        raise ValueError(
            f"Inestable: alpha*dt/dx^2 = {alpha * dt / dx**2:.3f} > 0.25. "
            f"Usa dt <= {dx**2 / (4 * alpha):.4f} s."
        )

    M = rack_masks(ny, nx, racks)
    fases = np.random.default_rng(seed).uniform(0, 2 * np.pi, (2, len(racks)))
    P1, P2 = periodos_carga

    def carga(n):
        f = 1 + variacion_carga * (0.6 * np.sin(2 * np.pi * n / P1 + fases[0])
                                   + 0.4 * np.sin(2 * np.pi * n / P2 + fases[1]))
        return np.tensordot(f, M, axes=1)

    def paso(T, n):
        Q = carga(n)
        T = T + dt * (alpha * laplacian(T, dx) + Q - k_cool * (T - t_amb))
        # Paredes: temperatura fija del aire acondicionado
        T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = t_amb
        return T

    if T0 is not None:
        T = T0.astype(float).copy()
    else:
        T = np.full((ny, nx), t_amb, dtype=float)
        if estacionario:
            # Calentamiento hasta régimen (≫ tiempo de respuesta), terminando en n = 0
            for n in range(-4000, 0):
                T = paso(T, n)

    frames = [T.copy()]
    for n in range(1, pasos + 1):
        T = paso(T, n)
        if n % save_every == 0:
            frames.append(T.copy())
    return np.stack(frames)  # [tiempo, y, x]


simulate = simular  # alias en inglés


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def plot_heatmap(field: np.ndarray, path: str = "figuras/heatmap.png", times=None, sensors=None):
    """
    Dibuja varios instantes del campo de temperatura para comprobar la difusión.
    `times` son índices de tiempo; por defecto 4 instantes repartidos.
    `sensors`: lista opcional de posiciones (x, y) para marcarlas.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    nt = field.shape[0]
    if times is None:
        times = np.linspace(0, nt - 1, 4).astype(int)
    vmin, vmax = field.min(), field.max()

    fig, axes = plt.subplots(1, len(times), figsize=(4.2 * len(times), 3.6),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, times):
        im = ax.imshow(field[t], cmap="inferno", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"paso {t}")
        ax.set_xticks([]); ax.set_yticks([])
        if sensors is not None:
            sx, sy = zip(*sensors)
            ax.scatter(sx, sy, s=14, c="cyan", marker="o", edgecolors="k", linewidths=0.4)
    fig.colorbar(im, ax=axes, label="Temperatura (°C)", shrink=0.9)
    fig.suptitle("Difusión de calor en el cuarto de servidores")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    # Desde frío para ver cómo se difunde el calor de los racks
    field = simular(3000, estacionario=False)
    print("forma:", field.shape, "| T min/max final:",
          round(field[-1].min(), 2), round(field[-1].max(), 2))
    plot_heatmap(field, "figuras/heatmap.png")
    print("Mapa de calor guardado en figuras/heatmap.png")

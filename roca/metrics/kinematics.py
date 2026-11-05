"""Marco de referencia del caballo: centro, rumbo, velocidad."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from ..skeleton import hidx


def _nanfill(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    t = np.arange(len(x))
    for k in range(x.shape[1]):
        bad = np.isnan(x[:, k])
        if bad.any() and (~bad).any():
            x[bad, k] = np.interp(t[bad], t[~bad], x[~bad, k])
    return x


def horse_frame(H: np.ndarray, fps: float) -> dict:
    """H (F,J,3). Devuelve centro xy, vectores fwd/left (F,3), velocidad (F,) y rumbo."""
    # centro del tronco = media de 4 puntos rígidos (reduce el ruido de reconstrucción)
    torso = [hidx(n) for n in ("withers", "mid_back", "croup", "chest")]
    center = _nanfill(np.nanmean(H[:, torso, :2], axis=1))
    w = _nanfill(H[:, hidx("withers"), :2])
    c = _nanfill(H[:, hidx("croup"), :2])
    axis = w - c
    axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9
    fwd = np.column_stack([axis, np.zeros(len(axis))])
    left = np.column_stack([-axis[:, 1], axis[:, 0], np.zeros(len(axis))])
    win = int(fps * 0.6) | 1
    cs = savgol_filter(center, win, 2, axis=0)
    # velocidad: derivada de Savitzky-Golay con ventana de 1 s (robusta al ruido)
    vel = savgol_filter(center, int(fps * 1.0) | 1, 2, deriv=1, axis=0) * fps
    speed = np.linalg.norm(vel, axis=1)
    return {"center": center, "center_s": cs, "fwd": fwd, "left": left, "speed": speed,
            "yaw": np.unwrap(np.arctan2(axis[:, 1], axis[:, 0])), "withers": w}

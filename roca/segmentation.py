"""Segmenta la trayectoria del caballo en los ejercicios de la hoja.

Recorre los puntos de paso de protocol.BOUNDARIES en orden: para cada ejercicio
debe visitar primero los puntos "vía" (tolerancia amplia) y luego el punto de
inicio (mínimo local de distancia). Robusto ante figuras mal trazadas.
"""
from __future__ import annotations

import numpy as np

from .protocol import BOUNDARIES

VIA_TOL = 4.5
PT_TOL = 2.5


def segment(xy: np.ndarray, fps: float) -> list[dict]:
    F = len(xy)
    f = 0
    starts = []
    for mov, vias, pt in BOUNDARIES:
        for v in vias:
            d = np.linalg.norm(xy[f:] - np.asarray(v), axis=1)
            hit = np.where(d < VIA_TOL)[0]
            if len(hit) == 0:
                raise RuntimeError(f"Ejercicio {mov}: no se encontró el punto de paso {v}")
            f += int(hit[0])
        d = np.linalg.norm(xy[f:] - np.asarray(pt), axis=1)
        hit = np.where(d < PT_TOL)[0]
        if len(hit) == 0:
            raise RuntimeError(f"Ejercicio {mov}: no se encontró el inicio {pt}")
        a = hit[0]
        b = a
        while b + 1 < len(d) and d[b + 1] < PT_TOL:
            b += 1
        f += int(a + np.argmin(d[a:b + 1]))
        starts.append((mov, f))
        f += int(0.5 * fps)
    out = []
    for k, (mov, s) in enumerate(starts):
        e = starts[k + 1][1] if k + 1 < len(starts) else F
        out.append({"mov": mov, "f0": s, "f1": e})
    return out

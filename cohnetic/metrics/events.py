"""Eventos sancionables detectables con visión (Art. 19 del reglamento).

* Acariciar al caballo por delante de las riendas (-5 puntos en doma).
* Uso de la mano derecha en las riendas (descalificación > 3 s, senior domados).
* Arrear con la voz / chasquidos (-2): requiere AUDIO; no evaluable sólo con vídeo.
"""
from __future__ import annotations

import numpy as np

from ..skeleton import hidx, ridx
from .gait import _runs


def _dist_point_segment(p, a, b):
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, axis=1) / (np.sum(ab * ab, axis=1) + 1e-9), 0, 1)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1)


def detect_caress(R: np.ndarray, H: np.ndarray, fwd: np.ndarray, fps: float, thr: float = 0.16, min_s: float = 0.4) -> list[dict]:
    rh = R[:, ridx("R_hand")]
    lh = R[:, ridx("L_hand")]
    d = _dist_point_segment(rh, H[:, hidx("withers")], H[:, hidx("poll")])
    ahead = np.sum((rh - lh) * fwd, axis=1) > 0.05
    m = np.nan_to_num(d, nan=9) < thr
    m &= ahead
    ev = []
    for a, b in _runs(m):
        if b - a >= int(min_s * fps):
            ev.append({"f0": int(a), "f1": int(b), "dur_s": round((b - a) / fps, 2), "min_dist_m": round(float(np.nanmin(d[a:b])), 2)})
    return ev


def detect_two_hands(R: np.ndarray, fps: float, thr: float = 0.10, max_s: float = 3.0) -> list[dict]:
    """Mano derecha junto a la izquierda (sobre las riendas) de forma prolongada."""
    d = np.linalg.norm(R[:, ridx("R_hand")] - R[:, ridx("L_hand")], axis=1)
    ev = []
    for a, b in _runs(np.nan_to_num(d, nan=9) < thr):
        dur = (b - a) / fps
        if dur >= 0.5:
            ev.append({"f0": int(a), "f1": int(b), "dur_s": round(dur, 2), "disqualifying": dur > max_s})
    return ev

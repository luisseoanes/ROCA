"""Postura: posición cabeza-cuello (HNP) del caballo y posición del jinete.

HNP: ángulo del plano nasal (nuca -> nariz) respecto a la vertical, en el plano
sagital del caballo. Convención de este módulo: positivo = POR DELANTE de la
vertical, negativo = DETRÁS de la vertical (BV). Los artículos de la carpeta
usan el signo opuesto (positivo = detrás); el informe lo explicita en texto.
Categorías (Kienapfel et al. 2014): por delante / hasta 10° detrás / >10° detrás.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..skeleton import hidx, ridx


def hnp_angle(H: np.ndarray, fwd: np.ndarray) -> dict:
    poll = H[:, hidx("poll")]
    nose = H[:, hidx("nose")]
    withers = H[:, hidx("withers")]
    n = nose - poll
    ang = np.degrees(np.arctan2(np.sum(n * fwd, axis=1), -n[:, 2]))
    neck = poll - withers
    neck_elev = np.degrees(np.arctan2(neck[:, 2], np.sum(neck * fwd, axis=1)))
    poll_height = poll[:, 2]
    return {"nasal_deg": ang, "neck_elev_deg": neck_elev, "poll_height_m": poll_height}


def hnp_summary(ang: np.ndarray) -> dict:
    a = ang[~np.isnan(ang)]
    if len(a) == 0:
        return {}
    return {
        "mean_deg": round(float(np.mean(a)), 1),
        "min_deg": round(float(np.min(a)), 1),
        "pct_behind": round(float(np.mean(a < 0) * 100), 1),
        "pct_behind10": round(float(np.mean(a < -10) * 100), 1),
    }


def rider_posture(R: np.ndarray, H: np.ndarray, fwd: np.ndarray, fps: float) -> dict:
    pelvis = (R[:, ridx("L_hip")] + R[:, ridx("R_hip")]) / 2
    neck = R[:, ridx("neck")]
    tr = neck - pelvis
    lean = np.degrees(np.arctan2(np.sum(tr * fwd, axis=1), tr[:, 2]))
    # rebote: movimiento vertical de la pelvis del jinete relativo al dorso del caballo
    rel_z = pelvis[:, 2] - H[:, hidx("mid_back"), 2]
    rel_z = np.nan_to_num(rel_z - np.nanmean(rel_z))
    # banda ~1-8 Hz: quita la deriva lenta y el ruido de reconstrucción de alta frecuencia
    lp = max(1, int(0.1 * fps))
    rel_z = uniform_filter1d(rel_z, lp)
    bounce = rel_z - uniform_filter1d(rel_z, int(fps))
    # estabilidad de la mano de las riendas respecto a la cruz (marco del caballo)
    hand = R[:, ridx("L_hand")] - H[:, hidx("withers")]
    hand = uniform_filter1d(np.nan_to_num(hand), lp, axis=0)
    hand_hp = hand - uniform_filter1d(hand, int(1.5 * fps), axis=0)
    hand_jit = np.linalg.norm(hand_hp, axis=1)
    return {"lean_deg": lean, "bounce_m": bounce, "hand_jitter_m": hand_jit}


def window_std(x: np.ndarray, fps: float) -> np.ndarray:
    """Desviación típica móvil (1 s)."""
    m = uniform_filter1d(x, int(fps))
    m2 = uniform_filter1d(x * x, int(fps))
    return np.sqrt(np.clip(m2 - m * m, 0, None))

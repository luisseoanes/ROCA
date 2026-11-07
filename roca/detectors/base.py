"""Interfaz común de los detectores 2D y traducción de esqueletos.

Cada detector devuelve, para una lista de frames de un vídeo:
    kp   (F, J, 2) píxeles (con distorsión, tal cual salen de la imagen), NaN si no hay
    conf (F, J)    confianza en [0, 1]
con J y el orden de `HORSE_KP` o `RIDER_KP`.
"""
from __future__ import annotations

from typing import Protocol, Union

import numpy as np

# Un punto destino se obtiene de: un nombre del detector, la media de varios, o nada (None).
Source = Union[str, dict, None]


class Detector(Protocol):
    names: list[str]

    def detect_video(self, path: str, frames: list[int], context: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        ...


def apply_mapping(src_names: list[str], kp: np.ndarray, conf: np.ndarray,
                  mapping: dict[str, Source], target_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Traduce keypoints del esqueleto del detector al de ROCA.

    kp (..., Js, 2), conf (..., Js) -> (..., Jt, 2), (..., Jt).
    mapping[destino] = "nombre_origen" | {"mean": [nombres]} | None.
    Para una media se exige que todos los puntos existan; la confianza es la mínima.
    """
    idx = {n: i for i, n in enumerate(src_names)}
    lead = kp.shape[:-2]
    out = np.full((*lead, len(target_names), 2), np.nan)
    oc = np.zeros((*lead, len(target_names)))
    for t, name in enumerate(target_names):
        src = mapping.get(name)
        if src is None:
            continue
        parts = [src] if isinstance(src, str) else list(src["mean"])
        missing = [p for p in parts if p not in idx]
        if missing:
            raise KeyError(f"El detector no tiene los puntos {missing} (para '{name}'). Disponibles: {src_names}")
        ii = [idx[p] for p in parts]
        out[..., t, :] = kp[..., ii, :].mean(axis=-2)
        oc[..., t] = conf[..., ii].min(axis=-1)
    return out, oc


def bbox(kp: np.ndarray, conf: np.ndarray, min_conf: float = 0.3) -> np.ndarray | None:
    """Caja (x0, y0, x1, y1) de los keypoints fiables de un frame (J,2) o None."""
    ok = (conf >= min_conf) & ~np.isnan(kp[:, 0])
    if ok.sum() < 2:
        return None
    p = kp[ok]
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])

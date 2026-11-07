"""Lectura de vídeo (OpenCV) y sincronización entre cámaras.

Sincronización: lo ideal es genlock / timecode por hardware. Si no hay, se
dispara un flash (o se enciende un LED) visible por todas las cámaras al
inicio del recorrido y `estimate_offsets` alinea las señales de brillo.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - depende del entorno
        raise ImportError("Falta OpenCV: pip install opencv-contrib-python") from e
    return cv2


@dataclass
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: str | Path) -> VideoInfo:
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se puede abrir el vídeo: {path}")
    try:
        return VideoInfo(
            path=str(path),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def iter_frames(path: str | Path, start: int = 0, stop: int | None = None, step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Recorre el vídeo devolviendo (índice, frame BGR). Lee secuencialmente (más fiable que saltar)."""
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se puede abrir el vídeo: {path}")
    try:
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        while stop is None or i < stop:
            ok, frame = cap.read()
            if not ok:
                break
            if (i - start) % step == 0:
                yield i, frame
            i += 1
    finally:
        cap.release()


def read_frame(path: str | Path, index: int) -> np.ndarray:
    for _, f in iter_frames(path, start=index, stop=index + 1):
        return f
    raise IndexError(f"El vídeo {path} no tiene el frame {index}")


def brightness_signal(path: str | Path, max_frames: int | None = None, scale: int = 8) -> np.ndarray:
    """Brillo medio por frame (sobre una versión reducida de la imagen)."""
    cv2 = _cv2()
    vals = []
    for _, f in iter_frames(path, stop=max_frames):
        small = f[::scale, ::scale]
        vals.append(float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean()))
    return np.asarray(vals)


def estimate_offsets(signals: list[np.ndarray], max_lag: int) -> list[int]:
    """Desfase en frames de cada cámara respecto a la primera, por correlación cruzada
    de la derivada positiva del brillo (el flash es un salto brusco).

    Convención: el frame `f` de la cámara 0 corresponde al frame `f + offset[c]` de la cámara c.
    """
    def onset(s):
        d = np.diff(np.asarray(s, float), prepend=s[0])
        d = np.clip(d, 0, None)
        sd = d.std()
        return (d - d.mean()) / sd if sd > 0 else d * 0

    ref = onset(signals[0])
    out = [0]
    for s in signals[1:]:
        x = onset(s)
        best, best_lag = -np.inf, 0
        for lag in range(-max_lag, max_lag + 1):
            a = ref[max(0, -lag):]
            b = x[max(0, lag):]
            n = min(len(a), len(b))
            if n < 2:
                continue
            v = float(np.dot(a[:n], b[:n]))
            if v > best:
                best, best_lag = v, lag
        out.append(best_lag)
    return out

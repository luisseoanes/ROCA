"""Pista de doma 40x20 m con letras (Anexo III del reglamento RFHE)."""
from __future__ import annotations

import numpy as np

LENGTH = 40.0
WIDTH = 20.0
HALF_W = WIDTH / 2

# Letras en coordenadas de pista (x, y). Vista desde A mirando a C:
# izquierda K-E-H (y=+10), derecha F-B-M (y=-10). D, X, G en la línea central.
LETTERS: dict[str, tuple[float, float]] = {
    "A": (0.0, 0.0),
    "K": (6.0, HALF_W),
    "E": (20.0, HALF_W),
    "H": (34.0, HALF_W),
    "C": (40.0, 0.0),
    "M": (34.0, -HALF_W),
    "B": (20.0, -HALF_W),
    "F": (6.0, -HALF_W),
    "D": (6.0, 0.0),
    "X": (20.0, 0.0),
    "G": (34.0, 0.0),
}

CORNERS = {
    "AK": (0.0, HALF_W),
    "AF": (0.0, -HALF_W),
    "CH": (LENGTH, HALF_W),
    "CM": (LENGTH, -HALF_W),
}


def letter(name: str) -> np.ndarray:
    return np.array(LETTERS[name], dtype=float)


def control_points_3d() -> tuple[list[str], np.ndarray]:
    """Puntos de control topografiados (letras + esquinas) en el suelo, z=0.

    En la instalación real se miden con estación total / GPS RTK y se usan
    para la calibración extrínseca de cada cámara.
    """
    names = list(LETTERS) + list(CORNERS)
    pts = [(*LETTERS[n], 0.0) for n in LETTERS] + [(*CORNERS[n], 0.0) for n in CORNERS]
    return names, np.array(pts, dtype=float)

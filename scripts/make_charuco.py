"""Genera el tablero ChArUco para imprimir (calibración intrínseca).

Uso:
    python scripts/make_charuco.py --out charuco.png
Imprimir al 100% (sin "ajustar a página"), pegar sobre una superficie rígida y plana,
y medir con regla el lado del cuadrado y del marcador: esas medidas van en
calibrate_intrinsics.py (--square-m, --marker-m).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from roca.calibration import BoardSpec, board_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="charuco.png")
    ap.add_argument("--squares-x", type=int, default=7)
    ap.add_argument("--squares-y", type=int, default=5)
    ap.add_argument("--px", type=int, default=300, help="píxeles por cuadrado")
    a = ap.parse_args()
    spec = BoardSpec(a.squares_x, a.squares_y)
    cv2.imwrite(a.out, board_image(spec, a.px))
    print(f"Tablero {a.squares_x}x{a.squares_y} ({spec.dictionary}) guardado en {a.out}")


if __name__ == "__main__":
    main()

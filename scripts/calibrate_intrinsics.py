"""Calibración intrínseca de una cámara con un vídeo del tablero ChArUco.

Filma el tablero moviéndolo por todo el encuadre (esquinas incluidas), inclinado en
distintos ángulos, con la misma óptica, zoom y resolución que en la pista.

Uso:
    python scripts/calibrate_intrinsics.py --video cam1_tablero.mp4 --name CAM1 \
        --square-m 0.08 --marker-m 0.06 --out calib/cam1_intr.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roca.calibration import BoardSpec, calibrate_intrinsics_video, save_intrinsics  # noqa: E402
from roca.video import probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--squares-x", type=int, default=7)
    ap.add_argument("--squares-y", type=int, default=5)
    ap.add_argument("--square-m", type=float, required=True, help="lado del cuadrado medido, en metros")
    ap.add_argument("--marker-m", type=float, required=True, help="lado del marcador ArUco medido, en metros")
    ap.add_argument("--max-views", type=int, default=60)
    a = ap.parse_args()
    spec = BoardSpec(a.squares_x, a.squares_y, a.square_m, a.marker_m)
    intr = calibrate_intrinsics_video(a.video, spec, a.max_views)
    save_intrinsics(a.out, a.name, intr, probe(a.video).fps)
    K = intr["K"]
    print(f"{a.name}: {intr['n_views']} vistas, RMS {intr['rms_px']:.3f} px, f=({K[0, 0]:.1f}, {K[1, 1]:.1f}) px, "
          f"centro=({K[0, 2]:.1f}, {K[1, 2]:.1f}) -> {a.out}")
    if intr["rms_px"] > 1.0:
        print("AVISO: RMS > 1 px. Repite con más vistas, mejor enfoque o el tablero más plano.")


if __name__ == "__main__":
    main()

"""Ejecuta ROCA sobre vídeo real y genera el informe web.

Uso:
    python scripts/run_video.py sesion/session.json              # informe en web/data/
    python scripts/run_video.py sesion/session.json --solo-3d    # sólo reconstrucción -> work/reconstruction.npz

Formato de session.json: ver roca/session.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402

from roca.report import write_web_data  # noqa: E402
from roca.session import load_session, reconstruct, run_session  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--web", default=str(ROOT / "web"))
    ap.add_argument("--solo-3d", action="store_true", help="no analiza ni puntúa (útil con clips cortos)")
    a = ap.parse_args()
    cfg = load_session(a.session)

    if a.solo_3d:
        rec = reconstruct(cfg)
        out = Path(cfg["work_dir"]) / "reconstruction.npz"
        np.savez_compressed(out, X=rec["X"], fps=rec["fps"], frames=rec["frames"], offsets=rec["offsets"],
                            reproj_px=rec["rec"]["reproj_px"], n_inliers=rec["rec"]["n_inliers"])
        print(f"Reconstrucción 3D en {out}")
        return

    out = run_session(cfg)
    meta = {"synthetic": False, "rider": cfg.get("rider", "Jinete"), "horse": cfg.get("horse", "Caballo"),
            "session": Path(cfg["_path"]).name}
    p1, p2 = write_web_data(out, a.web, meta)
    print(f"\nDatos web: {p1}, {p2}\nAbre {Path(a.web) / 'index.html'} en el navegador.")


if __name__ == "__main__":
    main()

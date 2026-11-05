"""Ejecuta el proof of concept completo y genera los datos de la web.

Uso:
    python scripts/run_poc.py                 # binomio de demostración (con errores)
    python scripts/run_poc.py --profile ideal # ejecución de referencia
    python scripts/run_poc.py --cams 8 --seed 3
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

from roca.pipeline import run_poc  # noqa: E402
from roca.protocol import DEMO_PROFILE, IDEAL_PROFILE  # noqa: E402
from roca.report import write_web_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="ROCA PoC: captura 3D sintética -> análisis -> puntuación -> web")
    ap.add_argument("--profile", choices=["demo", "ideal"], default="demo")
    ap.add_argument("--cams", type=int, default=6, choices=range(4, 9))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--web", default=str(ROOT / "web"))
    args = ap.parse_args()

    profile = DEMO_PROFILE if args.profile == "demo" else IDEAL_PROFILE
    out = run_poc(profile, seed=args.seed, n_cams=args.cams)
    meta = {"rider": "Binomio de demostración" if args.profile == "demo" else "Binomio de referencia (ideal)",
            "profile": args.profile, "seed": args.seed}
    p1, p2 = write_web_data(out, args.web, meta)
    print(f"\nDatos web: {p1} ({p1.stat().st_size / 1e6:.1f} MB), {p2} ({p2.stat().st_size / 1e6:.1f} MB)")
    print(f"Abre {Path(args.web) / 'index.html'} en el navegador.")


if __name__ == "__main__":
    main()

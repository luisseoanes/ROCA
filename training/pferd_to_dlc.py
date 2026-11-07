"""Convierte PFERD (keypoints 2D por cámara + vídeo) en datos etiquetados de DeepLabCut
con los 26 puntos del caballo de ROCA.

PFERD: https://github.com/Celiali/PFERD  ·  datos: https://doi.org/10.7910/DVN/2EXONE
LICENCIA: los datos son sólo para investigación no comercial y no se pueden redistribuir;
revisarlo antes de usar un modelo entrenado con PFERD en un producto comercial.

El mapeo PFERD -> ROCA está en training/pferd_mapping.json (nombres sin el prefijo IDn_).
Lectura y normalización de los datos: roca/pferd.py.

Inspeccionar (sujetos, cámaras, marcadores):
    python training/pferd_to_dlc.py inspect --dataset data/PFERD/dataset

Convertir (1 frame de cada N, una carpeta por vídeo con CollectedData_<scorer>.csv):
    python training/pferd_to_dlc.py convert --dataset data/PFERD/dataset --out data/pferd_dlc/labeled-data \
        --subjects ID_3,ID_5 --every 20
Dejar un caballo fuera (p. ej. ID_1) para medir con validate_pferd.py sobre un caballo no visto.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from roca.pferd import find_trials, load_kp2d, load_mapping, to_roca  # noqa: E402
from roca.skeleton import HORSE_KP  # noqa: E402


def cmd_inspect(a):
    trials = find_trials(a.dataset)
    if not trials:
        raise SystemExit(f"No se encontró ningún */KP2D_DATA/*/*_2Dkp.npz en {a.dataset}")
    by_subj = {}
    for t in trials:
        by_subj.setdefault(t["subject"], set()).add(t["trial"])
    for s, tr in sorted(by_subj.items()):
        vids = sum(1 for t in trials if t["subject"] == s and t["video"].exists())
        n = sum(1 for t in trials if t["subject"] == s)
        print(f"{s}: {len(tr)} pruebas, {n} pares prueba/cámara, {vids} con vídeo")
    t = trials[0]
    d = load_kp2d(t["kp2d"])
    print(f"\nEjemplo {t['kp2d'].name}: kp {d['kp'].shape}, {d['fps']:g} fps")
    print("marcadores:", d["labels"])
    roca = to_roca(d["labels"], d["kp"], load_mapping(a.mapping))
    cov = np.isfinite(roca[..., 0]).mean(0)
    print("\ncobertura por punto de ROCA en ese vídeo:")
    for n, c in zip(HORSE_KP, cov):
        print(f"  {n:12s} {c * 100:5.1f}%")


def cmd_convert(a):
    import cv2

    mapping = load_mapping(a.mapping)
    out = Path(a.out)
    trials = [t for t in find_trials(a.dataset) if t["video"].exists()]
    if a.subjects:
        trials = [t for t in trials if t["subject"] in a.subjects.split(",")]
    if a.cams:
        trials = [t for t in trials if t["serial"] in a.cams.split(",") or t["cam"] in a.cams.split(",")]
    if not trials:
        raise SystemExit("No hay pruebas con vídeo para esos filtros")
    total = 0
    for t in trials:
        d = load_kp2d(t["kp2d"])
        kp = to_roca(d["labels"], d["kp"], mapping)
        n_ok = np.isfinite(kp[..., 0]).sum(axis=1)
        good = np.where(n_ok >= a.min_points)[0][::a.every]
        if len(good) == 0:
            continue
        name = f"{t['subject']}_{t['trial']}_{t['cam']}"
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        want = set(int(x) for x in good)
        cap = cv2.VideoCapture(str(t["video"]))
        rows, f = [], 0
        last = max(want)
        while f <= last:
            ok = cap.grab()
            if not ok:
                break
            if f in want:
                ok, img = cap.retrieve()
                if ok:
                    fn = f"img{f:05d}.png"
                    cv2.imwrite(str(folder / fn), img)
                    vals = []
                    for j in range(len(HORSE_KP)):
                        x, y = kp[f, j]
                        vals += ["", ""] if np.isnan(x) else [f"{x:.2f}", f"{y:.2f}"]
                    rows.append(["labeled-data", name, fn, *vals])
            f += 1
        cap.release()
        with open(folder / f"CollectedData_{a.scorer}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["scorer", "", "", *[a.scorer] * (2 * len(HORSE_KP))])
            w.writerow(["bodyparts", "", "", *[n for n in HORSE_KP for _ in (0, 1)]])
            w.writerow(["coords", "", "", *["x", "y"] * len(HORSE_KP)])
            w.writerows(rows)
        total += len(rows)
        print(f"  {name}: {len(rows)} frames")
    print(f"\n{total} frames etiquetados en {out}. Siguiente paso: training/train_horse.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("--dataset", required=True)
    p.add_argument("--mapping", help="por defecto training/pferd_mapping.json")
    p = sub.add_parser("convert")
    p.add_argument("--dataset", required=True)
    p.add_argument("--mapping", help="por defecto training/pferd_mapping.json")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=10, help="1 de cada N frames válidos (los vecinos son casi iguales)")
    p.add_argument("--min-points", type=int, default=12)
    p.add_argument("--scorer", default="PFERD")
    p.add_argument("--subjects", help="lista separada por comas (por defecto, todos)")
    p.add_argument("--cams", help="números de serie o códigos de cámara separados por comas (por defecto, todas)")
    a = ap.parse_args()
    {"inspect": cmd_inspect, "convert": cmd_convert}[a.cmd](a)


if __name__ == "__main__":
    main()

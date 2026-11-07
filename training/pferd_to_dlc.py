"""Convierte PFERD (keypoints 2D por cámara + vídeo) en datos etiquetados de DeepLabCut
con los 26 puntos del caballo de ROCA.

PFERD: https://github.com/Celiali/PFERD  ·  datos: https://doi.org/10.7910/DVN/2EXONE
LICENCIA: el código es CC BY-NC-SA 4.0 y los datos tienen sus propios términos en Dataverse;
revisarlos antes de usar un modelo entrenado con PFERD en un producto comercial.

Estructura esperada (según el README de PFERD):
    dataset/<Subject>/KP2D_DATA/<Trial>/<Trial>_<Cam>_2Dkp.npz    claves: kp2d, labels, videoFps, videoFrameNum
    dataset/<Subject>/VIDEO_DATA/<Trial>/<Trial>_<Cam>.avi

Paso 1, inspeccionar (nombres de puntos y formas de los arrays):
    python training/pferd_to_dlc.py inspect --dataset /ruta/PFERD/dataset

Paso 2, escribir el mapeo PFERD -> ROCA en un JSON (partir de training/pferd_mapping.example.json):
    {"nose": "<label PFERD>", "eye": {"mean": ["<label>", "<label>"]}, "LF_fetlock": "<label>", ...}
    Los puntos de ROCA sin equivalente se ponen a null (DeepLabCut los tratará como no etiquetados).

Paso 3, convertir (extrae 1 frame de cada N y escribe CollectedData_<scorer>.csv por vídeo):
    python training/pferd_to_dlc.py convert --dataset /ruta/PFERD/dataset --mapping mapping.json \
        --out pferd_dlc/labeled-data --every 10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from roca.detectors.base import apply_mapping  # noqa: E402
from roca.skeleton import HORSE_KP  # noqa: E402


def find_trials(dataset: Path) -> list[dict]:
    out = []
    for npz in sorted(dataset.glob("*/KP2D_DATA/*/*_2Dkp.npz")):
        subject = npz.parts[-4]
        trial = npz.parent.name
        cam = npz.stem[len(trial) + 1:-len("_2Dkp")]
        video = dataset / subject / "VIDEO_DATA" / trial / f"{trial}_{cam}.avi"
        out.append({"subject": subject, "trial": trial, "cam": cam, "kp2d": npz, "video": video})
    return out


def load_kp2d(npz: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """-> (labels, kp (F,J,2), conf (F,J)). Acepta (F,J,2|3) o (J,F,2|3)."""
    d = np.load(npz, allow_pickle=True)
    labels = [str(x) for x in np.asarray(d["labels"]).ravel()]
    kp = np.asarray(d["kp2d"], float)
    if kp.ndim != 3 or kp.shape[-1] not in (2, 3):
        raise ValueError(f"{npz.name}: forma de kp2d inesperada {kp.shape}; ejecuta 'inspect' y adapta load_kp2d")
    if kp.shape[1] != len(labels) and kp.shape[0] == len(labels):
        kp = kp.transpose(1, 0, 2)
    if kp.shape[1] != len(labels):
        raise ValueError(f"{npz.name}: {len(labels)} labels pero kp2d tiene forma {kp.shape}")
    xy = kp[..., :2].copy()
    bad = ~np.isfinite(xy).all(axis=-1) | (xy <= 0).all(axis=-1)  # ceros = no visible en muchos datasets
    if kp.shape[-1] == 3:
        bad |= ~(kp[..., 2] > 0)
    xy[bad] = np.nan
    return labels, xy, (~bad).astype(float)


def cmd_inspect(a):
    trials = find_trials(Path(a.dataset))
    if not trials:
        raise SystemExit(f"No se encontró ningún */KP2D_DATA/*/*_2Dkp.npz en {a.dataset}")
    subjects = sorted({t["subject"] for t in trials})
    cams = sorted({t["cam"] for t in trials})
    print(f"{len(trials)} pares prueba/cámara · sujetos: {subjects} · cámaras: {cams}")
    t = trials[0]
    d = np.load(t["kp2d"], allow_pickle=True)
    print(f"\nEjemplo {t['kp2d'].name}:")
    for k in d.files:
        v = d[k]
        print(f"  {k}: shape={v.shape} dtype={v.dtype}" + (f" -> {v.ravel()[:3]}" if v.size < 10 else ""))
    print("\nlabels:", [str(x) for x in np.asarray(d["labels"]).ravel()])
    print(f"vídeo: {t['video']} ({'existe' if t['video'].exists() else 'NO ENCONTRADO'})")


def cmd_convert(a):
    import cv2

    mapping = json.loads(Path(a.mapping).read_text(encoding="utf-8"))
    unknown = sorted(set(mapping) - set(HORSE_KP))
    if unknown:
        raise SystemExit(f"Claves del mapeo que no son puntos de ROCA: {unknown}")
    out = Path(a.out)
    trials = find_trials(Path(a.dataset))
    if a.subjects:
        trials = [t for t in trials if t["subject"] in a.subjects.split(",")]
    total = 0
    for t in trials:
        if not t["video"].exists():
            print(f"  (sin vídeo) {t['video']}")
            continue
        labels, kp, conf = load_kp2d(t["kp2d"])
        kp, conf = apply_mapping(labels, kp, conf, mapping, HORSE_KP)
        good = np.where((conf > 0).sum(axis=1) >= a.min_points)[0]
        good = good[::a.every]
        if len(good) == 0:
            continue
        name = f"{t['trial']}_{t['cam']}"
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        want = set(int(x) for x in good)
        cap = cv2.VideoCapture(str(t["video"]))
        rows, f = [], 0
        while want:
            ok, img = cap.read()
            if not ok:
                break
            if f in want:
                fn = f"img{f:05d}.png"
                cv2.imwrite(str(folder / fn), img)
                vals = []
                for j in range(len(HORSE_KP)):
                    vals += ["", ""] if np.isnan(kp[f, j, 0]) else [f"{kp[f, j, 0]:.2f}", f"{kp[f, j, 1]:.2f}"]
                rows.append(["labeled-data", name, fn, *vals])
                want.discard(f)
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
    p = sub.add_parser("convert")
    p.add_argument("--dataset", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=10, help="1 de cada N frames válidos (los vecinos son casi iguales)")
    p.add_argument("--min-points", type=int, default=8)
    p.add_argument("--scorer", default="PFERD")
    p.add_argument("--subjects", help="lista separada por comas (por defecto, todos)")
    a = ap.parse_args()
    {"inspect": cmd_inspect, "convert": cmd_convert}[a.cmd](a)


if __name__ == "__main__":
    main()

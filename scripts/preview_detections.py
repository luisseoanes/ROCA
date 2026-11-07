"""Dibuja las detecciones 2D sobre el vídeo para revisarlas a ojo.

Úsalo antes de fiarte de una reconstrucción: confirma que el detector encuentra al
caballo y al jinete correctos y que cada punto cae donde debe (sobre todo con
SuperAnimal sin ajustar, cuyo mapeo a los puntos de ROCA es aproximado).

Uso:
    python scripts/preview_detections.py sesion/session.json --camera CAM1 --out cam1_preview.mp4
(usa las detecciones guardadas en work/detections; ejecuta antes run_video.py o --solo-3d)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from roca.session import _cache_key, frame_plan, load_session, resolve_offsets  # noqa: E402
from roca.skeleton import HORSE_EDGES, HORSE_KP, RIDER_EDGES, RIDER_KP, edges_idx  # noqa: E402
from roca.video import iter_frames, probe  # noqa: E402


def draw(img, kp, conf, edges, color, min_conf, names=None):
    for a, b in edges:
        if conf[a] >= min_conf and conf[b] >= min_conf:
            cv2.line(img, tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b]), color, 2)
    for j, (p, c) in enumerate(zip(kp, conf)):
        if c >= min_conf and not np.isnan(p[0]):
            cv2.circle(img, (int(p[0]), int(p[1])), 4, color, -1)
            if names:
                cv2.putText(img, names[j], (int(p[0]) + 5, int(p[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", action="store_true", help="escribe el nombre de cada punto")
    ap.add_argument("--max-frames", type=int, default=500)
    a = ap.parse_args()
    cfg = load_session(a.session)
    names = [c["name"] for c in cfg["cameras"]]
    if a.camera not in names:
        ap.error(f"Cámara desconocida {a.camera}; hay {names}")
    ci = names.index(a.camera)
    cam = cfg["cameras"][ci]
    infos = [probe(c["video"]) for c in cfg["cameras"]]
    offs = resolve_offsets(cfg, infos)
    _, per_cam = frame_plan([i.n_frames for i in infos], offs, cfg.get("start_frame", 0) or 0, cfg.get("stop_frame"), int(cfg["stride"]))
    frames = per_cam[ci]
    cache = Path(cfg["work_dir"]) / "detections"
    det = {}
    for kind in ("horse", "rider"):
        f = cache / f"{a.camera}_{kind}_{_cache_key(cfg['detectors'].get(kind, {}), frames, cam['video'])}.npz"
        if not f.exists():
            raise SystemExit(f"No hay detecciones guardadas ({f.name}). Ejecuta antes scripts/run_video.py --solo-3d")
        det[kind] = np.load(f)
    pos = {int(f): k for k, f in enumerate(frames)}
    info = infos[ci]
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), info.fps / int(cfg["stride"]), (info.width, info.height))
    n = 0
    for f, img in iter_frames(cam["video"], start=int(frames[0]), stop=int(frames[-1]) + 1):
        k = pos.get(f)
        if k is None:
            continue
        draw(img, det["horse"]["kp"][k], det["horse"]["conf"][k], edges_idx(HORSE_KP, HORSE_EDGES), (0, 200, 255), cfg["min_conf"],
             HORSE_KP if a.labels else None)
        draw(img, det["rider"]["kp"][k], det["rider"]["conf"][k], edges_idx(RIDER_KP, RIDER_EDGES), (255, 120, 0), cfg["min_conf"],
             RIDER_KP if a.labels else None)
        wr.write(img)
        n += 1
        if n >= a.max_frames:
            break
    wr.release()
    print(f"{n} frames -> {a.out}")


if __name__ == "__main__":
    main()

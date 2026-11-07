"""Valida la reconstrucción 3D de ROCA con datos reales de PFERD (cámaras, vídeo y mocap reales).

Usa el mismo código que una sesión de pista (roca/session.reconstruct): calibración de cámaras,
corrección de distorsión, detección 2D, triangulación DLT + RANSAC y suavizado. La referencia es el
mocap de marcadores de PFERD (Qualisys, precisión submilimétrica) llevado a los 26 puntos de ROCA.

Fuentes de 2D (--source):
  gt          keypoints 2D de PFERD (proyección del mocap): valida calibración + triangulación.
              Con --noise-px y --dropout simula un detector con ese error.
  detector    el detector configurado (--horse superanimal | dlc_project --config ...) sobre los
              vídeos: es el error real de extremo a extremo. Además mide el error 2D del detector
              en píxeles frente a los keypoints de PFERD.

Ejemplos:
    python scripts/validate_pferd.py --dataset data/PFERD/dataset --subject ID_1 --trial 20201128_ID_1_0007 \
        --source gt --noise-px 3 --dropout 0.1 --out data/validacion/gt_ruido3.json
    python scripts/validate_pferd.py ... --source detector --horse dlc_project --config modelos/.../config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402

from roca.detectors import NoRider  # noqa: E402
from roca.pferd import GroundTruthHorse, load_kp2d, load_mapping, to_roca, write_session  # noqa: E402
from roca.session import load_session, reconstruct  # noqa: E402
from roca.skeleton import HORSE_KP  # noqa: E402

GROUPS = {
    "Cabeza": ["nose", "eye", "poll", "throat"],
    "Tronco": ["withers", "chest", "mid_back", "croup", "tail_base"],
    "Miembros": [n for n in HORSE_KP if any(k in n for k in ("shoulder", "knee", "hip", "hock", "fetlock"))],
    "Cascos": [n for n in HORSE_KP if n.endswith("hoof")],
}


def _r(x, n=1):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def errors_3d(X: np.ndarray, G: np.ndarray, derived: list[str]) -> dict:
    """X, G (F,26,3) en metros -> errores en cm (los puntos derivados por geometría se excluyen)."""
    skip = [HORSE_KP.index(n) for n in derived]
    e = np.linalg.norm(X - G, axis=-1) * 100
    e[:, skip] = np.nan
    ok = np.isfinite(e)
    return {
        "mpjpe_cm": _r(np.nanmean(e), 2),
        "median_cm": _r(np.nanmedian(e), 2),
        "p95_cm": _r(np.nanpercentile(e, 95), 2) if ok.any() else None,
        "pct_under_5cm": _r((e[ok] < 5).mean() * 100),
        "pct_under_10cm": _r((e[ok] < 10).mean() * 100),
        "coverage_pct": _r(np.isfinite(X[..., 0]).mean() * 100),
        "by_group_cm": {g: _r(np.nanmean(e[:, [HORSE_KP.index(n) for n in names]]), 2) for g, names in GROUPS.items()},
        "by_point_cm": {n: _r(np.nanmean(e[:, j]), 2) for j, n in enumerate(HORSE_KP) if np.isfinite(e[:, j]).any()},
    }


def errors_2d(uv_det: np.ndarray, uv_gt: np.ndarray) -> dict:
    """Error del detector en píxeles (sólo puntos que el detector da y existen en PFERD)."""
    e = np.linalg.norm(uv_det - uv_gt, axis=-1)
    ok = np.isfinite(e)
    det_rate = np.isfinite(uv_det[..., 0])[np.isfinite(uv_gt[..., 0])].mean()
    return {"mean_px": _r(np.nanmean(e), 2), "median_px": _r(np.nanmedian(e), 2),
            "pck10_pct": _r((e[ok] < 10).mean() * 100), "pck20_pct": _r((e[ok] < 20).mean() * 100),
            "detected_pct": _r(det_rate * 100),
            "by_point_px": {n: _r(np.nanmean(e[..., j]), 1) for j, n in enumerate(HORSE_KP) if np.isfinite(e[..., j]).any()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(ROOT / "data" / "PFERD" / "dataset"))
    ap.add_argument("--subject", default="ID_1")
    ap.add_argument("--trial", default="20201128_ID_1_0007")
    ap.add_argument("--cams", help="números de serie separados por comas (por defecto, las 10)")
    ap.add_argument("--source", choices=["gt", "detector"], default="gt")
    ap.add_argument("--noise-px", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--horse", default="superanimal", choices=["superanimal", "dlc_project"])
    ap.add_argument("--config", help="config.yaml del proyecto DeepLabCut (con --horse dlc_project)")
    ap.add_argument("--device", help="cpu | mps | cuda (DeepLabCut)")
    ap.add_argument("--clip", nargs=2, type=int, metavar=("INICIO", "FIN"),
                    help="recorta los vídeos a esos frames de PFERD (el detector no recorre la prueba entera)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--thr-px", type=float, default=20.0, help="umbral RANSAC (PFERD es 1920x1088 con cámaras a ~15 m)")
    ap.add_argument("--session-dir", help="carpeta de trabajo (por defecto data/validacion/<subject>_<trial>)")
    ap.add_argument("--out", help="JSON con los resultados")
    ap.add_argument("--save-3d", help="npz con X (reconstrucción), G (mocap) y frames")
    a = ap.parse_args()

    horse_cfg = {"type": a.horse}
    if a.horse == "dlc_project":
        if not a.config:
            raise SystemExit("--horse dlc_project necesita --config")
        horse_cfg["config"] = str(Path(a.config).resolve())
    if a.device:
        horse_cfg["device"] = a.device
    sdir = Path(a.session_dir or ROOT / "data" / "validacion" / f"{a.subject}_{a.trial}")
    sp = write_session(a.dataset, a.subject, a.trial, sdir, cams=a.cams.split(",") if a.cams else None,
                       horse=horse_cfg, clip=tuple(a.clip) if a.clip else None, start_frame=a.start, stop_frame=a.stop, stride=a.stride, min_conf=a.min_conf, thr_px=a.thr_px)
    cfg = load_session(sp)
    kp2d = {c["video"]: c["kp2d"] for c in json.loads(sp.read_text())["cameras"]}
    frame0 = cfg["pferd"]["frame0"]

    if a.source == "gt":
        dets = {"horse": GroundTruthHorse(kp2d, a.noise_px, a.dropout, frame0=frame0), "rider": NoRider()}
        cfg["detectors"]["horse"] = {"type": "pferd_gt", "noise_px": a.noise_px, "dropout": a.dropout}  # clave de caché
    else:
        dets = None
    t0 = time.time()
    rec = reconstruct(cfg, dets)
    frames = rec["frames"]

    mapping = load_mapping()
    first = load_kp2d(next(iter(kp2d.values())))
    G = to_roca(first["labels"], first["mocap"], mapping)[frames + frame0]
    X = rec["X"][:, :len(HORSE_KP)]
    res = {
        "dataset": "PFERD", "subject": a.subject, "trial": a.trial, "source": a.source,
        "detector": cfg["detectors"]["horse"], "n_cams": len(rec["rig"]), "cams": [c.name for c in rec["rig"]],
        "n_frames": int(len(frames)), "pferd_frames": [int(frames[0] + frame0), int(frames[-1] + frame0)], "fps": rec["fps"], "noise_px": a.noise_px, "dropout": a.dropout,
        "derived_points": rec["derived"],
        "reproj_px": _r(np.nanmean(rec["rec"]["reproj_px"][:, :len(HORSE_KP)]), 2),
        "mean_views": _r(np.nanmean(np.where(rec["rec"]["raw_valid"], rec["rec"]["n_inliers"], np.nan)[:, :len(HORSE_KP)]), 2),
        "error_3d": errors_3d(X, G, rec["derived"]),
        "seconds": round(time.time() - t0, 1),
    }
    if a.source == "detector":
        # uv de la sesión está sin distorsión; se compara con PFERD también sin distorsión
        from roca.calibration import load_camera, undistort_points
        per_cam = {}
        for c, uv_det, off in zip(cfg["cameras"], rec["uv"], rec["offsets"]):
            cam, D = load_camera(c["calibration"])
            d = load_kp2d(kp2d[c["video"]])
            gt = undistort_points(to_roca(d["labels"], d["kp"], mapping)[frames + off + frame0], cam.K, D)
            per_cam[c["name"]] = errors_2d(uv_det[:, :len(HORSE_KP)], gt)
        res["error_2d"] = {"mean_px": _r(np.nanmean([v["mean_px"] for v in per_cam.values()]), 2),
                           "pck20_pct": _r(np.nanmean([v["pck20_pct"] for v in per_cam.values()])),
                           "detected_pct": _r(np.nanmean([v["detected_pct"] for v in per_cam.values()])),
                           "by_camera": per_cam}

    e = res["error_3d"]
    print(f"\nPFERD {a.subject}/{a.trial} · {res['n_cams']} cámaras · {res['n_frames']} frames · fuente: {a.source}"
          + (f" (ruido {a.noise_px} px, pérdidas {a.dropout:.0%})" if a.source == "gt" else ""))
    print(f"  Error 3D medio {e['mpjpe_cm']} cm · mediana {e['median_cm']} cm · p95 {e['p95_cm']} cm · "
          f"<5 cm {e['pct_under_5cm']}% · cobertura {e['coverage_pct']}%")
    print("  Por grupo: " + ", ".join(f"{g} {v} cm" for g, v in e["by_group_cm"].items()))
    print(f"  Reproyección {res['reproj_px']} px · vistas por punto {res['mean_views']}")
    if "error_2d" in res:
        print(f"  Detector 2D: {res['error_2d']['mean_px']} px de media · PCK@20px {res['error_2d']['pck20_pct']}% · "
              f"detectado {res['error_2d']['detected_pct']}%")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  -> {a.out}")
    if a.save_3d:
        np.savez_compressed(a.save_3d, X=X, G=G, frames=frames, fps=rec["fps"])


if __name__ == "__main__":
    main()

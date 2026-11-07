"""Pipeline con vídeo real.

    session.json -> vídeos sincronizados -> detectores 2D (caballo, jinete)
    -> corrección de distorsión -> triangulate_session (el mismo del PoC)
    -> analyze -> score -> informe web

Ejemplo de session.json (rutas relativas al propio fichero):

{
  "rider": "Nombre del jinete", "horse": "Nombre del caballo",
  "cameras": [
    {"name": "CAM1", "video": "videos/cam1.mp4", "calibration": "calib/cam1.json"},
    {"name": "CAM2", "video": "videos/cam2.mp4", "calibration": "calib/cam2.json", "offset_frames": 3}
  ],
  "sync": {"method": "offsets"},             // o {"method": "flash", "search_s": 20, "max_lag_s": 3}
  "start_frame": 0, "stop_frame": null,      // tramo de la prueba, en frames de la primera cámara
  "stride": 1,                               // 2 = analizar un frame de cada dos (más rápido)
  "min_conf": 0.4,
  "detectors": {"horse": {"type": "superanimal"}, "rider": {"type": "rtmpose", "mode": "balanced"}},
  "work_dir": "work"
}
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import detectors as DET
from .advice import opportunities
from .analysis import analyze
from .calibration import load_camera, undistort_points
from .pipeline import N_H, triangulate_session
from .scoring import score
from .skeleton import HORSE_KP, LEGS, RIDER_KP, hidx
from .video import brightness_signal, estimate_offsets, probe

ALL_KP = HORSE_KP + RIDER_KP


# ----------------------------------------------------------------------------- configuración

def load_session(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    if len(cfg.get("cameras", [])) < 2:
        raise ValueError("Hacen falta al menos 2 cámaras (recomendado 6)")
    for c in cfg["cameras"]:
        c["video"] = str((base / c["video"]).resolve())
        c["calibration"] = str((base / c["calibration"]).resolve())
    cfg["work_dir"] = str((base / cfg.get("work_dir", "work")).resolve())
    cfg.setdefault("sync", {"method": "offsets"})
    cfg.setdefault("stride", 1)
    cfg.setdefault("min_conf", 0.4)
    cfg.setdefault("detectors", {"horse": {"type": "superanimal"}, "rider": {"type": "rtmpose"}})
    cfg["_path"] = str(path)
    return cfg


def resolve_offsets(cfg: dict, infos: list, log=print) -> list[int]:
    sync = cfg["sync"]
    if sync.get("method", "offsets") == "offsets":
        return [int(c.get("offset_frames", 0)) for c in cfg["cameras"]]
    if sync["method"] == "flash":
        fps = infos[0].fps
        n = int(sync.get("search_s", 20) * fps)
        sig = [brightness_signal(c["video"], max_frames=n) for c in cfg["cameras"]]
        offs = estimate_offsets(sig, max_lag=int(sync.get("max_lag_s", 3) * fps))
        log("Sincronización por flash: desfases " + ", ".join(f"{c['name']} {o:+d}" for c, o in zip(cfg["cameras"], offs)))
        return offs
    raise ValueError(f"Método de sincronización desconocido: {sync['method']}")


def frame_plan(n_frames: list[int], offsets: list[int], start: int = 0, stop: int | None = None, stride: int = 1) -> tuple[np.ndarray, list[np.ndarray]]:
    """Frames de la línea de tiempo común (índices de la cámara 0) y su frame en cada cámara."""
    lo = max([start] + [-o for o in offsets])
    hi = min(n - o for n, o in zip(n_frames, offsets))
    if stop is not None:
        hi = min(hi, stop)
    if hi <= lo:
        raise ValueError(f"Los vídeos no se solapan con estos desfases (rango {lo}..{hi})")
    base = np.arange(lo, hi, stride)
    return base, [base + o for o in offsets]


# ----------------------------------------------------------------------------- detecciones

def _cache_key(det_cfg: dict, frames: np.ndarray, video: str) -> str:
    st = Path(video).stat()
    raw = json.dumps([det_cfg, int(frames[0]), int(frames[-1]), len(frames), st.st_size, int(st.st_mtime)], sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def detect_cached(det, det_cfg: dict, video: str, frames: np.ndarray, cache: Path, context: dict | None = None,
                  log=print, label: str = "") -> tuple[np.ndarray, np.ndarray]:
    key = _cache_key(det_cfg, frames, video)
    f = cache / f"{label}_{key}.npz"
    if f.exists():
        d = np.load(f)
        return d["kp"], d["conf"]
    t0 = time.time()
    kp, conf = det.detect_video(video, [int(x) for x in frames], context)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, kp=kp, conf=conf, frames=frames)
    log(f"  {label}: {len(frames)} frames en {time.time() - t0:.0f} s")
    return kp, conf


def assemble_uv(horse: list[tuple[np.ndarray, np.ndarray]], rider: list[tuple[np.ndarray, np.ndarray]],
                cams: list, dists: list, min_conf: float) -> np.ndarray:
    """-> uv (C, F, 40, 2) sin distorsión, NaN donde la confianza es baja."""
    out = []
    for (hk, hc), (rk, rc), cam, D in zip(horse, rider, cams, dists):
        kp = np.concatenate([hk, rk], axis=1)
        conf = np.concatenate([hc, rc], axis=1)
        kp = np.where((conf >= min_conf)[..., None], kp, np.nan)
        out.append(undistort_points(kp, cam.K, D))
    return np.stack(out)


def derive_missing(X: np.ndarray) -> list[str]:
    """Completa en 3D los puntos que el detector no da (p. ej. menudillos con SuperAnimal).

    Menudillo ≈ 30% del camino del casco a la rodilla/corvejón. Devuelve los nombres derivados.
    """
    derived = []
    for leg in LEGS:
        f = hidx(f"{leg}_fetlock")
        if np.isnan(X[:, f]).all():
            upper = hidx(f"{leg}_knee" if leg[1] == "F" else f"{leg}_hock")
            X[:, f] = X[:, hidx(f"{leg}_hoof")] + 0.3 * (X[:, upper] - X[:, hidx(f"{leg}_hoof")])
            derived.append(f"{leg}_fetlock")
    return derived


# ----------------------------------------------------------------------------- ejecución

def reconstruct(cfg: dict, detectors: dict | None = None, log=print) -> dict:
    """Vídeos -> 3D. `detectors` permite inyectar detectores (tests); si no, se crean desde cfg."""
    cams_cfg = cfg["cameras"]
    infos = [probe(c["video"]) for c in cams_cfg]
    fps0 = infos[0].fps
    for c, inf in zip(cams_cfg, infos):
        if abs(inf.fps - fps0) > 0.005 * fps0:
            raise ValueError(f"{c['name']} graba a {inf.fps:.2f} fps y {cams_cfg[0]['name']} a {fps0:.2f}: "
                             "todas las cámaras deben ir a la misma frecuencia")
    loaded = [load_camera(c["calibration"]) for c in cams_cfg]
    rig = [cam for cam, _ in loaded]
    dists = [D for _, D in loaded]
    for c, cam, inf in zip(cams_cfg, rig, infos):
        cam.name = c["name"]
        if (cam.width, cam.height) != (inf.width, inf.height):
            raise ValueError(f"{c['name']}: el vídeo es {inf.width}x{inf.height} y la calibración {cam.width}x{cam.height}")

    offsets = resolve_offsets(cfg, infos, log)
    base, per_cam = frame_plan([i.n_frames for i in infos], offsets, cfg.get("start_frame", 0) or 0,
                               cfg.get("stop_frame"), int(cfg["stride"]))
    fps = fps0 / int(cfg["stride"])
    log(f"[1/4] {len(rig)} cámaras, {len(base)} frames a {fps:g} fps ({len(base) / fps:.0f} s)")

    work = Path(cfg["work_dir"])
    dets = detectors or {k: DET.build(k, cfg["detectors"][k], work) for k in ("horse", "rider")}
    cache = work / "detections"
    horse, rider = [], []
    log("[2/4] Detección 2D")
    for c, frames in zip(cams_cfg, per_cam):
        hk, hc = detect_cached(dets["horse"], cfg["detectors"].get("horse", {}), c["video"], frames, cache,
                               log=log, label=f"{c['name']}_horse")
        rk, rc = detect_cached(dets["rider"], cfg["detectors"].get("rider", {}), c["video"], frames, cache,
                               context={"horse_kp": hk, "horse_conf": hc}, log=log, label=f"{c['name']}_rider")
        horse.append((hk, hc))
        rider.append((rk, rc))

    uv = assemble_uv(horse, rider, rig, dists, cfg["min_conf"])
    t0 = time.time()
    rec = triangulate_session(rig, uv, fps)
    X = rec["X"]
    derived = derive_missing(X)
    log(f"[3/4] Triangulación en {time.time() - t0:.0f} s: reproyección media {np.nanmean(rec['reproj_px']):.2f} px, "
        f"cobertura {rec['raw_valid'].mean() * 100:.0f}%" + (f"; derivados: {', '.join(derived)}" if derived else ""))
    return {"X": X, "rec": rec, "uv": uv, "rig": rig, "fps": fps, "frames": base, "offsets": offsets,
            "derived": derived, "quality": quality(rec, uv, rig, derived)}


def quality(rec: dict, uv: np.ndarray, rig: list, derived: list[str]) -> dict:
    """Indicadores de calidad sin verdad de referencia (vídeo real)."""
    groups = {
        "Cabeza": ["nose", "eye", "poll", "throat"],
        "Tronco": ["withers", "chest", "mid_back", "croup", "tail_base"],
        "Miembros": [n for n in HORSE_KP if any(k in n for k in ("shoulder", "knee", "hip", "hock", "fetlock"))],
        "Cascos": [n for n in HORSE_KP if n.endswith("hoof")],
        "Jinete": RIDER_KP,
    }
    rp = rec["reproj_px"]
    cov = rec["raw_valid"]
    det_rate = (~np.isnan(uv[..., 0])).mean(axis=(1, 2))
    with np.errstate(invalid="ignore"):
        return {
            "real": True,
            "raw_coverage_pct": round(float(cov.mean()) * 100, 1),
            "reproj_px": round(float(np.nanmean(rp)), 2),
            "p95_reproj_px": round(float(np.nanpercentile(rp, 95)), 2) if np.isfinite(rp).any() else None,
            "mean_views": round(float(rec["n_inliers"][cov].mean()), 2) if cov.any() else 0.0,
            "group_reproj_px": {g: round(float(np.nanmean(rp[:, [ALL_KP.index(n) for n in names]])), 2) for g, names in groups.items()},
            "group_coverage_pct": {g: round(float(cov[:, [ALL_KP.index(n) for n in names]].mean()) * 100, 1) for g, names in groups.items()},
            "detection_rate_pct": {cam.name: round(float(r) * 100, 1) for cam, r in zip(rig, det_rate)},
            "derived_points": derived,
        }


def run_session(cfg: dict, detectors: dict | None = None, log=print) -> dict:
    rec = reconstruct(cfg, detectors, log)
    X, fps = rec["X"], rec["fps"]
    an = analyze(X[:, :N_H], X[:, N_H:], fps)
    sc = score(an)
    opp = opportunities(sc)
    log(f"[4/4] Puntuación: {sc['total']}/{sc['max']} = {sc['percent']}%  ({len(opp)} áreas de mejora)")
    calib = [{"name": c.name, "calib_points": c.meta.get("calib_points"), "calib_rmse_px": c.meta.get("calib_rmse_px"),
              "pos_err_cm": None} for c in rec["rig"]]
    return {"X": X, "rec": rec["rec"], "rig": rec["rig"], "calib": calib, "analysis": an, "score": sc,
            "opportunities": opp, "validation": rec["quality"], "fps": fps}

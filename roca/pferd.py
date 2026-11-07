"""Acceso a PFERD (Poses for Equine Research Dataset) para entrenar y validar ROCA con datos reales.

PFERD: https://github.com/Celiali/PFERD · datos: https://doi.org/10.7910/DVN/2EXONE
Licencia de los datos: sólo investigación no comercial, sin redistribución (ver docs/ARQUITECTURA.md).
Los datos se guardan en data/ (ignorado por git).

Estructura:
    dataset/<Subject>/CAM_DATA/Camera_Miqus_Video_<id>.npz          R (3,3), T (1,3) m, K (3,3), D (5,) OpenCV
    dataset/<Subject>/KP2D_DATA/<Trial>/<Trial>_<Cam>_2Dkp.npz      kp2d (F,132,2), mocap3d (F,132,3) m, labels,
                                                                    videoFps, videoFrameNum
    dataset/<Subject>/VIDEO_DATA/<Trial>/<Trial>_<Cam>.avi          1920x1088

Comprobado: kp2d == cv2.projectPoints(mocap3d, Rodrigues(R), T, K, D) (error 0 px), es decir
x_cam = R X + T, la misma convención que roca.camera. kp2d es la proyección del mocap, por lo que
incluye marcadores tapados por el propio cuerpo. Mundo en metros, z hacia arriba.

Los archivos .npz se abren siempre con allow_pickle=False (son datos descargados).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .camera import Camera
from .detectors.base import apply_mapping
from .skeleton import HORSE_KP

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = ROOT / "training" / "pferd_mapping.json"
WIDTH, HEIGHT = 1920, 1088


def normalize_label(s: str) -> str:
    """'ID1_L_HockJoint_30' -> 'L_HockJoint_30' (los nombres llevan el prefijo del caballo)."""
    return re.sub(r"^ID\d+_", "", str(s))


def load_mapping(path: str | Path | None = None) -> dict:
    m = json.loads(Path(path or DEFAULT_MAPPING).read_text(encoding="utf-8"))
    m = {k: v for k, v in m.items() if not k.startswith("_")}
    unknown = sorted(set(m) - set(HORSE_KP))
    if unknown:
        raise ValueError(f"Claves del mapeo que no son puntos de ROCA: {unknown}")
    return m


def cam_code(npz_or_video: Path, trial: str) -> str:
    """'<Trial>_Miqus_46_23350_2Dkp.npz' -> 'Miqus_46_23350'."""
    stem = Path(npz_or_video).stem
    stem = stem[len(trial) + 1:]
    return stem[:-len("_2Dkp")] if stem.endswith("_2Dkp") else stem


def find_trials(dataset: str | Path) -> list[dict]:
    """Un registro por (sujeto, prueba, cámara)."""
    dataset = Path(dataset)
    out = []
    for npz in sorted(dataset.glob("**/KP2D_DATA/*/*_2Dkp.npz")):  # dataset/ID_n/... o dataset/DEMO/ID_1/...
        subject_dir = npz.parents[2]
        trial = npz.parent.name
        cam = cam_code(npz, trial)
        out.append({"subject": subject_dir.name, "subject_dir": subject_dir, "trial": trial, "cam": cam,
                    "serial": cam.split("_")[-1], "kp2d": npz,
                    "video": subject_dir / "VIDEO_DATA" / trial / f"{trial}_{cam}.avi",
                    "calib": subject_dir / "CAM_DATA" / f"Camera_Miqus_Video_{cam.split('_')[-1]}.npz"})
    return out


def load_kp2d(npz: str | Path, width: int = WIDTH, height: int = HEIGHT) -> dict:
    """-> labels (normalizados), kp (F,J,2) con NaN si falta o cae fuera de la imagen, mocap (F,J,3), fps."""
    d = np.load(npz, allow_pickle=False)
    labels = [normalize_label(x) for x in np.asarray(d["labels"]).ravel()]
    kp = np.asarray(d["kp2d"], float).copy()
    if kp.ndim != 3 or kp.shape[1] != len(labels) or kp.shape[2] != 2:
        raise ValueError(f"{Path(npz).name}: forma de kp2d inesperada {kp.shape} para {len(labels)} labels")
    out = ~np.isfinite(kp).all(-1) | (kp[..., 0] < 0) | (kp[..., 0] >= width) | (kp[..., 1] < 0) | (kp[..., 1] >= height)
    kp[out] = np.nan
    mocap = np.asarray(d["mocap3d"], float) if "mocap3d" in d.files else None
    return {"labels": labels, "kp": kp, "mocap": mocap, "fps": float(d["videoFps"]), "n_frames": int(d["videoFrameNum"])}


def to_roca(labels: list[str], pts: np.ndarray, mapping: dict | None = None) -> np.ndarray:
    """Marcadores PFERD (..., 132, D) -> puntos ROCA (..., 26, D); NaN donde falta algún marcador."""
    mapping = mapping or load_mapping()
    conf = np.isfinite(pts).all(-1).astype(float)
    out, c = apply_mapping(labels, pts, conf, mapping, HORSE_KP)
    out[c <= 0] = np.nan
    return out


def load_camera_npz(path: str | Path, name: str, width: int = WIDTH, height: int = HEIGHT,
                    fps: float = 30.0) -> tuple[Camera, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    cam = Camera(name, np.asarray(d["K"], float), np.asarray(d["R"], float), np.asarray(d["T"], float).ravel(),
                 width, height, fps, {"source": "PFERD"})
    return cam, np.asarray(d["D"], float).ravel()


def trim_video(src: str | Path, dst: str | Path, start: int, stop: int) -> Path:
    """Copia los frames [start, stop) de un vídeo (para no pasar el detector por la prueba entera)."""
    import cv2

    dst = Path(dst)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    tmp = dst.with_suffix(".tmp" + dst.suffix)
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for _ in range(stop - start):
        ok, img = cap.read()
        if not ok:
            break
        vw.write(img)
    vw.release()
    cap.release()
    tmp.rename(dst)
    return dst


def write_session(dataset: str | Path, subject: str, trial: str, out_dir: str | Path,
                  cams: list[str] | None = None, horse: dict | None = None,
                  clip: tuple[int, int] | None = None, **extra) -> Path:
    """Crea out_dir/session.json + calib/*.json para una prueba de PFERD (sin jinete).

    cams: códigos ('Miqus_46_23350') o números de serie ('23350'); por defecto, todas.
    clip: (inicio, fin) en frames; recorta los vídeos a out_dir/videos/ y guarda el desfase en
    cfg["pferd"]["frame0"] (el frame k de la sesión es el frame0 + k de PFERD).
    """
    from .calibration import save_camera

    out_dir = Path(out_dir).resolve()
    rows = [t for t in find_trials(Path(dataset).resolve()) if t["subject"] == subject and t["trial"] == trial]
    if not rows:
        raise FileNotFoundError(f"No hay KP2D para {subject}/{trial} en {dataset}")
    if cams:
        rows = [t for t in rows if t["cam"] in cams or t["serial"] in cams]
    entries = []
    for t in rows:
        meta = np.load(t["kp2d"], allow_pickle=False)
        cam, D = load_camera_npz(t["calib"], t["cam"], fps=float(meta["videoFps"]))
        cpath = out_dir / "calib" / f"{t['cam']}.json"
        save_camera(cpath, cam, D)
        video = t["video"]
        if clip:
            video = trim_video(video, out_dir / "videos" / f"{t['cam']}_{clip[0]}_{clip[1]}.avi", *clip)
        entries.append({"name": t["cam"], "video": str(video), "calibration": str(cpath),
                        "kp2d": str(t["kp2d"])})
    cfg = {"rider": "(sin jinete)", "horse": f"PFERD {subject}",
           "pferd": {"subject": subject, "trial": trial, "frame0": clip[0] if clip else 0},
           "cameras": entries, "sync": {"method": "offsets"}, "min_conf": 0.4,
           "detectors": {"horse": horse or {"type": "superanimal"}, "rider": {"type": "none"}},
           "work_dir": "work", **extra}
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "session.json"
    p.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    return p


class GroundTruthHorse:
    """Detector 'perfecto' con los keypoints 2D de PFERD (proyección del mocap), con ruido opcional.

    Sirve para validar calibración + triangulación con cámaras reales independientemente del detector.
    noise_px: ruido gaussiano; dropout: fracción de puntos perdidos; los puntos tapados no se quitan.
    """
    names = HORSE_KP

    def __init__(self, kp2d_by_video: dict[str, str], noise_px: float = 0.0, dropout: float = 0.0,
                 mapping: dict | None = None, seed: int = 0, frame0: int = 0):
        self.kp2d_by_video = {str(Path(k).resolve()): v for k, v in kp2d_by_video.items()}
        self.noise_px, self.dropout, self.mapping = noise_px, dropout, mapping or load_mapping()
        self.rng = np.random.default_rng(seed)
        self.seed, self.frame0 = seed, frame0

    def cache_token(self) -> str:
        return json.dumps([self.mapping, self.noise_px, self.dropout, self.seed, self.frame0], sort_keys=True)

    def detect_video(self, path, frames, context=None):
        d = load_kp2d(self.kp2d_by_video[str(Path(path).resolve())])
        kp = to_roca(d["labels"], d["kp"], self.mapping)[np.asarray(frames) + self.frame0]
        if self.noise_px:
            kp = kp + self.rng.normal(0, self.noise_px, kp.shape)
        conf = np.isfinite(kp[..., 0]).astype(float)
        if self.dropout:
            conf[self.rng.random(conf.shape) < self.dropout] = 0
        kp[conf == 0] = np.nan
        return kp, conf

"""Calibración de cámaras reales.

1. Intrínseca (una vez por cámara + óptica): tablero ChArUco filmado desde varios
   ángulos -> K (focal, centro óptico) y D (distorsión), con OpenCV.
2. Extrínseca (cada vez que se monta la cámara): píxeles de las letras y esquinas de
   la pista -> R, t con `camera.calibrate_extrinsics_planar` (el mismo código del PoC),
   previa corrección de la distorsión.

Formato de cámara en disco (JSON):
    {"name", "width", "height", "fps", "K": 3x3, "D": [k1,k2,p1,p2,k3],
     "R": 3x3, "t": [3], "calib_points", "calib_rmse_px", "intrinsic_rms_px"}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import arena
from .camera import Camera, calibrate_extrinsics_planar
from .video import _cv2, iter_frames, probe


# ----------------------------------------------------------------------------- ChArUco

@dataclass(frozen=True)
class BoardSpec:
    """Tablero ChArUco. Medidas en metros: mídelas en el tablero impreso."""
    squares_x: int = 7
    squares_y: int = 5
    square_m: float = 0.08
    marker_m: float = 0.06
    dictionary: str = "DICT_5X5_100"

    def build(self):
        cv2 = _cv2()
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_m, self.marker_m, d)


def board_image(spec: BoardSpec, px_per_square: int = 200, margin_px: int = 40) -> np.ndarray:
    """Imagen del tablero para imprimir (imprimir al 100%, sin ajustar a la página)."""
    board = spec.build()
    size = (spec.squares_x * px_per_square + 2 * margin_px, spec.squares_y * px_per_square + 2 * margin_px)
    return board.generateImage(size, marginSize=margin_px)


def detect_charuco(image: np.ndarray, spec: BoardSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (puntos 2D (N,2), puntos 3D del tablero (N,3), ids (N,)). N puede ser 0."""
    cv2 = _cv2()
    board = spec.build()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) == 0:
        return np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0, int)
    ids = ids.ravel()
    obj = board.getChessboardCorners()[ids]
    return corners.reshape(-1, 2).astype(float), np.asarray(obj, float), ids


def calibrate_intrinsics(images: list[np.ndarray], spec: BoardSpec, min_corners: int = 8) -> dict:
    """Calibración intrínseca a partir de imágenes del tablero.

    Devuelve {"K", "D", "rms_px", "width", "height", "n_views"}.
    """
    cv2 = _cv2()
    if not images:
        raise ValueError("No hay imágenes para calibrar")
    h, w = images[0].shape[:2]
    obj_pts, img_pts = [], []
    for im in images:
        uv, xyz, _ = detect_charuco(im, spec)
        if len(uv) >= min_corners:
            obj_pts.append(xyz.astype(np.float32))
            img_pts.append(uv.astype(np.float32).reshape(-1, 1, 2))
    if len(obj_pts) < 4:
        raise ValueError(f"Sólo {len(obj_pts)} vistas válidas del tablero (mínimo 4). "
                         "Filma el tablero más cerca, más ángulos y con buena luz.")
    rms, K, D, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None)
    return {"K": K, "D": D.ravel()[:5], "rms_px": float(rms), "width": w, "height": h, "n_views": len(obj_pts)}


def calibrate_intrinsics_video(path: str | Path, spec: BoardSpec, max_views: int = 60) -> dict:
    """Calibración intrínseca desde un vídeo del tablero: toma `max_views` frames repartidos."""
    info = probe(path)
    step = max(1, info.n_frames // (max_views * 3))
    candidates = []
    for _, f in iter_frames(path, step=step):
        uv, _, _ = detect_charuco(f, spec)
        if len(uv) >= 8:
            candidates.append(f)
    if len(candidates) > max_views:
        idx = np.linspace(0, len(candidates) - 1, max_views).round().astype(int)
        candidates = [candidates[i] for i in idx]
    return calibrate_intrinsics(candidates, spec)


# ----------------------------------------------------------------------------- distorsión

def undistort_points(uv: np.ndarray, K: np.ndarray, D: np.ndarray | None) -> np.ndarray:
    """Píxeles con distorsión -> píxeles de una cámara pinhole ideal con la misma K.

    Acepta cualquier forma (..., 2) y conserva los NaN.
    """
    if D is None or not np.any(D):
        return np.asarray(uv, float).copy()
    cv2 = _cv2()
    uv = np.asarray(uv, float)
    flat = uv.reshape(-1, 2)
    ok = ~np.isnan(flat).any(axis=1)
    out = np.full_like(flat, np.nan)
    if ok.any():
        und = cv2.undistortPoints(flat[ok].reshape(-1, 1, 2).astype(np.float64), K, np.asarray(D, float), P=K)
        out[ok] = und.reshape(-1, 2)
    return out.reshape(uv.shape)


# ----------------------------------------------------------------------------- extrínseca

def calibrate_extrinsics(K: np.ndarray, D: np.ndarray | None, pixel_points: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray, float, int]:
    """`pixel_points`: {"A": [u, v], "K": [u, v], ..., "AK": [u, v]} (letras y esquinas visibles).

    Devuelve (R, t, rmse_px, n_puntos).
    """
    names, W = arena.control_points_3d()
    unknown = sorted(set(pixel_points) - set(names))
    if unknown:
        raise ValueError(f"Puntos desconocidos: {unknown}. Válidos: {names}")
    used = [n for n in names if n in pixel_points]
    if len(used) < 4:
        raise ValueError(f"Hacen falta al menos 4 puntos de la pista (hay {len(used)})")
    world = np.array([W[names.index(n)] for n in used])
    img = undistort_points(np.array([pixel_points[n] for n in used], float), K, D)
    R, t, rmse = calibrate_extrinsics_planar(K, world, img)
    return R, t, rmse, len(used)


# ----------------------------------------------------------------------------- E/S

def save_camera(path: str | Path, cam: Camera, D: np.ndarray | None = None) -> None:
    d = cam.to_dict()
    d.pop("center", None)
    d["fps"] = cam.fps
    d["D"] = [] if D is None else np.asarray(D, float).ravel().tolist()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=1), encoding="utf-8")


def load_camera(path: str | Path) -> tuple[Camera, np.ndarray | None]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "R" not in d or "t" not in d:
        raise ValueError(f"{path}: falta la calibración extrínseca (R, t). Ejecuta scripts/calibrate_extrinsics.py")
    meta = {k: d[k] for k in ("calib_points", "calib_rmse_px", "pos_err_cm", "intrinsic_rms_px") if k in d}
    cam = Camera(d["name"], np.array(d["K"], float), np.array(d["R"], float), np.array(d["t"], float),
                 int(d["width"]), int(d["height"]), float(d.get("fps", 50.0)), meta)
    D = np.array(d["D"], float) if d.get("D") else None
    return cam, D


def save_intrinsics(path: str | Path, name: str, intr: dict, fps: float) -> None:
    d = {"name": name, "width": intr["width"], "height": intr["height"], "fps": fps,
         "K": np.asarray(intr["K"]).tolist(), "D": np.asarray(intr["D"]).ravel().tolist(),
         "intrinsic_rms_px": round(intr["rms_px"], 3), "n_views": intr["n_views"]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=1), encoding="utf-8")

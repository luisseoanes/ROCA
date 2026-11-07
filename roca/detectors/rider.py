"""Detector del jinete: RTMPose (COCO-17) vía rtmlib.

rtmlib descarga los modelos ONNX la primera vez. Backends:
  - "onnxruntime" (por defecto): requiere onnxruntime (Python 3.10-3.12, ver requirements-video.txt).
  - "opencv": cv2.dnn. Con OpenCV 5.0 falla en el detector YOLOX (error en la capa 'gather');
    sólo sirve con OpenCV 4.x.
"""
from __future__ import annotations

import numpy as np

from ..skeleton import RIDER_KP, hidx
from ..video import iter_frames
from .base import apply_mapping

COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO usa izquierda/derecha del sujeto, igual que ROCA.
COCO_TO_ROCA = {
    "head": {"mean": ["left_ear", "right_ear"]},
    "neck": {"mean": ["left_shoulder", "right_shoulder"]},
    "L_shoulder": "left_shoulder", "R_shoulder": "right_shoulder",
    "L_elbow": "left_elbow", "R_elbow": "right_elbow",
    "L_hand": "left_wrist", "R_hand": "right_wrist",
    "L_hip": "left_hip", "R_hip": "right_hip",
    "L_knee": "left_knee", "R_knee": "right_knee",
    "L_ankle": "left_ankle", "R_ankle": "right_ankle",
}


def select_rider(people_kp: np.ndarray, people_conf: np.ndarray, horse_kp: np.ndarray | None,
                 horse_conf: np.ndarray | None, min_conf: float = 0.3) -> int | None:
    """Elige a la persona montada entre todas las detectadas en el frame.

    Con caballo: la persona con las caderas más cerca de la cruz (y por encima de ella en la imagen).
    Sin caballo: la de mayor confianza media por tamaño (el jinete suele ser la persona más grande).
    """
    n = len(people_kp)
    if n == 0:
        return None
    hips = people_kp[:, [11, 12]].mean(axis=1)
    hip_ok = (people_conf[:, [11, 12]] >= min_conf).all(axis=1)
    if horse_kp is not None and horse_conf is not None:
        w = hidx("withers")
        if horse_conf[w] >= min_conf and not np.isnan(horse_kp[w, 0]):
            wx, wy = horse_kp[w]
            d = np.linalg.norm(hips - [wx, wy], axis=1)
            d[~hip_ok] = np.inf
            d[hips[:, 1] > wy + 0.5 * _horse_height(horse_kp, horse_conf, min_conf)] = np.inf  # caderas bajo el caballo
            if np.isfinite(d).any():
                return int(np.argmin(d))
    span = np.nanmax(people_kp[..., 1], axis=1) - np.nanmin(people_kp[..., 1], axis=1)
    return int(np.argmax(people_conf.mean(axis=1) * np.nan_to_num(span)))


def _horse_height(kp, conf, min_conf):
    ok = conf >= min_conf
    if ok.sum() < 2:
        return np.inf
    return float(np.nanmax(kp[ok, 1]) - np.nanmin(kp[ok, 1]))


class RTMPoseRider:
    names = RIDER_KP

    def __init__(self, mode: str = "balanced", backend: str = "onnxruntime", device: str = "cpu", min_conf: float = 0.3):
        try:
            from rtmlib import Body
        except ImportError as e:  # pragma: no cover
            raise ImportError("Falta rtmlib: pip install rtmlib tqdm") from e
        self.model = Body(mode=mode, backend=backend, device=device)
        self.min_conf = min_conf

    def detect_frame(self, frame: np.ndarray, horse_kp=None, horse_conf=None) -> tuple[np.ndarray, np.ndarray]:
        kps, scores = self.model(frame)
        kps, scores = np.asarray(kps, float), np.asarray(scores, float)
        if kps.ndim != 3 or len(kps) == 0:
            return np.full((len(RIDER_KP), 2), np.nan), np.zeros(len(RIDER_KP))
        i = select_rider(kps, scores, horse_kp, horse_conf, self.min_conf)
        if i is None:
            return np.full((len(RIDER_KP), 2), np.nan), np.zeros(len(RIDER_KP))
        return apply_mapping(COCO17, kps[i], scores[i], COCO_TO_ROCA, RIDER_KP)

    def detect_video(self, path: str, frames: list[int], context: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        """context opcional: {"horse_kp": (F,26,2), "horse_conf": (F,26)} alineado con `frames`."""
        want = {f: k for k, f in enumerate(frames)}
        kp = np.full((len(frames), len(RIDER_KP), 2), np.nan)
        conf = np.zeros((len(frames), len(RIDER_KP)))
        hk = (context or {}).get("horse_kp")
        hc = (context or {}).get("horse_conf")
        if not frames:
            return kp, conf
        for f, img in iter_frames(path, start=min(frames), stop=max(frames) + 1):
            k = want.get(f)
            if k is None:
                continue
            kp[k], conf[k] = self.detect_frame(img, None if hk is None else hk[k], None if hc is None else hc[k])
        return kp, conf

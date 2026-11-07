"""Detector del caballo con DeepLabCut 3 (PyTorch).

Dos modos:
  - SuperAnimalHorse: modelo preentrenado SuperAnimal-Quadruped, sin entrenar nada
    (zero-shot). Sirve para empezar; no tiene menudillos y el pecho es aproximado.
  - DLCProjectHorse: modelo propio (proyecto DeepLabCut con los 26 puntos de ROCA),
    ajustado con PFERD y frames propios (ver training/). Es el objetivo para producción.

DeepLabCut necesita Python 3.10-3.12 (sus dependencias no publican paquetes para versiones
más nuevas). Se importa sólo al usar estos detectores.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..skeleton import HORSE_KP
from .base import apply_mapping

# Nombres de SuperAnimal-Quadruped (39 puntos; "thai" = muslo, así viene en el modelo).
SA_QUADRUPED = [
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
    "right_eye", "right_earbase", "right_earend", "right_antler_base", "right_antler_end",
    "left_eye", "left_earbase", "left_earend", "left_antler_base", "left_antler_end",
    "neck_base", "neck_end", "throat_base", "throat_end",
    "back_base", "back_end", "back_middle", "tail_base", "tail_end",
    "front_left_thai", "front_left_knee", "front_left_paw",
    "front_right_thai", "front_right_knee", "front_right_paw",
    "back_left_paw", "back_left_thai", "back_right_thai", "back_left_knee", "back_right_knee", "back_right_paw",
    "belly_bottom", "body_middle_right", "body_middle_left",
]

# Aproximación SuperAnimal -> ROCA. Revisar con scripts/preview_detections.py sobre vídeo propio.
SA_TO_ROCA = {
    "nose": "nose",
    "eye": {"mean": ["left_eye", "right_eye"]},
    "poll": {"mean": ["left_earbase", "right_earbase"]},
    "throat": "throat_base",
    "withers": "neck_base",
    "chest": "throat_end",
    "mid_back": "back_middle",
    "croup": "back_end",
    "tail_base": "tail_base",
    "tail_tip": "tail_end",
    "LF_shoulder": "front_left_thai", "LF_knee": "front_left_knee", "LF_fetlock": None, "LF_hoof": "front_left_paw",
    "RF_shoulder": "front_right_thai", "RF_knee": "front_right_knee", "RF_fetlock": None, "RF_hoof": "front_right_paw",
    "LH_hip": "back_left_thai", "LH_hock": "back_left_knee", "LH_fetlock": None, "LH_hoof": "back_left_paw",
    "RH_hip": "back_right_thai", "RH_hock": "back_right_knee", "RH_fetlock": None, "RH_hoof": "back_right_paw",
}


def _deeplabcut():
    try:
        import deeplabcut
    except ImportError as e:  # pragma: no cover
        raise ImportError("Falta DeepLabCut (necesita Python 3.10-3.12): pip install 'deeplabcut[modelzoo]'") from e
    return deeplabcut


def read_dlc_h5(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Lee la salida .h5 de DeepLabCut -> (bodyparts, kp (F,J,2), conf (F,J)).

    Si hay varios individuos, en cada frame se queda con el de mayor confianza media.
    """
    import pandas as pd

    df = pd.read_hdf(path)
    cols = df.columns
    levels = list(cols.names)
    bp_level = levels.index("bodyparts")
    bodyparts = list(dict.fromkeys(cols.get_level_values(bp_level)))
    if "individuals" in levels:
        inds = list(dict.fromkeys(cols.get_level_values("individuals")))
    else:
        inds = [None]
    F, J = len(df), len(bodyparts)
    all_kp = np.full((len(inds), F, J, 2), np.nan)
    all_c = np.zeros((len(inds), F, J))
    for a, ind in enumerate(inds):
        sub = df if ind is None else df.xs(ind, level="individuals", axis=1)
        for j, bp in enumerate(bodyparts):
            s = sub.xs(bp, level="bodyparts", axis=1)
            s.columns = s.columns.get_level_values("coords")
            all_kp[a, :, j, 0] = s["x"].to_numpy()
            all_kp[a, :, j, 1] = s["y"].to_numpy()
            all_c[a, :, j] = s["likelihood"].to_numpy()
    # DeepLabCut 3 escribe -1 en x, y y likelihood cuando no hay detección
    missing = ~(all_c > 0) | ~np.isfinite(all_kp).all(-1)
    all_kp[missing] = np.nan
    all_c[missing] = 0.0
    best = all_c.mean(axis=2).argmax(axis=0)  # (F,)
    fr = np.arange(F)
    return bodyparts, all_kp[best, fr], np.nan_to_num(all_c[best, fr])


def _latest_h5(folder: Path, stem: str) -> Path:
    found = sorted(folder.glob(f"{stem}*.h5"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"DeepLabCut no generó ningún .h5 para {stem} en {folder}")
    return found[-1]


def _select(kp, conf, frames):
    frames = np.asarray(frames)
    if len(frames) and frames.max() >= len(kp):
        raise IndexError(f"El análisis tiene {len(kp)} frames y se pidió el {frames.max()}")
    return kp[frames], conf[frames]


class SuperAnimalHorse:
    names = HORSE_KP

    def __init__(self, work_dir: str | Path, model_name: str = "hrnet_w32",
                 detector_name: str = "fasterrcnn_resnet50_fpn_v2", device: str | None = None,
                 max_individuals: int = 2, batch_size: int = 8, bbox_threshold: float = 0.5):
        self.work_dir = Path(work_dir)
        self.model_name, self.detector_name, self.device = model_name, detector_name, device
        self.max_individuals, self.batch_size, self.bbox_threshold = max_individuals, batch_size, bbox_threshold

    def detect_video(self, path: str, frames: list[int], context: dict | None = None):
        dlc = _deeplabcut()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem
        try:
            h5 = _latest_h5(self.work_dir, stem)
        except FileNotFoundError:
            kwargs = {"device": self.device} if self.device else {}
            # API de DeepLabCut 3.0 (video_extensions; antes videotype)
            dlc.video_inference_superanimal([str(path)], "superanimal_quadruped", model_name=self.model_name,
                                            detector_name=self.detector_name, video_extensions=Path(path).suffix,
                                            dest_folder=str(self.work_dir), max_individuals=self.max_individuals,
                                            batch_size=self.batch_size, detector_batch_size=self.batch_size,
                                            bbox_threshold=self.bbox_threshold, create_labeled_video=False, **kwargs)
            h5 = _latest_h5(self.work_dir, stem)
        names, kp, conf = read_dlc_h5(h5)
        kp, conf = apply_mapping(names, kp, conf, SA_TO_ROCA, HORSE_KP)
        return _select(kp, conf, frames)


class DLCProjectHorse:
    """Modelo propio: proyecto DeepLabCut cuyos bodyparts son exactamente HORSE_KP."""
    names = HORSE_KP

    def __init__(self, config_path: str | Path, work_dir: str | Path, shuffle: int = 1, device: str | None = None,
                 batch_size: int = 8):
        self.config, self.work_dir, self.shuffle = str(config_path), Path(work_dir), shuffle
        self.device, self.batch_size = device, batch_size

    def cache_token(self) -> str:
        """Cambia cuando se reentrena el modelo (snapshots más recientes)."""
        snaps = list(Path(self.config).parent.glob("dlc-models-pytorch/**/*.pt"))
        return str(max((int(p.stat().st_mtime) for p in snaps), default=0))

    def detect_video(self, path: str, frames: list[int], context: dict | None = None):
        dlc = _deeplabcut()
        out = self.work_dir / f"modelo_{self.cache_token()}"  # una carpeta por versión del modelo
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem
        try:
            h5 = _latest_h5(out, stem)
        except FileNotFoundError:
            kwargs = {"device": self.device} if self.device else {}
            dlc.analyze_videos(self.config, [str(path)], shuffle=self.shuffle, destfolder=str(out),
                               video_extensions=Path(path).suffix, batch_size=self.batch_size, **kwargs)
            h5 = _latest_h5(out, stem)
        names, kp, conf = read_dlc_h5(h5)
        missing = [n for n in HORSE_KP if n not in names]
        if missing:
            raise ValueError(f"El modelo no tiene los puntos de ROCA {missing}. Bodyparts del proyecto: {names}")
        kp, conf = apply_mapping(names, kp, conf, {n: n for n in HORSE_KP}, HORSE_KP)
        return _select(kp, conf, frames)

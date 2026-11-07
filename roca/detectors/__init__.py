"""Detectores 2D de keypoints (caballo y jinete) sobre vídeo real."""
from __future__ import annotations

from pathlib import Path

from .base import Detector, apply_mapping


def build(kind: str, cfg: dict, work_dir: str | Path) -> Detector:
    """Crea un detector a partir de la sección `detectors.<horse|rider>` del session.json."""
    t = cfg.get("type")
    if kind == "rider" and t == "none":
        return NoRider()
    if kind == "rider" and t == "rtmpose":
        from .rider import RTMPoseRider
        return RTMPoseRider(mode=cfg.get("mode", "balanced"), backend=cfg.get("backend", "onnxruntime"),
                            device=cfg.get("device", "cpu"))
    if kind == "horse" and t == "superanimal":
        from .horse import SuperAnimalHorse
        return SuperAnimalHorse(Path(work_dir) / "dlc", model_name=cfg.get("model_name", "hrnet_w32"),
                                detector_name=cfg.get("detector_name", "fasterrcnn_resnet50_fpn_v2"),
                                device=cfg.get("device"), max_individuals=cfg.get("max_individuals", 2),
                                batch_size=cfg.get("batch_size", 8), bbox_threshold=cfg.get("bbox_threshold", 0.5))
    if kind == "horse" and t == "dlc_project":
        from .horse import DLCProjectHorse
        return DLCProjectHorse(cfg["config"], Path(work_dir) / "dlc", shuffle=cfg.get("shuffle", 1),
                               device=cfg.get("device"), batch_size=cfg.get("batch_size", 8))
    raise ValueError(f"Detector desconocido para {kind}: {cfg!r}. "
                     "Caballo: superanimal | dlc_project. Jinete: rtmpose | none.")


class NoRider:
    """Sesión sin jinete (caballo a la mano, datasets como PFERD): no devuelve ningún punto."""

    def __init__(self):
        from ..skeleton import RIDER_KP
        self.names = RIDER_KP

    def detect_video(self, path, frames, context=None):
        import numpy as np
        return np.full((len(frames), len(self.names), 2), np.nan), np.zeros((len(frames), len(self.names)))


__all__ = ["Detector", "NoRider", "apply_mapping", "build"]

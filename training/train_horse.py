"""Entrena el detector del caballo de ROCA (26 puntos) con DeepLabCut 3 (PyTorch).

Requiere Python 3.10-3.12 y requirements-dlc.txt (entorno .venv-dlc). En Apple Silicon usa MPS
(--device mps); con GPU NVIDIA, cuda.

Por defecto parte de los pesos de SuperAnimal-Quadruped (ajuste fino, top-down: detector Faster R-CNN
+ HRNet-w32), que converge con muchas menos épocas que entrenar desde cero. --from-scratch lo desactiva.

Entrada: una o varias carpetas labeled-data/ (de pferd_to_dlc.py y/o frames propios etiquetados con la
GUI de DeepLabCut). Para producción hay que añadir 500-1000 frames propios de pista: PFERD está grabado
en interior, sin jinete, y la pista real cambia luz, fondo y distancia.

Uso:
    python training/train_horse.py --project-dir modelos --labeled data/pferd_dlc/labeled-data \
        --epochs 60 --detector-epochs 20 --device mps
El config.yaml resultante se usa en session.json: {"horse": {"type": "dlc_project", "config": ".../config.yaml"}}
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roca.skeleton import HORSE_EDGES, HORSE_KP  # noqa: E402


def placeholder_video(path: Path, w: int, h: int) -> Path:
    """DeepLabCut exige un vídeo legible para crear el proyecto; se usa uno negro de 5 frames."""
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for _ in range(5):
        vw.write(np.zeros((h, w, 3), np.uint8))
    vw.release()
    return path


def image_size(folders: list[Path]) -> tuple[int, int]:
    import cv2

    img = cv2.imread(str(next(folders[0].glob("*.png"))))
    return img.shape[1], img.shape[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--labeled", nargs="+", required=True, help="carpetas labeled-data/ a combinar")
    ap.add_argument("--name", default="roca-horse")
    ap.add_argument("--experimenter", default="roca")
    ap.add_argument("--net", default="hrnet_w32", help="red de pose (con SuperAnimal: hrnet_w32)")
    ap.add_argument("--detector", default="fasterrcnn_resnet50_fpn_v2")
    ap.add_argument("--from-scratch", action="store_true", help="no partir de SuperAnimal-Quadruped")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--detector-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--detector-batch-size", type=int, default=4)
    ap.add_argument("--save-epochs", type=int, default=10)
    ap.add_argument("--device", help="cpu | mps | cuda (por defecto, el que elija DeepLabCut)")
    ap.add_argument("--check-labels", action="store_true", help="dibuja las etiquetas para revisarlas (lento)")
    a = ap.parse_args()

    try:
        import deeplabcut
        from deeplabcut.core.engine import Engine
        from deeplabcut.core.weight_init import WeightInitialization
        from deeplabcut.utils import auxiliaryfunctions
    except ImportError as e:
        raise SystemExit("Falta DeepLabCut (Python 3.10-3.12): pip install -r requirements-dlc.txt") from e

    folders = [d for src in a.labeled for d in sorted(Path(src).iterdir()) if d.is_dir() and any(d.glob("CollectedData_*.csv"))]
    if not folders:
        raise SystemExit("No hay carpetas con CollectedData_*.csv")
    w, h = image_size(folders)
    proj = Path(a.project_dir).resolve()
    dummy = placeholder_video(proj / "_placeholder" / "placeholder.mp4", w, h)
    config = deeplabcut.create_new_project(a.name, a.experimenter, [str(dummy)], working_directory=str(proj),
                                           copy_videos=False)
    if config == "nothingcreated":
        raise SystemExit("DeepLabCut no creó el proyecto")
    config = str(config)
    cfg = auxiliaryfunctions.read_config(config)
    cfg["bodyparts"] = HORSE_KP
    cfg["skeleton"] = [list(e) for e in HORSE_EDGES]
    # labeled-data/<carpeta> se asocia al vídeo <carpeta>.* de video_sets (no hace falta que exista)
    cfg["video_sets"] = {str(Path(cfg["project_path"]) / "videos" / f"{d.name}.mp4"): {"crop": f"0, {w}, 0, {h}"} for d in folders}
    auxiliaryfunctions.write_config(config, cfg)

    dst = Path(cfg["project_path"]) / "labeled-data"
    for old in dst.iterdir():  # carpeta del vídeo marcador
        shutil.rmtree(old)
    scorers = set()
    for d in folders:
        target = dst / d.name
        shutil.copytree(d, target)
        for csvf in target.glob("CollectedData_*.csv"):
            scorer = csvf.stem.split("_", 1)[1]
            scorers.add(scorer)
            if scorer != a.experimenter:  # DeepLabCut espera CollectedData_<experimenter>
                txt = csvf.read_text().splitlines()
                txt[0] = ",".join(["scorer", "", ""] + [a.experimenter] * (2 * len(HORSE_KP)))
                (target / f"CollectedData_{a.experimenter}.csv").write_text("\n".join(txt) + "\n")
                csvf.unlink()
    n_img = sum(len(list(d.glob("*.png"))) for d in folders)
    print(f"{len(folders)} vídeos, {n_img} frames etiquetados (scorers originales: {sorted(scorers)}), imágenes {w}x{h}")

    deeplabcut.convertcsv2h5(config, userfeedback=False)
    if a.check_labels:
        deeplabcut.check_labels(config, visualizeindividuals=False)

    kwargs = {}
    if a.from_scratch:
        net = a.net
    else:
        net = f"top_down_{a.net}"
        kwargs["weight_init"] = WeightInitialization.build(cfg, super_animal="superanimal_quadruped",
                                                           model_name=a.net, detector_name=a.detector,
                                                           with_decoder=False)
        kwargs["detector_type"] = a.detector
    deeplabcut.create_training_dataset(config, net_type=net, augmenter_type="albumentations",
                                       engine=Engine.PYTORCH, userfeedback=False, **kwargs)
    train_kw = {"device": a.device} if a.device else {}
    deeplabcut.train_network(config, shuffle=1, epochs=a.epochs, batch_size=a.batch_size, save_epochs=a.save_epochs,
                             detector_epochs=0 if a.from_scratch else a.detector_epochs,
                             detector_batch_size=a.detector_batch_size, detector_save_epochs=a.save_epochs, **train_kw)
    deeplabcut.evaluate_network(config, shuffles=[1], plotting=False, per_keypoint_evaluation=True, **train_kw)
    print(f"\nModelo listo. En session.json: \"horse\": {{\"type\": \"dlc_project\", \"config\": \"{config}\"}}")


if __name__ == "__main__":
    main()

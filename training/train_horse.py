"""Entrena el detector del caballo de ROCA (26 puntos) con DeepLabCut 3 (PyTorch).

Requiere Python 3.10-3.12 y `pip install 'deeplabcut[modelzoo]'`; idealmente GPU NVIDIA
(en Apple Silicon funciona con MPS, más lento).

Entrada: una o varias carpetas labeled-data/ (de pferd_to_dlc.py y/o frames propios
etiquetados con la GUI de DeepLabCut). Se recomienda añadir 500-1000 frames propios de
pista: PFERD está grabado en interior y la pista real cambia luz, fondo y distancia.

Uso:
    python training/train_horse.py --project-dir modelos --labeled pferd_dlc/labeled-data propios/labeled-data \
        --net hrnet_w32 --epochs 200
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--labeled", nargs="+", required=True, help="carpetas labeled-data/ a combinar")
    ap.add_argument("--name", default="roca-horse")
    ap.add_argument("--experimenter", default="roca")
    ap.add_argument("--net", default="hrnet_w32")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    try:
        import deeplabcut
        from deeplabcut.utils import auxiliaryfunctions
    except ImportError as e:
        raise SystemExit("Falta DeepLabCut (Python 3.10-3.12): pip install 'deeplabcut[modelzoo]'") from e

    folders = [d for src in a.labeled for d in sorted(Path(src).iterdir()) if d.is_dir() and any(d.glob("CollectedData_*.csv"))]
    if not folders:
        raise SystemExit("No hay carpetas con CollectedData_*.csv")
    # DeepLabCut necesita al menos un vídeo para crear el proyecto; se usa un marcador vacío.
    dummy = Path(a.project_dir) / "_placeholder.mp4"
    dummy.parent.mkdir(parents=True, exist_ok=True)
    dummy.touch()
    config = deeplabcut.create_new_project(a.name, a.experimenter, [str(dummy)], working_directory=a.project_dir,
                                           copy_videos=False)
    cfg = auxiliaryfunctions.read_config(config)
    cfg["bodyparts"] = HORSE_KP
    cfg["skeleton"] = [list(e) for e in HORSE_EDGES]
    cfg["video_sets"] = {str(Path(cfg["project_path"]) / "videos" / f"{d.name}.mp4"): {"crop": "0, 3840, 0, 2160"} for d in folders}
    auxiliaryfunctions.write_config(config, cfg)

    dst = Path(cfg["project_path"]) / "labeled-data"
    scorers = set()
    for d in folders:
        target = dst / d.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(d, target)
        for csvf in target.glob("CollectedData_*.csv"):
            scorer = csvf.stem.split("_", 1)[1]
            scorers.add(scorer)
            if scorer != a.experimenter:  # DeepLabCut espera CollectedData_<experimenter>
                txt = csvf.read_text().splitlines()
                txt[0] = ",".join(["scorer", "", ""] + [a.experimenter] * (2 * len(HORSE_KP)))
                (target / f"CollectedData_{a.experimenter}.csv").write_text("\n".join(txt) + "\n")
                csvf.unlink()
    print(f"{len(folders)} vídeos etiquetados copiados (scorers originales: {sorted(scorers)})")

    deeplabcut.convertcsv2h5(config, userfeedback=False)
    deeplabcut.check_labels(config, visualizeindividuals=False)
    deeplabcut.create_training_dataset(config, net_type=a.net, augmenter_type="albumentations")
    deeplabcut.train_network(config, shuffle=1, epochs=a.epochs, batch_size=a.batch_size)
    deeplabcut.evaluate_network(config, Shuffles=[1], plotting=True)
    print(f"\nModelo listo. En session.json: \"horse\": {{\"type\": \"dlc_project\", \"config\": \"{config}\"}}")


if __name__ == "__main__":
    main()

"""Descarga PFERD desde Harvard Dataverse y lo deja con la estructura que espera ROCA (roca/pferd.py).

Multiplataforma (Windows, macOS, Linux), sin dependencias. Reanuda descargas cortadas, comprueba el
MD5 de cada archivo y extrae los .tar.gz de forma segura.

    python scripts/download_pferd.py --demo                        # caballo ID_1 de test (~7 GB)
    python scripts/download_pferd.py --subjects ID_3 ID_5          # entrenamiento (~56 GB)
    python scripts/download_pferd.py --subjects ID_3 --no-video    # sólo calibración y keypoints
    python scripts/download_pferd.py --organize                    # reordenar lo ya extraído

Resultado: data/PFERD/dataset/<ID_n>/{CAM_DATA,KP2D_DATA/<prueba>,VIDEO_DATA/<prueba>}
           data/PFERD/dataset/DEMO/ID_1/... (con --demo)

Tamaños: ID_1 ~78 GB de vídeo, ID_2 ~118 GB, ID_3 ~20 GB, ID_4 ~88 GB, ID_5 ~36 GB.
Licencia de los datos: investigación no comercial, sin redistribución.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/2EXONE"
FILE = "https://dataverse.harvard.edu/api/access/datafile/{id}"
UA = {"User-Agent": "roca-pferd-downloader"}


def list_files() -> list[dict]:
    with urllib.request.urlopen(urllib.request.Request(API, headers=UA), timeout=60) as r:
        d = json.load(r)
    out = []
    for f in d["data"]["latestVersion"]["files"]:
        df = f["dataFile"]
        out.append({"id": df["id"], "name": df["filename"], "dir": f.get("directoryLabel", ""),
                    "size": df["filesize"], "md5": df.get("md5") or df.get("checksum", {}).get("value")})
    return out


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(f: dict, dest: Path, retries: int = 8) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == f["size"]:
        return dest
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > f["size"]:
            part.unlink()
            have = 0
        req = urllib.request.Request(FILE.format(id=f["id"]), headers={**UA, "Range": f"bytes={have}-"} if have else UA)
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have and r.status == 206 else "wb") as out:
                if have and r.status != 206:
                    have = 0
                t0, done = time.time(), have
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if time.time() - t0 > 5:
                        print(f"\r  {dest.name}: {done / 1e9:.2f}/{f['size'] / 1e9:.2f} GB", end="", flush=True)
                        t0 = time.time()
            if part.stat().st_size == f["size"]:
                break
        except Exception as e:  # red inestable: reintentar reanudando
            print(f"\n  {dest.name}: {e} (reintento {attempt}/{retries})")
            time.sleep(min(60, 5 * attempt))
    else:
        raise SystemExit(f"No se pudo descargar {dest.name}")
    print(f"\r  {dest.name}: {f['size'] / 1e9:.2f} GB, comprobando MD5...", end="", flush=True)
    if f["md5"] and md5sum(part) != f["md5"]:
        part.unlink()
        raise SystemExit(f"\nMD5 incorrecto en {dest.name}; vuelve a ejecutar para descargarlo de nuevo")
    part.replace(dest)
    print(" ok")
    return dest


class _Concat:
    """Lectura secuencial de varios archivos como si fueran uno (partes .part-aa, -ab...)."""

    def __init__(self, paths):
        self.paths, self.fh = list(paths), None

    def read(self, n=-1):
        while True:
            if self.fh is None:
                if not self.paths:
                    return b""
                self.fh = open(self.paths.pop(0), "rb")
            data = self.fh.read(n)
            if data:
                return data
            self.fh.close()
            self.fh = None


def extract(sources: list[Path], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    stream = _Concat(sources)
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for m in tar:
            if not (m.isfile() or m.isdir()):
                continue
            target = (dest / m.name).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise SystemExit(f"Ruta peligrosa en el tar: {m.name}")
            tar.extract(m, dest, filter="data")


def organize(subject_dir: Path) -> None:
    """Coloca cada archivo según su tipo, sea cual sea la estructura interna de los tar."""
    for p in list(subject_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(subject_dir)
        if rel.parts[0] in ("CAM_DATA", "KP2D_DATA", "VIDEO_DATA") and len(rel.parts) >= (2 if rel.parts[0] == "CAM_DATA" else 3):
            continue
        if re.match(r"Camera_.*\.npz$", p.name) or p.name == "note.txt" and "VIDEO" not in str(rel):
            dst = subject_dir / "CAM_DATA" / p.name
        elif p.name.endswith("_2Dkp.npz"):
            dst = subject_dir / "KP2D_DATA" / p.parent.name / p.name
        elif p.suffix.lower() == ".avi":
            dst = subject_dir / "VIDEO_DATA" / p.parent.name / p.name
        else:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst != p:
            shutil.move(str(p), str(dst))
    for d in sorted((x for x in subject_dir.rglob("*") if x.is_dir()), key=lambda x: -len(x.parts)):
        if not any(d.iterdir()):
            d.rmdir()


def fetch_subject(files: list[dict], subject: str, data: Path, video: bool) -> None:
    sub = [f for f in files if f["dir"].split("/")[0] == subject]
    if not sub:
        raise SystemExit(f"No hay archivos de {subject} en Dataverse")
    out = data / "dataset" / subject
    dl = data / "_downloads" / subject
    for kind in ("CAM_DATA", "KP2D_DATA"):
        f = next(x for x in sub if x["name"] == f"{kind}.tar.gz")
        if (out / kind).exists() and any((out / kind).iterdir()):
            print(f"{subject} {kind}: ya extraído")
            continue
        print(f"{subject} {kind} ({f['size'] / 1e9:.2f} GB)")
        extract([download(f, dl / f["name"])], out)
        organize(out)
    if video:
        parts = sorted((x for x in sub if x["name"].startswith("VIDEO_DATA.tar.gz.part-")), key=lambda x: x["name"])
        if (out / "VIDEO_DATA").exists() and any((out / "VIDEO_DATA").rglob("*.avi")):
            print(f"{subject} VIDEO_DATA: ya extraído")
        else:
            total = sum(p["size"] for p in parts)
            print(f"{subject} VIDEO_DATA: {len(parts)} partes, {total / 1e9:.1f} GB")
            paths = [download(p, dl / p["name"]) for p in parts]
            print(f"{subject}: extrayendo vídeo...")
            extract(paths, out)
            organize(out)
            for p in paths:
                p.unlink()
    print(f"{subject} listo en {out}")


def fetch_demo(files: list[dict], data: Path) -> None:
    for f in files:
        if not f["dir"].startswith("DEMO/") or f["name"].endswith(("_seg.mp4", ".fbx", "_hsmal.npz", "stagei.npz")):
            continue
        download(f, data / "dataset" / f["dir"] / f["name"])
    print(f"DEMO listo en {data / 'dataset' / 'DEMO'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "PFERD"))
    ap.add_argument("--demo", action="store_true", help="prueba de demostración de ID_1 (caballo de test)")
    ap.add_argument("--subjects", nargs="*", default=[], help="ID_1 ... ID_5")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--organize", action="store_true", help="sólo reordenar las carpetas ya extraídas")
    a = ap.parse_args()
    data = Path(a.data)
    if a.organize:
        for d in sorted((data / "dataset").glob("ID_*")):
            organize(d)
            print(f"{d.name}: ordenado")
        return
    if not a.demo and not a.subjects:
        ap.error("indica --demo y/o --subjects")
    files = list_files()
    if a.demo:
        fetch_demo(files, data)
    for s in a.subjects:
        fetch_subject(files, s, data, video=not a.no_video)


if __name__ == "__main__":
    main()

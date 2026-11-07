"""Vídeo de comprobación: esqueleto 3D reconstruido (y el del mocap) proyectado sobre el vídeo real
de cada cámara de PFERD. Mosaico de las cámaras, recortado alrededor del caballo, en H.264.

    python scripts/pferd_overlay.py --runs superanimal_6c gt_ruido3_6c
    -> web/data/pferd_video/<run>.mp4   (local: contiene imágenes de PFERD, no se sube a git)

Colores: naranja = reconstrucción de ROCA desde el vídeo · verde = mocap (referencia).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from roca.calibration import load_camera  # noqa: E402
from roca.skeleton import HORSE_EDGES, HORSE_KP  # noqa: E402
from roca.video import iter_frames  # noqa: E402

REC = (0, 140, 255)   # BGR naranja
REF = (90, 200, 60)   # BGR verde
EDGES = [(HORSE_KP.index(a), HORSE_KP.index(b)) for a, b in HORSE_EDGES]


def project(cam, D, X: np.ndarray) -> np.ndarray:
    """X (F,J,3) -> píxeles con distorsión (F,J,2); NaN se conserva."""
    F, J, _ = X.shape
    flat = X.reshape(-1, 3)
    ok = np.isfinite(flat).all(1)
    uv = np.full((len(flat), 2), np.nan)
    if ok.any():
        p, _ = cv2.projectPoints(flat[ok], cv2.Rodrigues(cam.R)[0], cam.t, cam.K, D if D is not None else np.zeros(5))
        uv[ok] = p.reshape(-1, 2)
    return uv.reshape(F, J, 2)


def draw(img, uv, color, thick):
    for a, b in EDGES:
        p, q = uv[a], uv[b]
        if np.isfinite(p).all() and np.isfinite(q).all():
            cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), color, thick, cv2.LINE_AA)
    for p in uv:
        if np.isfinite(p).all():
            cv2.circle(img, tuple(np.round(p).astype(int)), thick + 2, color, -1, cv2.LINE_AA)


def crop_track(uv_ref: np.ndarray, W: int, H: int, aspect: float, margin: float = 1.5, smooth: int = 15):
    """Ventana de recorte por frame: centrada en el caballo (suavizada), tamaño fijo."""
    with np.errstate(all="ignore"):
        lo, hi = np.nanmin(uv_ref, axis=1), np.nanmax(uv_ref, axis=1)
    c = (lo + hi) / 2
    size = np.nanpercentile(hi - lo, 90, axis=0) * margin
    w = max(size[0], size[1] * aspect, 160)
    h = w / aspect
    t = np.arange(len(c))
    for k in range(2):
        ok = np.isfinite(c[:, k])
        c[:, k] = np.interp(t, t[ok], c[ok, k]) if ok.any() else (W / 2, H / 2)[k]
        c[:, k] = np.convolve(np.pad(c[:, k], smooth // 2, mode="edge"), np.ones(smooth) / smooth, mode="valid")
    x0 = np.clip(c[:, 0] - w / 2, 0, max(0, W - w)).astype(int)
    y0 = np.clip(c[:, 1] - h / 2, 0, max(0, H - h)).astype(int)
    return x0, y0, int(min(w, W)), int(min(h, H))


class Writer:
    """H.264 con imageio-ffmpeg (se reproduce en el navegador); si no está, mp4v de OpenCV."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float):
        self.path, self.cv = path, None
        try:
            import imageio_ffmpeg
            self.gen = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec="libx264", quality=7,
                                                   pix_fmt_out="yuv420p", macro_block_size=8)
            self.gen.send(None)
        except Exception:
            self.gen = None
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            print("  (aviso: sin imageio-ffmpeg; el mp4v puede no reproducirse en el navegador)")

    def write(self, bgr):
        if self.gen is not None:
            self.gen.send(np.ascontiguousarray(bgr[..., ::-1]))
        else:
            self.cv.write(bgr)

    def close(self):
        if self.gen is not None:
            self.gen.close()
        else:
            self.cv.release()


def render(run: str, session_dir: Path, out: Path, tile_w: int = 480, cols: int = 3, show_ref: bool = True):
    d = np.load(session_dir.parent / f"{run}.npz", allow_pickle=False)
    X, G, frames, fps = d["X"], d["G"], d["frames"], float(d["fps"])
    info = json.loads((session_dir.parent / f"{run}.json").read_text(encoding="utf-8"))
    cfg = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    cams = [c for c in cfg["cameras"] if c["name"] in info["cams"]]
    aspect = 4 / 3
    tile_h = int(tile_w / aspect)
    tiles = []
    for c in cams:
        cam, D = load_camera(c["calibration"])
        uv_x, uv_g = project(cam, D, X), project(cam, D, G)
        x0, y0, w, h = crop_track(uv_g, cam.width, cam.height, aspect)
        tiles.append({"name": c["name"], "video": c["video"], "uv_x": uv_x, "uv_g": uv_g,
                      "x0": x0, "y0": y0, "w": w, "h": h, "thick": max(1, int(round(w / 320)))})
    rows = int(np.ceil(len(tiles) / cols))
    size = (tile_w * cols, tile_h * rows + 40)
    readers = [iter_frames(t["video"], start=int(frames[0]), stop=int(frames[-1]) + 1) for t in tiles]
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = Writer(out, size, fps)
    err = np.nanmean(np.linalg.norm(X - G, axis=-1), axis=1) * 100
    for k in range(len(frames)):
        canvas = np.zeros((size[1], size[0], 3), np.uint8)
        for i, (t, rd) in enumerate(zip(tiles, readers)):
            _, img = next(rd)
            if show_ref:
                draw(img, t["uv_g"][k], REF, t["thick"])
            draw(img, t["uv_x"][k], REC, t["thick"] + 1)
            crop = img[t["y0"][k]:t["y0"][k] + t["h"], t["x0"][k]:t["x0"][k] + t["w"]]
            tile = cv2.resize(crop, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, t["name"].split("_")[-1], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            r, cidx = divmod(i, cols)
            canvas[40 + r * tile_h:40 + (r + 1) * tile_h, cidx * tile_w:(cidx + 1) * tile_w] = tile
        head = f"{info['detector'].get('type', '')}  ·  frame {int(frames[k]) + info['pferd_frames'][0] - int(frames[0])}  ·  error medio {err[k]:.1f} cm"
        cv2.putText(canvas, "ROCA (reconstruccion)", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, REC, 2, cv2.LINE_AA)
        if show_ref:
            cv2.putText(canvas, "mocap (referencia)", (260, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, REF, 2, cv2.LINE_AA)
        cv2.putText(canvas, head, (480, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        wr.write(canvas)
    wr.close()
    print(f"  {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", help="nombres de data/validacion/<run>.npz (por defecto, todos)")
    ap.add_argument("--session-dir", default=str(ROOT / "data" / "validacion" / "trote_450_750"))
    ap.add_argument("--out-dir", default=str(ROOT / "web" / "data" / "pferd_video"))
    ap.add_argument("--tile-width", type=int, default=480)
    ap.add_argument("--sin-mocap", action="store_true", help="no dibujar el esqueleto del mocap")
    a = ap.parse_args()
    sdir = Path(a.session_dir)
    runs = a.runs or sorted(p.stem for p in sdir.parent.glob("*.npz"))
    for r in runs:
        render(r, sdir, Path(a.out_dir) / f"{r}.mp4", a.tile_width, show_ref=not a.sin_mocap)


if __name__ == "__main__":
    main()

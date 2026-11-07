"""Calibración extrínseca de una cámara con las letras y esquinas de la pista.

Necesita la intrínseca (calibrate_intrinsics.py) y los píxeles de al menos 4 puntos de la
pista (recomendado 6+, repartidos). Los puntos son la base de cada letra en el suelo
(A K E H C M B F D X G) y las esquinas AK AF CH CM. Ver roca/arena.py.

Dos formas de dar los píxeles:
  --points puntos_cam1.json       {"A": [u, v], "K": [u, v], ...}
  --click  (abre una ventana: pulsa la tecla de la letra y luego haz clic en el punto;
            para esquinas pulsa 1=AK 2=AF 3=CH 4=CM; Enter para terminar; Esc cancela)

Uso:
    python scripts/calibrate_extrinsics.py --intrinsics calib/cam1_intr.json \
        --frame-from videos/cam1.mp4 --click --out calib/cam1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from roca import arena  # noqa: E402
from roca.calibration import calibrate_extrinsics, save_camera  # noqa: E402
from roca.camera import Camera  # noqa: E402
from roca.video import read_frame  # noqa: E402

CORNER_KEYS = {"1": "AK", "2": "AF", "3": "CH", "4": "CM"}


def click_points(img) -> dict[str, list[float]]:  # pragma: no cover - interactivo
    import cv2
    pts: dict[str, list[float]] = {}
    state = {"name": None}
    scale = min(1.0, 1600 / img.shape[1])
    view = cv2.resize(img, None, fx=scale, fy=scale)

    def redraw():
        v = view.copy()
        for n, (u, w) in pts.items():
            p = (int(u * scale), int(w * scale))
            cv2.circle(v, p, 6, (0, 255, 255), 2)
            cv2.putText(v, n, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        msg = f"Punto: {state['name'] or '(pulsa letra o 1-4)'}  |  Enter=terminar  Esc=cancelar"
        cv2.putText(v, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("ROCA calibracion", v)

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and state["name"]:
            pts[state["name"]] = [x / scale, y / scale]
            state["name"] = None
            redraw()

    cv2.namedWindow("ROCA calibracion")
    cv2.setMouseCallback("ROCA calibracion", on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Cancelado")
        if k in (13, 10):
            break
        ch = chr(k).upper()
        if ch in arena.LETTERS:
            state["name"] = ch
        elif ch in CORNER_KEYS:
            state["name"] = CORNER_KEYS[ch]
        redraw()
    cv2.destroyAllWindows()
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--points")
    g.add_argument("--click", action="store_true")
    ap.add_argument("--frame-from", help="vídeo (o imagen) de la cámara montada, para --click")
    ap.add_argument("--frame", type=int, default=0)
    a = ap.parse_args()

    intr = json.loads(Path(a.intrinsics).read_text(encoding="utf-8"))
    K, D = np.array(intr["K"], float), np.array(intr["D"], float)
    if a.points:
        pts = json.loads(Path(a.points).read_text(encoding="utf-8"))
    else:
        import cv2
        if not a.frame_from:
            ap.error("--click necesita --frame-from")
        img = cv2.imread(a.frame_from) if Path(a.frame_from).suffix.lower() in (".png", ".jpg", ".jpeg") else read_frame(a.frame_from, a.frame)
        pts = click_points(img)
        Path(a.out).with_suffix(".points.json").write_text(json.dumps(pts, indent=1), encoding="utf-8")

    R, t, rmse, n = calibrate_extrinsics(K, D, pts)
    cam = Camera(intr["name"], K, R, t, intr["width"], intr["height"], intr.get("fps", 50.0),
                 {"calib_points": n, "calib_rmse_px": round(rmse, 2), "intrinsic_rms_px": intr.get("intrinsic_rms_px")})
    save_camera(a.out, cam, D)
    c = cam.center
    print(f"{cam.name}: {n} puntos, RMSE {rmse:.2f} px, cámara en ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) m -> {a.out}")
    if rmse > 3:
        print("AVISO: RMSE > 3 px. Revisa que cada letra esté bien asignada y marcada en su base.")
    if c[2] < 0:
        print("AVISO: la cámara queda bajo el suelo; probablemente hay letras confundidas (izquierda/derecha).")


if __name__ == "__main__":
    main()

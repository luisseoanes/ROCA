"""Resume las validaciones con PFERD (data/validacion/*.json) para la web -> web/data/pferd.js.

web/data/pferd.js (se sube a git): sólo métricas agregadas y el error por frame.
web/data/pferd_local.js (NO se sube a git): esqueletos 3D (reconstrucciones y mocap) para el visor y la
lista de vídeos de scripts/pferd_overlay.py. Contiene datos derivados de PFERD, cuya licencia no permite
redistribuirlos: sólo para uso interno.

    python scripts/pferd_report.py                          # todos los JSON de data/validacion/
    python scripts/pferd_report.py --runs a.json b.json     # en ese orden
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from roca.skeleton import HORSE_KP  # noqa: E402


def label(r: dict) -> str:
    n = r["n_cams"]
    if r["source"] == "gt":
        if not r.get("noise_px"):
            return f"2D exacto · {n} cám."
        return f"Simulado {r['noise_px']:g} px, {r['dropout']:.0%} pérdidas · {n} cám."
    t = r["detector"].get("type")
    name = {"superanimal": "SuperAnimal sin ajustar", "dlc_project": "ROCA ajustado con PFERD"}.get(t, t)
    return f"{name} · {n} cám."


def kind(r: dict) -> str:
    if r["source"] == "gt":
        return "referencia" if not r.get("noise_px") else "simulado"
    return "detector"


def trace(npz: Path, step: int = 2) -> dict | None:
    """Error 3D medio por frame y altura del casco anterior izquierdo reconstruida (no el mocap)."""
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=False)
    X, G, fps = d["X"], d["G"], float(d["fps"])
    e = np.nanmean(np.linalg.norm(X - G, axis=-1), axis=1) * 100
    hz = X[:, HORSE_KP.index("LF_hoof"), 2] * 100
    sl = slice(None, None, step)
    r = lambda a: [None if not np.isfinite(v) else round(float(v), 1) for v in a[sl]]  # noqa: E731
    return {"t": [round(i / fps, 3) for i in range(len(X))][sl], "err_cm": r(e), "lf_hoof_z_cm": r(hz)}


def write_local(files: list[Path], runs: list[dict], out: Path) -> None:
    """Esqueletos 3D en mm (enteros) para el visor + vídeos disponibles. Uso interno."""
    from roca.skeleton import HORSE_EDGES

    mm = lambda A: [None if not np.isfinite(v) else int(round(v * 1000)) for v in A.ravel()]  # noqa: E731
    by_id = {f.stem: f for f in files}
    data = {"names": HORSE_KP, "edges": [[HORSE_KP.index(a), HORSE_KP.index(b)] for a, b in HORSE_EDGES],
            "runs": {}, "videos": {}}
    for r in runs:
        npz = by_id[r["id"]].with_suffix(".npz")
        if not npz.exists():
            continue
        d = np.load(npz, allow_pickle=False)
        X, G = d["X"], d["G"]
        if "mocap" not in data:
            data.update({"fps": float(d["fps"]), "n": int(len(X)), "mocap": mm(G)})
        data["runs"][r["id"]] = mm(X)
        vid = ROOT / "web" / "data" / "pferd_video" / f"{r['id']}.mp4"
        if vid.exists():
            data["videos"][r["id"]] = f"data/pferd_video/{vid.name}"
    out.write_text("window.ROCA_PFERD_LOCAL = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"Visor 3D y vídeos (uso interno, no se sube a git): {out} ({out.stat().st_size / 1e6:.1f} MB), "
          f"{len(data['runs'])} reconstrucciones, {len(data['videos'])} vídeos")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--out", default=str(ROOT / "web" / "data" / "pferd.js"))
    ap.add_argument("--local-out", default=str(ROOT / "web" / "data" / "pferd_local.js"))
    ap.add_argument("--sin-local", action="store_true", help="no escribir pferd_local.js")
    a = ap.parse_args()
    files = [Path(f) for f in a.runs] if a.runs else sorted((ROOT / "data" / "validacion").glob("*.json"))
    runs = []
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("dataset") != "PFERD":
            continue
        e = r["error_3d"]
        item = {"id": f.stem, "label": label(r), "kind": kind(r), "n_cams": r["n_cams"], "n_frames": r["n_frames"],
                "fps": r["fps"], "subject": r["subject"], "trial": r["trial"], "frames": r.get("pferd_frames"),
                "mpjpe_cm": e["mpjpe_cm"], "median_cm": e["median_cm"], "p95_cm": e["p95_cm"],
                "pct_under_5cm": e["pct_under_5cm"], "pct_under_10cm": e["pct_under_10cm"],
                "coverage_pct": e["coverage_pct"], "by_group_cm": e["by_group_cm"], "by_point_cm": e["by_point_cm"],
                "reproj_px": r["reproj_px"], "mean_views": r["mean_views"], "derived_points": r["derived_points"]}
        if "error_2d" in r:
            item["det_2d"] = {k: r["error_2d"][k] for k in ("mean_px", "pck20_pct", "detected_pct")}
            item["det_2d_by_point_px"] = {}
            per = [c["by_point_px"] for c in r["error_2d"]["by_camera"].values()]
            for n in HORSE_KP:
                v = [p[n] for p in per if p.get(n) is not None]
                if v:
                    item["det_2d_by_point_px"][n] = round(float(np.mean(v)), 1)
        tr = trace(f.with_suffix(".npz"))
        if tr:
            item["trace"] = tr
        runs.append(item)
    order = {"referencia": 0, "simulado": 1, "detector": 2}
    runs.sort(key=lambda x: (order[x["kind"]], -x["n_cams"] if x["kind"] != "detector" else 0, x["mpjpe_cm"] or 0))
    data = {"dataset": "PFERD (Li et al., Scientific Data 2024)", "doi": "10.7910/DVN/2EXONE",
            "note": "Caballo ID_1 (no usado para entrenar). Referencia: mocap de marcadores (Qualisys).",
            "runs": runs}
    Path(a.out).write_text("window.ROCA_PFERD = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"{len(runs)} validaciones -> {a.out}")
    if not a.sin_local:
        write_local(files, runs, Path(a.local_out))
    for r in runs:
        print(f"  {r['mpjpe_cm']:6.2f} cm  {r['label']}")


if __name__ == "__main__":
    main()

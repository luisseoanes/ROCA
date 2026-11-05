"""Exporta el resultado del pipeline a ficheros JS que consume la web (web/).

Se usan .js (window.ROCA_*) en lugar de .json para que index.html funcione
abriéndolo directamente desde el disco, sin servidor.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from . import arena
from .protocol import MOVEMENTS, TP
from .skeleton import HORSE_EDGES, HORSE_KP, RIDER_EDGES, RIDER_KP, edges_idx

GAIT_CODE = {"halt": 0, "walk": 1, "trot": 2, "canter_L": 3, "canter_R": 4, "unknown": 5}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if np.isnan(v) else round(v, 4)
    if isinstance(o, float):
        return None if np.isnan(o) else round(o, 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _smooth(x, fps, win_s):
    """Media móvil para visualización (ignora NaN)."""
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    num = uniform_filter1d(np.where(ok, x, 0.0), max(1, int(win_s * fps)))
    den = uniform_filter1d(ok.astype(float), max(1, int(win_s * fps)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def _downsample(x, src_fps, dst_fps):
    step = max(1, int(round(src_fps / dst_fps)))
    return x[::step]


def build_report(out: dict, meta: dict | None = None) -> dict:
    an, sc, fps = out["analysis"], out["score"], out["fps"]
    ser = an["series"]
    sfps = 5.0
    lab = np.array([GAIT_CODE.get(str(g), 5) for g in ser["labels"]])
    expected = []
    for mv in MOVEMENTS:
        for ck in mv.checks:
            if ck["type"] == "circle":
                expected.append({"type": "circle", "mov": mv.n, "center": list(ck["center"]), "r": ck["diameter"] / 2})
            elif ck["type"] == "line":
                expected.append({"type": "line", "mov": mv.n, "a": list(TP[ck["from"]]), "b": list(TP[ck["to"]])})
    movements_metrics = []
    for m in an["movements"]:
        mm = {k: v for k, v in m.items() if k not in ("f0", "f1")}
        movements_metrics.append(mm)
    hnp_canter = an["by_gait"].get("canter", {}).get("hnp", {})
    welfare = {
        "by_gait": {g: v["hnp"] for g, v in an["by_gait"].items()},
        "alert": bool(hnp_canter and hnp_canter.get("pct_behind10", 0) > 25),
        "message": None,
        "refs": [
            "Kienapfel et al. (2014) — Prevalence of different head-neck positions in horses shown at dressage competitions (hnp_prevalencia_concurso_2014.pdf).",
            "Revisión sistemática 2024 — efectos de montar detrás de la vertical / hiperflexión (hiperflexion_revision_sistematica_2024.pdf).",
            "Preprint élite: calentamiento vs. concurso (hnp_elite_calentamiento_vs_concurso_preprint.pdf).",
            "FEI — Manual de jueces de doma 2025: nuca como punto más alto y nariz ligeramente por delante de la vertical.",
        ],
    }
    if welfare["alert"]:
        welfare["message"] = (f"En el galope la nariz va de media {abs(hnp_canter['mean_deg']):.1f}° detrás de la vertical y "
                              f"{hnp_canter['pct_behind10']:.0f}% del tiempo más de 10° detrás. La literatura asocia esta postura con "
                              f"más conductas de conflicto; el sistema la penaliza explícitamente y no la aprende de las notas de los jueces.")
    cams = [c.to_dict() for c in out["rig"]]
    report = {
        "meta": {
            "title": "Informe de prueba de doma",
            "test": "Prueba de doma — Jinetes alevines (RFHE 2026, Anexo II)",
            "rider": "Binomio de demostración",
            "horse": "Caballo sintético (PoC)",
            "synthetic": True,
            "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "fps": fps, "duration_s": an["duration_s"], "n_cams": len(cams),
            **(meta or {}),
        },
        "score": sc,
        "opportunities": out["opportunities"],
        "welfare": welfare,
        "analysis": {"by_gait": an["by_gait"], "transitions": an["transitions"], "events": an["events"],
                     "movements": movements_metrics},
        "validation": out["validation"],
        "calibration": out["calib"],
        "cameras": cams,
        "arena": {"letters": arena.LETTERS, "track": TP, "length": arena.LENGTH, "width": arena.WIDTH},
        "expected": expected,
        "series": {
            "fps": sfps,
            "hnp": _downsample(_smooth(ser["hnp"], fps, 1.0), fps, sfps).round(1),
            "lean": _downsample(_smooth(ser["lean"], fps, 1.0), fps, sfps).round(1),
            "speed": _downsample(ser["speed"], fps, sfps).round(2),
            "gait": _downsample(lab, fps, sfps),
            "xy": _downsample(ser["xy"], fps, sfps).round(2),
        },
    }
    return _clean(report)


def build_motion(out: dict, dst_fps: float = 25.0) -> dict:
    fps = out["fps"]
    X = _downsample(out["X"], fps, dst_fps)
    X = np.nan_to_num(X, nan=0.0)
    labels = _downsample(np.array([GAIT_CODE.get(str(g), 5) for g in out["analysis"]["series"]["labels"]]), fps, dst_fps)
    hnp = _downsample(_smooth(out["analysis"]["series"]["hnp"], fps, 0.5), fps, dst_fps)
    return {
        "fps": dst_fps, "n": int(X.shape[0]), "nh": len(HORSE_KP), "nr": len(RIDER_KP),
        "horse_names": HORSE_KP, "rider_names": RIDER_KP,
        "horse_edges": edges_idx(HORSE_KP, HORSE_EDGES), "rider_edges": edges_idx(RIDER_KP, RIDER_EDGES),
        "unit": "cm",
        "data": np.round(X * 100).astype(int).ravel().tolist(),
        "gait": labels.tolist(),
        "hnp": np.round(np.nan_to_num(hnp), 1).tolist(),
    }


def write_web_data(out: dict, web_dir: str | Path, meta: dict | None = None) -> tuple[Path, Path]:
    d = Path(web_dir) / "data"
    d.mkdir(parents=True, exist_ok=True)
    rep = build_report(out, meta)
    mot = build_motion(out)
    p1 = d / "report.js"
    p2 = d / "motion.js"
    p1.write_text("window.ROCA_REPORT = " + json.dumps(rep, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    p2.write_text("window.ROCA_MOTION = " + json.dumps(mot, separators=(",", ":")) + ";\n", encoding="utf-8")
    (d / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return p1, p2

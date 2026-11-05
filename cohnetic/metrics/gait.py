"""Detección de apoyos, trancos, aire y mano de galope a partir de los cascos 3D.

Patrones de pisada (inicio de apoyo relativo al anterior izquierdo LF, en
fracción de tranco):
  paso      lateral en 4 tiempos  LH .75, RH .25, RF .50
  trote     diagonal en 2 tiempos RH .00, RF .50, LH .50
  galope D  LH, (RH+LF), RF       RH .00, RF .15, LH .70
  galope I  RH, (LH+RF), LF       RH .55, RF .85, LH .85
"""
from __future__ import annotations

import numpy as np

from ..skeleton import hidx

TEMPLATES = {
    "walk": {"RH": 0.25, "RF": 0.50, "LH": 0.75},
    "trot": {"RH": 0.00, "RF": 0.50, "LH": 0.50},
    "canter_R": {"RH": 0.00, "RF": 0.15, "LH": 0.70},
    "canter_L": {"RH": 0.55, "RF": 0.85, "LH": 0.85},
}
LEGS = ("LF", "RF", "LH", "RH")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def detect_contacts(H: np.ndarray, fps: float, z_thr: float = 0.07, v_thr: float = 0.9, min_s: float = 0.08) -> dict:
    from scipy.signal import savgol_filter
    out = {}
    win = max(5, int(0.12 * fps) | 1)
    for leg in LEGS:
        p = H[:, hidx(f"{leg}_hoof")]
        z = savgol_filter(np.nan_to_num(p[:, 2], nan=1.0), win, 2)
        # derivada suavizada: el ruido de reconstrucción (~1 cm) no debe parecer movimiento
        vxy = savgol_filter(np.nan_to_num(p[:, :2]), win, 2, deriv=1, axis=0) * fps
        vh = np.linalg.norm(vxy, axis=1)
        st = (z < z_thr) & (vh < v_thr)
        # eliminar apoyos y vuelos demasiado cortos
        n_min = int(min_s * fps)
        for a, b in _runs(~st):
            if b - a < n_min and a > 0:
                st[a:b] = True
        for a, b in _runs(st):
            if b - a < n_min:
                st[a:b] = False
        out[leg] = st
    return out


def _cdist(a: float, b: float) -> float:
    d = abs(a - b) % 1.0
    return min(d, 1 - d)


def classify_strides(contacts: dict, speed: np.ndarray, fps: float) -> list[dict]:
    onsets = {leg: np.array([a for a, _ in _runs(contacts[leg])]) for leg in LEGS}
    lf = onsets["LF"]
    strides = []
    for a, b in zip(lf[:-1], lf[1:]):
        T = (b - a) / fps
        v = float(np.mean(speed[a:b]))
        if not (0.4 <= T <= 1.8) or v < 0.3:
            continue
        rel = {}
        for leg in ("RH", "RF", "LH"):
            o = onsets[leg]
            o = o[(o >= a - 0.1 * (b - a)) & (o < b - 0.1 * (b - a))]
            if len(o):
                rel[leg] = float(((o[0] - a) / (b - a)) % 1.0)
        gait, best = "unknown", 9.0
        if len(rel) == 3:
            for g, tpl in TEMPLATES.items():
                dsum = sum(_cdist(rel[k], tpl[k]) for k in tpl)
                if dsum < best:
                    gait, best = g, dsum
            if best > 0.33:
                gait = "unknown"
        strides.append({"f0": int(a), "f1": int(b), "T": T, "speed": v, "length": v * T,
                        "gait": gait, "fit": round(best, 3), "rel": rel})
    return strides


def frame_labels(strides: list[dict], speed: np.ndarray, fps: float) -> np.ndarray:
    F = len(speed)
    lab = np.array(["unknown"] * F, dtype=object)
    for s in strides:
        if s["gait"] != "unknown":
            lab[s["f0"]:s["f1"]] = s["gait"]
    still = speed < 0.15
    for a, b in _runs(still):
        if b - a >= int(0.5 * fps):
            lab[a:b] = "halt"
    # rellenar huecos cortos con la etiqueta vecina
    known = np.where(lab != "unknown")[0]
    if len(known):
        idx = np.searchsorted(known, np.arange(F)).clip(0, len(known) - 1)
        prev = known[np.maximum(idx - 1, 0)]
        nxt = known[idx]
        near = np.where(np.abs(np.arange(F) - prev) <= np.abs(nxt - np.arange(F)), prev, nxt)
        gap = np.minimum(np.abs(np.arange(F) - prev), np.abs(nxt - np.arange(F)))
        fill = (lab == "unknown") & (gap < int(0.8 * fps))
        lab[fill] = lab[near[fill]]
    return lab


def detect_halts(speed: np.ndarray, fps: float, thr: float = 0.15, min_s: float = 0.6) -> list[tuple[int, int]]:
    return [(a, b) for a, b in _runs(speed < thr) if b - a >= int(min_s * fps)]


def gait_runs(strides: list[dict], halts: list[tuple[int, int]], min_strides: int = 2) -> list[dict]:
    """Agrupa trancos consecutivos del mismo aire (y paradas) en rachas.

    Las rachas de un único tranco se descartan como ruido de clasificación.
    """
    blocks = [{"f0": s["f0"], "f1": s["f1"], "gait": s["gait"], "T": s["T"]} for s in strides if s["gait"] != "unknown"]
    blocks += [{"f0": a, "f1": b, "gait": "halt", "T": 0.0} for a, b in halts]
    blocks.sort(key=lambda b: b["f0"])
    runs: list[dict] = []
    for b in blocks:
        if runs and runs[-1]["gait"] == b["gait"]:
            runs[-1]["f1"] = max(runs[-1]["f1"], b["f1"])
            runs[-1]["n"] += 1
            runs[-1]["T_last"] = b["T"]
        else:
            runs.append({**b, "n": 1, "T_first": b["T"], "T_last": b["T"]})
    keep = [r for r in runs if r["gait"] == "halt" or r["n"] >= min_strides]
    merged: list[dict] = []
    for r in keep:
        if merged and merged[-1]["gait"] == r["gait"]:
            merged[-1]["f1"] = r["f1"]
            merged[-1]["n"] += r["n"]
            merged[-1]["T_last"] = r["T_last"]
        else:
            merged.append(dict(r))
    return merged


def transitions(runs: list[dict], fps: float, speed: np.ndarray | None = None) -> list[dict]:
    """Instante de cada cambio de aire.

    * Si los aires difieren claramente en velocidad (paso<->trote), el cambio es el
      cruce de la velocidad intermedia dentro de la zona de transición.
    * Si hay hueco entre rachas (trancos de transición sin clasificar), medio tranco
      después del final de la racha anterior (o el centro del hueco si es más corto).
    * Si son contiguas, medio tranco antes del primer apoyo detectado del nuevo aire.
    """
    out = []
    for a, b in zip(runs[:-1], runs[1:]):
        va = vb = None
        if speed is not None and "walk" in (a["gait"], b["gait"]) and "halt" not in (a["gait"], b["gait"]):
            w = int(1.5 * fps)
            va = float(np.median(speed[max(a["f0"], a["f1"] - w):a["f1"]]))
            vb = float(np.median(speed[b["f0"]:min(b["f1"], b["f0"] + w)]))
        if va is not None and abs(vb - va) > 0.6:
            lo = max(a["f0"], a["f1"] - int(1.0 * fps))
            hi = min(b["f1"], b["f0"] + int(0.5 * fps))
            mid = (va + vb) / 2
            seg = speed[lo:hi]
            cross = np.where(seg > mid)[0] if vb > va else np.where(seg < mid)[0]
            f = lo + int(cross[0]) if len(cross) else (a["f1"] + b["f0"]) // 2
        elif b["f0"] > a["f1"] + 2:
            # el primer tranco sin clasificar suele ser ya el de transición
            f = a["f1"] + min((b["f0"] - a["f1"]) // 2, int(0.5 * a["T_last"] * fps))
        elif b["gait"] == "halt":
            f = b["f0"]
        else:
            f = max(a["f0"], b["f0"] - int(0.5 * b["T_first"] * fps))
        out.append({"f": int(f), "from": a["gait"], "to": b["gait"]})
    return out

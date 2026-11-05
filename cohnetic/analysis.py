"""Análisis completo de un recorrido a partir de la reconstrucción 3D.

Entrada: H (F,26,3) caballo, R (F,14,3) jinete, fps.
Salida: dict serializable con series temporales, eventos y métricas por ejercicio.
"""
from __future__ import annotations

import numpy as np

from .metrics import events as EV
from .metrics import gait as G
from .metrics import geometry as GEO
from .metrics import kinematics as KIN
from .metrics import posture as POS
from .protocol import MOVEMENTS, TP
from .segmentation import segment
from .skeleton import hidx


def _r(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def _fam(g: str) -> str:
    return "canter" if g.startswith("canter") else g


def analyze(H: np.ndarray, R: np.ndarray, fps: float) -> dict:
    fr = KIN.horse_frame(H, fps)
    xy, speed, fwd = fr["center"], fr["speed"], fr["fwd"]
    contacts = G.detect_contacts(H, fps)
    strides = G.classify_strides(contacts, speed, fps)
    labels = G.frame_labels(strides, speed, fps)
    halts = G.detect_halts(speed, fps)
    runs = G.gait_runs(strides, halts)
    trans = G.transitions(runs, fps, speed)
    hnp = POS.hnp_angle(H, fwd)
    rider = POS.rider_posture(R, H, fwd, fps)
    caress = EV.detect_caress(R, H, fwd, fps)
    two_hands = EV.detect_two_hands(R, fps)
    segs = segment(xy, fps)
    seg_of = {s["mov"]: s for s in segs}
    withers = fr["withers"]

    movs = []
    for mv in MOVEMENTS:
        sg = seg_of[mv.n]
        f0, f1 = sg["f0"], sg["f1"]
        res = {"n": mv.n, "f0": f0, "f1": f1, "t0": _r(f0 / fps, 1), "t1": _r(f1 / fps, 1), "checks": []}
        win_strides = [s for s in strides if s["f0"] >= f0 and s["f1"] <= f1]
        # Regularidad: coeficiente de variación de la duración del tranco por aire
        reg = {}
        for g in ("walk", "trot", "canter_L", "canter_R"):
            # sólo trancos "estables": mismo aire que sus vecinos y velocidad cercana a la mediana
            ws = [s for k, s in enumerate(win_strides) if s["gait"] == g
                  and 0 < k < len(win_strides) - 1
                  and win_strides[k - 1]["gait"] == g and win_strides[k + 1]["gait"] == g]
            if len(ws) >= 3:
                vmed = np.median([s["speed"] for s in ws])
                ws = [s for s in ws if abs(s["speed"] - vmed) < 0.2 * vmed]
            T = np.array([s["T"] for s in ws])
            if len(T) >= 3:
                med = np.median(T)
                cv_robust = 100 * 1.4826 * np.median(np.abs(T - med)) / med  # CV robusto (MAD)
                reg[g] = {"n": int(len(T)), "T_mean_s": _r(T.mean(), 3), "cv_pct": _r(cv_robust, 1),
                          "length_m": _r(np.mean([s["length"] for s in ws]), 2)}
        res["regularity"] = reg
        # Conformidad del aire (ruptura de aire / galope a mano contraria)
        if mv.gaits:
            margin = int(1.2 * fps)
            bad = [{"f0": r_["f0"], "f1": r_["f1"], "gait": r_["gait"], "strides": r_["n"],
                    "t0": _r(r_["f0"] / fps, 1), "t1": _r(r_["f1"] / fps, 1)}
                   for r_ in runs if r_["gait"] not in mv.gaits
                   and r_["f0"] > f0 + margin and r_["f1"] < f1 - margin]
            if bad:
                res["gait_faults"] = bad
        # HNP y jinete en la ventana
        res["hnp"] = POS.hnp_summary(hnp["nasal_deg"][f0:f1])
        res["rider"] = {"lean_mean_deg": _r(np.nanmean(rider["lean_deg"][f0:f1]), 1),
                        "bounce_std_cm": _r(100 * np.nanstd(rider["bounce_m"][f0:f1]), 1),
                        "hand_jitter_cm": _r(100 * np.nanmean(rider["hand_jitter_m"][f0:f1]), 1)}
        res["speed_mean"] = _r(np.mean(speed[f0:f1]), 2)

        for ck in mv.checks:
            c = {"type": ck["type"]}
            if ck["type"] == "circle":
                m = GEO.circle_metrics(xy[f0:f1], ck["center"], ck["diameter"], TP[ck["at"]], ck["dir"])
                c.update(m or {"missing": True})
            elif ck["type"] == "line":
                m = GEO.line_metrics(xy[f0:f1], TP[ck["from"]], TP[ck["to"]])
                c.update({"from": ck["from"], "to": ck["to"], **(m or {"missing": True})})
            elif ck["type"] == "halt":
                c.update(_halt_check(H, fr, hnp, halts, f0, f1, TP[ck["at"]], fps))
                c.update({"at": ck["at"], "min_s": ck["min_s"]})
            elif ck["type"] == "transition":
                c.update(_transition_check(trans, withers, fwd, f0, f1, ck, fps))
            elif ck["type"] == "transition_zone":
                c.update(_zone_check(trans, withers, f0, f1, ck, fps))
            elif ck["type"] == "lengthening":
                c.update(_lengthening(withers, speed, strides, seg_of, f0, f1, ck, fps))
            elif ck["type"] == "hnp":
                c.update(res["hnp"])
            res["checks"].append(c)
        movs.append(res)

    # Resúmenes globales por aire
    fam = np.array([_fam(g) for g in labels])
    by_gait = {}
    for g in ("walk", "trot", "canter"):
        m = fam == g
        if m.sum() < fps:
            continue
        by_gait[g] = {
            "seconds": _r(m.sum() / fps, 1),
            "speed_mean": _r(speed[m].mean(), 2),
            "hnp": POS.hnp_summary(hnp["nasal_deg"][m]),
            "rider_lean_deg": _r(np.nanmean(rider["lean_deg"][m]), 1),
            "rider_bounce_cm": _r(100 * np.nanstd(rider["bounce_m"][m]), 1),
            "hand_jitter_cm": _r(100 * np.nanmean(rider["hand_jitter_m"][m]), 1),
        }
    return {
        "fps": fps, "n_frames": int(len(xy)), "duration_s": _r(len(xy) / fps, 1),
        "movements": movs, "by_gait": by_gait,
        "events": {"caress": [{**e, "t": _r(e["f0"] / fps, 1)} for e in caress],
                   "two_hands": [{**e, "t": _r(e["f0"] / fps, 1)} for e in two_hands]},
        "transitions": [{**t, "t": _r(t["f"] / fps, 1)} for t in trans],
        "strides": strides,
        "series": {"labels": labels, "speed": speed, "hnp": hnp["nasal_deg"], "lean": rider["lean_deg"],
                   "bounce": rider["bounce_m"], "hand": rider["hand_jitter_m"], "xy": xy},
        "segments": segs,
    }


def _halt_check(H, fr, hnp, halts, f0, f1, letter_pt, fps):
    cands = [(a, b) for a, b in halts if b > f0 and a < f1]
    if not cands:
        return {"missing": True}
    a, b = min(cands, key=lambda h: np.linalg.norm(fr["center"][(h[0] + h[1]) // 2] - letter_pt))
    mid = slice(a + int(0.3 * fps), max(a + int(0.3 * fps) + 1, b - int(0.3 * fps)))
    fwd = fr["fwd"][mid]
    def lon(leg):
        return np.nanmean(np.sum(H[mid, hidx(f"{leg}_hoof")] * fwd, axis=1))
    front = abs(lon("LF") - lon("RF"))
    hind = abs(lon("LH") - lon("RH"))
    wpos = H[mid, hidx("withers")]
    immob = float(np.nanmean(np.nanstd(wpos, axis=0)))
    pos = fr["withers"][(a + b) // 2]
    return {
        "t0": _r(a / fps, 1), "t1": _r(b / fps, 1), "duration_s": _r((b - a) / fps, 1),
        "dist_letter_m": _r(np.linalg.norm(pos - letter_pt), 2),
        "front_offset_cm": _r(100 * front, 1), "hind_offset_cm": _r(100 * hind, 1),
        "immobility_cm": _r(100 * immob, 1),
        "hnp_mean_deg": _r(np.nanmean(hnp["nasal_deg"][a:b]), 1),
    }


def _transition_check(trans, withers, fwd, f0, f1, ck, fps):
    target = ck["to"]
    lo = f0 - int(2 * fps)
    cands = [t for t in trans if lo <= t["f"] <= f1 and t["to"] == target]
    letter_pt = np.asarray(TP[ck["at"]])
    out = {"at": ck["at"], "to": target}
    if not cands:
        out["missing"] = True
        return out
    t = min(cands, key=lambda t: np.linalg.norm(withers[t["f"]] - letter_pt))
    rel = withers[t["f"]] - letter_pt
    along = float(np.dot(rel, fwd[t["f"], :2]))
    out.update({"t": _r(t["f"] / fps, 1), "from": t["from"], "dist_m": _r(np.linalg.norm(rel), 2),
                "late_m": _r(along, 2)})
    return out


def _zone_check(trans, withers, f0, f1, ck, fps):
    cands = [t for t in trans if f0 <= t["f"] <= f1 and t["to"] == ck["to"]]
    if not cands:
        return {"missing": True, "to": ck["to"]}
    t = cands[0]
    x = float(withers[t["f"], 0])
    lo, hi = sorted([TP[ck["to_letter"]][0], TP[ck["from"]][0]])
    outside = max(lo - x, x - hi, 0.0)
    return {"t": _r(t["f"] / fps, 1), "x_m": _r(x, 1), "zone": [lo, hi], "outside_m": _r(outside, 2), "to": ck["to"]}


def _lengthening(withers, speed, strides, seg_of, f0, f1, ck, fps):
    x0, x1 = TP[ck["from"]][0] + 2, TP[ck["to"]][0] - 2
    yside = TP[ck["from"]][1]
    idx = np.arange(f0, f1)
    sel = idx[(withers[idx, 0] > x0) & (withers[idx, 0] < x1) & (np.abs(withers[idx, 1] - yside) < 2.5)]
    ref = seg_of[11]
    v_ext = float(np.mean(speed[sel])) if len(sel) else float("nan")
    v_ref = float(np.mean(speed[ref["f0"]:ref["f1"]]))
    def stride_len(a, b):
        L = [s["length"] for s in strides if s["f0"] >= a and s["f1"] <= b and s["gait"].startswith("canter")]
        return float(np.mean(L)) if L else float("nan")
    L_ext = stride_len(sel[0], sel[-1]) if len(sel) else float("nan")
    L_ref = stride_len(ref["f0"], ref["f1"])
    return {"speed_ext": _r(v_ext), "speed_ref": _r(v_ref), "ratio": _r(v_ext / v_ref, 2),
            "stride_ext_m": _r(L_ext), "stride_ref_m": _r(L_ref), "stride_ratio": _r(L_ext / L_ref, 2),
            "min_ratio": ck["min_ratio"]}

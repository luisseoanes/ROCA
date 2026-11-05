"""Orquestación del pipeline completo.

Real:   vídeos sincronizados -> detector 2D -> [triangulate_session] -> analyze -> score
PoC:    simulate_ride -> make_rig/calibrate_rig -> observe -> [triangulate_session] -> ...

triangulate_session es el mismo código en ambos casos: sólo cambia el origen
de las detecciones 2D.
"""
from __future__ import annotations

import time

import numpy as np

from . import synth
from .advice import opportunities
from .analysis import analyze
from .camera import Camera
from .scoring import score
from .skeleton import HORSE_KP, RIDER_KP
from .triangulation import fill_and_smooth, triangulate

N_H = len(HORSE_KP)


def side_vector() -> np.ndarray:
    s = []
    for n in HORSE_KP:
        s.append(1 if n[:2] in ("LF", "LH") else -1 if n[:2] in ("RF", "RH") else 0)
    for n in RIDER_KP:
        s.append(1 if n.startswith("L_") else -1 if n.startswith("R_") else 0)
    return np.array(s)


def triangulate_session(rig: list[Camera], uv: np.ndarray, fps: float, thr_px: float = 12.0) -> dict:
    """uv (C,F,J,2) detecciones 2D -> puntos 3D suavizados (F,J,3)."""
    C, F, J, _ = uv.shape
    Ps = np.stack([c.P for c in rig])
    X, n_in, err = triangulate(Ps, uv.reshape(C, F * J, 2), thr_px=thr_px)
    X = X.reshape(F, J, 3)
    raw_valid = ~np.isnan(X[..., 0])
    Xs = fill_and_smooth(X, fps)
    return {"X": Xs, "X_raw": X, "n_inliers": n_in.reshape(F, J), "reproj_px": err.reshape(F, J), "raw_valid": raw_valid}


def run_poc(profile: dict, seed: int = 7, n_cams: int = 6, fps: float = 50.0, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed + 1000)
    tlog = {}

    def log(msg):
        if verbose:
            print(msg, flush=True)

    t0 = time.time()
    gt = synth.simulate_ride(profile, fps=fps, seed=seed)
    X_gt = np.concatenate([gt["horse"], gt["rider"]], axis=1)
    tlog["simulate_s"] = time.time() - t0
    log(f"[1/5] Recorrido sintético: {X_gt.shape[0]} frames ({X_gt.shape[0] / fps:.0f} s), {X_gt.shape[1]} keypoints")

    true_rig = synth.make_rig(n_cams=n_cams, fps=fps)
    est_rig, calib = synth.calibrate_rig(true_rig, rng)
    log(f"[2/5] Calibración extrínseca con letras: " + ", ".join(f"{c['name'].split()[0]} {c['calib_rmse_px']}px/{c['pos_err_cm']}cm" for c in calib))

    t0 = time.time()
    uv = synth.observe(true_rig, X_gt, side_vector(), gt["gt"]["yaw"], rng)
    vis = (~np.isnan(uv[..., 0])).mean(axis=(1, 2))
    log(f"[3/5] Detecciones 2D simuladas. Visibilidad por cámara: " + ", ".join(f"{v * 100:.0f}%" for v in vis))

    rec = triangulate_session(est_rig, uv, fps)
    tlog["triangulate_s"] = time.time() - t0
    X = rec["X"]
    err = np.linalg.norm(X - X_gt, axis=-1)
    log(f"[4/5] Triangulación RANSAC + filtrado: MPJPE {np.nanmean(err) * 100:.1f} cm "
        f"(mediana {np.nanmedian(err) * 100:.1f} cm) en {tlog['triangulate_s']:.1f} s")

    H, R = X[:, :N_H], X[:, N_H:]
    an = analyze(H, R, fps)
    sc = score(an)
    opp = opportunities(sc)
    log(f"[5/5] Puntuación: {sc['total']}/{sc['max']} = {sc['percent']}%  ({len(opp)} áreas de mejora)")

    # Validación frente a la verdad (sólo posible en el PoC)
    an_gt = analyze(gt["horse"], gt["rider"], fps)
    valid = _validation(gt, an, an_gt, err, rec, fps)
    return {"gt": gt, "X": X, "rec": rec, "uv_vis": vis, "rig": est_rig, "true_rig": true_rig, "calib": calib,
            "analysis": an, "score": sc, "opportunities": opp, "validation": valid, "fps": fps}


def _validation(gt, an, an_gt, err, rec, fps) -> dict:
    groups = {
        "Cabeza": ["nose", "eye", "poll", "throat"],
        "Tronco": ["withers", "chest", "mid_back", "croup", "tail_base"],
        "Miembros": [n for n in HORSE_KP if any(k in n for k in ("shoulder", "knee", "hip", "hock", "fetlock"))],
        "Cascos": [n for n in HORSE_KP if n.endswith("hoof")],
        "Jinete": RIDER_KP,
    }
    allk = HORSE_KP + RIDER_KP
    g_err = {g: round(float(np.nanmean(err[:, [allk.index(n) for n in names]])) * 100, 1) for g, names in groups.items()}
    lab_est = an["series"]["labels"]
    lab_gt = np.array(gt["gt"]["gait"], dtype=object)
    moving = lab_gt != "halt"
    fam = lambda a: np.array(["canter" if str(x).startswith("canter") else str(x) for x in a])
    gait_acc = float(np.mean(lab_est[moving] == lab_gt[moving]))
    hnp_mae = float(np.nanmean(np.abs(an["series"]["hnp"] - gt["gt"]["hnp"])))
    lean_mae = float(np.nanmean(np.abs(an["series"]["lean"] - gt["gt"]["lean"])))
    score_gt = score(an_gt)
    return {
        "mpjpe_cm": round(float(np.nanmean(err)) * 100, 1),
        "median_cm": round(float(np.nanmedian(err)) * 100, 1),
        "p95_cm": round(float(np.nanpercentile(err, 95)) * 100, 1),
        "group_cm": g_err,
        "raw_coverage_pct": round(float(rec["raw_valid"].mean()) * 100, 1),
        "reproj_px": round(float(np.nanmean(rec["reproj_px"])), 2),
        "gait_accuracy_pct": round(gait_acc * 100, 1),
        "gait_family_accuracy_pct": round(float(np.mean(fam(lab_est[moving]) == fam(lab_gt[moving]))) * 100, 1),
        "hnp_mae_deg": round(hnp_mae, 2),
        "lean_mae_deg": round(lean_mae, 2),
        "score_from_truth_pct": score_gt["percent"],
        "notes_diff": [{"n": a["n"], "rec": a["note"], "truth": b["note"]}
                       for a, b in zip(score(an)["movements"], score_gt["movements"]) if a["note"] != b["note"]],
    }

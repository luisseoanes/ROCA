"""Geometría de figuras: círculos, líneas, posición respecto a las letras."""
from __future__ import annotations

import numpy as np


def fit_circle(xy: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Ajuste algebraico (Kåsa). Devuelve centro, radio y RMS radial."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0] / 2, c[1] / 2
    r = np.sqrt(c[2] + cx ** 2 + cy ** 2)
    res = np.hypot(x - cx, y - cy) - r
    return np.array([cx, cy]), float(r), float(np.sqrt(np.mean(res ** 2)))


def circle_metrics(xy: np.ndarray, spec_center, spec_diam: float, start_pt, direction: str) -> dict | None:
    """xy: trayectoria del ejercicio. Aísla la vuelta de 360° que empieza en la letra."""
    sc = np.asarray(spec_center, float)
    d_start = np.linalg.norm(xy - np.asarray(start_pt), axis=1)
    near = np.where(d_start < 3.0)[0]
    if len(near) == 0:
        return None
    # primer paso por la letra (mínimo local del primer tramo cercano)
    run_end = near[0]
    while run_end + 1 < len(xy) and d_start[run_end + 1] < 3.0:
        run_end += 1
    i0 = near[0] + int(np.argmin(d_start[near[0]:run_end + 1]))
    ang = np.unwrap(np.arctan2(xy[i0:, 1] - sc[1], xy[i0:, 0] - sc[0]))
    sweep = ang - ang[0]
    full = np.where(np.abs(sweep) >= np.radians(355))[0]
    i1 = i0 + (full[0] if len(full) else len(sweep) - 1)
    seg = xy[i0:i1 + 1]
    center, r, rms = fit_circle(seg)
    turned = float(np.degrees(sweep[i1 - i0]))
    dist_c = np.hypot(seg[:, 0] - center[0], seg[:, 1] - center[1])
    return {
        "i0": int(i0), "i1": int(i1),
        "diameter_m": round(2 * r, 2), "expected_m": spec_diam,
        "diam_err_pct": round(100 * (2 * r - spec_diam) / spec_diam, 1),
        "center": [round(float(center[0]), 2), round(float(center[1]), 2)],
        "center_offset_m": round(float(np.linalg.norm(center - sc)), 2),
        "roundness_rms_m": round(rms, 2),
        "radius_range_m": round(float(dist_c.max() - dist_c.min()), 2),
        "direction_ok": bool((turned > 0) == (direction == "L")),
        "closed": bool(abs(turned) >= 355),
        "path": seg[:: max(1, len(seg) // 120)].round(2).tolist(),
    }


def line_metrics(xy: np.ndarray, a, b) -> dict | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    L = np.linalg.norm(ab)
    u = ab / L
    n = np.array([-u[1], u[0]])
    rel = xy - a
    s = rel @ u / L
    lat = rel @ n
    sel = (s > 0.1) & (s < 0.9) & (np.abs(lat) < 3.0)
    if sel.sum() < 10:
        return None
    lat_s = lat[sel]
    return {
        "rms_dev_m": round(float(np.sqrt(np.mean(lat_s ** 2))), 2),
        "max_dev_m": round(float(np.max(np.abs(lat_s))), 2),
        "mean_offset_m": round(float(np.mean(lat_s)), 2),
        "length_m": round(float(L), 1),
    }


def dist_to(pt, p) -> float:
    return float(np.linalg.norm(np.asarray(pt, float) - np.asarray(p, float)))

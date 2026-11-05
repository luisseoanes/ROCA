"""Triangulación multicámara robusta y filtrado temporal.

Pipeline por punto clave y frame:
  1. Hipótesis DLT con cada par de cámaras que ven el punto (RANSAC exhaustivo).
  2. Se elige la hipótesis con más inliers (error de reproyección < umbral).
  3. DLT final con todos los inliers.
Después: relleno de huecos cortos e interpolación + filtro Butterworth de fase cero.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.signal import butter, filtfilt


def _dlt(Ps: np.ndarray, uv: np.ndarray, w: np.ndarray) -> np.ndarray:
    """DLT por lotes. Ps (C,3,4), uv (C,N,2), w (C,N) pesos -> X (N,3)."""
    u = np.nan_to_num(uv[..., 0])[..., None]
    v = np.nan_to_num(uv[..., 1])[..., None]
    rows_u = u * Ps[:, None, 2, :] - Ps[:, None, 0, :]  # (C,N,4)
    rows_v = v * Ps[:, None, 2, :] - Ps[:, None, 1, :]
    A = np.concatenate([rows_u, rows_v], axis=0)  # (2C,N,4)
    ww = np.concatenate([w, w], axis=0)[..., None]
    norm = np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12
    A = (A / norm) * ww
    A = np.transpose(A, (1, 0, 2))  # (N,2C,4)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        return Xh[:, :3] / Xh[:, 3:4]


def _reproj_err(Ps: np.ndarray, X: np.ndarray, uv: np.ndarray) -> np.ndarray:
    Xh = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    proj = np.einsum("cij,nj->cni", Ps, Xh)
    with np.errstate(divide="ignore", invalid="ignore"):
        p2 = proj[..., :2] / proj[..., 2:3]
        err = np.linalg.norm(p2 - uv, axis=-1)
    err[proj[..., 2] <= 0] = np.inf
    return err


def triangulate(Ps: np.ndarray, uv: np.ndarray, thr_px: float = 10.0, chunk: int = 60000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ps (C,3,4); uv (C,N,2) con NaN donde no hay detección.

    Devuelve X (N,3) (NaN si <2 vistas inlier), n_inliers (N,), err medio px (N,).
    """
    C, N, _ = uv.shape
    X_out = np.full((N, 3), np.nan)
    n_in = np.zeros(N, dtype=int)
    e_out = np.full(N, np.nan)
    pairs = list(combinations(range(C), 2))
    for s in range(0, N, chunk):
        sl = slice(s, min(N, s + chunk))
        uvc = uv[:, sl]
        valid = ~np.isnan(uvc[..., 0])
        n = uvc.shape[1]
        best_cnt = np.zeros(n, dtype=int)
        best_err = np.full(n, np.inf)
        best_mask = np.zeros((C, n), dtype=bool)
        for i, j in pairs:
            ok = valid[i] & valid[j]
            if not ok.any():
                continue
            sub = np.where(ok)[0]
            Xp = np.full((n, 3), np.nan)
            # hipótesis con sólo 2 cámaras: sistema 4x4 por punto
            Xp[sub] = _dlt(Ps[[i, j]], uvc[[i, j]][:, sub], np.ones((2, len(sub))))
            err = _reproj_err(Ps, np.nan_to_num(Xp), uvc)
            err[:, ~ok] = np.inf
            inl = valid & (err < thr_px)
            cnt = inl.sum(0)
            merr = np.where(inl, err, 0).sum(0) / np.maximum(cnt, 1)
            better = ok & ((cnt > best_cnt) | ((cnt == best_cnt) & (merr < best_err)))
            best_cnt = np.where(better, cnt, best_cnt)
            best_err = np.where(better, merr, best_err)
            best_mask[:, better] = inl[:, better]
        good = best_cnt >= 2
        Xf = _dlt(Ps, uvc, best_mask.astype(float))
        Xf[~good] = np.nan
        err = _reproj_err(Ps, np.nan_to_num(Xf), uvc)
        X_out[sl] = Xf
        n_in[sl] = np.where(good, best_cnt, 0)
        e_out[sl] = np.where(good, np.where(best_mask, err, 0).sum(0) / np.maximum(best_cnt, 1), np.nan)
    return X_out, n_in, e_out


def fill_and_smooth(X: np.ndarray, fps: float, cutoff_hz: float = 8.0, max_gap_s: float = 0.6) -> np.ndarray:
    """X (F,J,3) con NaN. Interpola huecos cortos y aplica filtro paso bajo de fase cero."""
    F, J, _ = X.shape
    out = X.copy()
    t = np.arange(F)
    max_gap = int(max_gap_s * fps)
    b, a = butter(4, cutoff_hz / (fps / 2))
    for j in range(J):
        for k in range(3):
            y = out[:, j, k]
            nan = np.isnan(y)
            if nan.all():
                continue
            if nan.any():
                y[nan] = np.interp(t[nan], t[~nan], y[~nan])
                # huecos demasiado largos se marcan como no fiables
                run = 0
                for f in range(F):
                    run = run + 1 if nan[f] else 0
                    if run > max_gap:
                        y[f - run + 1:f + 1] = np.nan
            seg_ok = ~np.isnan(y)
            if seg_ok.sum() > 30:
                yy = y.copy()
                yy[~seg_ok] = np.interp(t[~seg_ok], t[seg_ok], y[seg_ok])
                yy = filtfilt(b, a, yy)
                yy[~seg_ok] = np.nan
                y = yy
            out[:, j, k] = y
    return out

"""Generador sintético para el proof of concept (sin vídeo real).

1. simulate_ride(profile): cinemática 3D "verdad" de caballo + jinete realizando
   la prueba según un perfil de ejecución (protocol.DEMO_PROFILE incluye errores).
2. make_rig(): rig de N cámaras 4K alrededor de la pista.
3. observe(): proyección 2D con ruido, oclusiones y falsos positivos, imitando
   la salida de un detector de keypoints (RTMPose / DeepLabCut).

Esto permite ejercitar el pipeline real (calibración -> triangulación ->
métricas -> puntuación) de extremo a extremo y medir su error frente a la verdad.
"""
from __future__ import annotations

import numpy as np

from . import arena
from .camera import Camera, calibrate_extrinsics_planar, intrinsics, look_at
from .protocol import Segment, build_route
from .skeleton import HORSE_KP, RIDER_KP

# Parámetros de aire: frecuencia de tranco (Hz) a velocidad de referencia v0,
# fracción de apoyo, elevación del casco y desfase de inicio de apoyo por miembro.
GAIT_PARAMS = {
    "walk":     dict(v0=1.6, f0=0.95, duty=0.62, lift=0.12, bob=0.012, bob_k=2,
                     off={"LH": 0.0, "LF": 0.25, "RH": 0.5, "RF": 0.75}),
    "trot":     dict(v0=3.3, f0=1.38, duty=0.42, lift=0.20, bob=0.035, bob_k=2,
                     off={"LF": 0.0, "RH": 0.0, "RF": 0.5, "LH": 0.5}),
    "canter_R": dict(v0=4.0, f0=1.65, duty=0.38, lift=0.24, bob=0.05, bob_k=1,
                     off={"LH": 0.0, "RH": 0.30, "LF": 0.30, "RF": 0.45}),
    "canter_L": dict(v0=4.0, f0=1.65, duty=0.38, lift=0.24, bob=0.05, bob_k=1,
                     off={"RH": 0.0, "LH": 0.30, "RF": 0.30, "LF": 0.45}),
}
GAIT_SPEED = {"walk": 1.6, "trot": 3.3, "canter_L": 4.0, "canter_R": 4.0}
NECK_ELEV = {"halt": 34.0, "walk": 32.0, "trot": 38.0, "canter_L": 40.0, "canter_R": 40.0}
LEGS = ("LF", "RF", "LH", "RH")
LEG_UPPER = {"LF": (0.62, 0.20, 1.20), "RF": (0.62, -0.20, 1.20), "LH": (-0.62, 0.21, 1.30), "RH": (-0.62, -0.21, 1.30)}
LEG_HOOF0 = {"LF": (0.62, 0.15), "RF": (0.62, -0.15), "LH": (-0.60, 0.15), "RH": (-0.60, -0.15)}
LEG_LEN = {"F": (0.62, 0.45), "H": (0.75, 0.45)}
FETLOCK_H = 0.17


def family(g: str) -> str:
    return "canter" if g.startswith("canter") else g


def _lowpass_noise(rng, n, dt, tau, sigma):
    a = np.exp(-dt / tau)
    x = np.zeros(n)
    e = rng.normal(0, sigma * np.sqrt(1 - a * a), n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x


def _ema(x, dt, tau):
    a = 1 - np.exp(-dt / tau)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def _two_bone(A, C, l1, l2, hint):
    """IK de dos segmentos por lotes: A,C (N,3), hint (N,3) -> B (N,3)."""
    d_vec = C - A
    d = np.linalg.norm(d_vec, axis=1, keepdims=True)
    d = np.clip(d, 1e-6, l1 + l2 - 1e-4)
    u = d_vec / np.linalg.norm(d_vec, axis=1, keepdims=True)
    a = (l1 ** 2 - l2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(np.clip(l1 ** 2 - a ** 2, 0, None))
    perp = hint - np.sum(hint * u, axis=1, keepdims=True) * u
    perp /= np.linalg.norm(perp, axis=1, keepdims=True) + 1e-9
    return A + a * u + h * perp


# ---------------------------------------------------------------------------
def _dense_path(segs: list[Segment], ds=0.05):
    P, gait, mov, vt, tag = [], [], [], [], []
    halts = []  # (índice de muestra, duración, mov, tag)
    for sg in segs:
        if sg.gait == "halt":
            halts.append((len(P) - 1, sg.halt_s, sg.mov, sg.tag))
            continue
        pts = sg.pts
        seglen = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        n = max(2, int(seglen[-1] / ds))
        s = np.linspace(0, seglen[-1], n + 1)
        xy = np.column_stack([np.interp(s, seglen, pts[:, 0]), np.interp(s, seglen, pts[:, 1])])
        if P and np.linalg.norm(np.asarray(P[-1]) - xy[0]) < 1e-6:
            xy = xy[1:]
        for q in xy:
            P.append(q)
            gait.append(sg.gait)
            mov.append(sg.mov)
            vt.append(sg.speed if sg.speed else GAIT_SPEED[sg.gait])
            tag.append(sg.tag)
    P = np.asarray(P)
    # Suavizado (radio de giro realista en las esquinas)
    w = int(1.0 / ds)  # media móvil de 2 m: redondea esquinas sin encoger los círculos
    pad = np.pad(P, ((w, w), (0, 0)), mode="edge")
    ker = np.ones(2 * w + 1) / (2 * w + 1)
    Ps = np.column_stack([np.convolve(pad[:, k], ker, mode="valid") for k in range(2)])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Ps, axis=0), axis=1))]
    return Ps, s, np.array(gait), np.array(mov), np.array(vt, float), np.array(tag), halts


def simulate_ride(profile: dict, fps: float = 50.0, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    segs = build_route(profile)
    P, s_arr, g_arr, m_arr, vt_arr, tag_arr, halts = _dense_path(segs)
    n = len(s_arr)
    ds = np.diff(s_arr, append=s_arr[-1] + 1e-3)

    # Límite de velocidad por curvatura y por paradas (pasadas hacia atrás/adelante)
    tang = np.gradient(P, axis=0)
    head = np.unwrap(np.arctan2(tang[:, 1], tang[:, 0]))
    curv = np.abs(np.gradient(head) / np.maximum(np.gradient(s_arr), 1e-6))
    curv = np.convolve(curv, np.ones(21) / 21, mode="same")
    vlim = np.minimum(vt_arr, np.sqrt(3.5 / np.maximum(curv, 1e-3)))
    vlim = np.maximum(vlim, np.where(g_arr == "walk", 1.2, 1.9))
    halt_idx = {h[0]: h for h in halts}
    for i in halt_idx:
        vlim[i] = 0.0
    vlim[-1] = 0.0
    dec = 1.6
    for i in range(n - 2, -1, -1):
        vlim[i] = min(vlim[i], np.sqrt(vlim[i + 1] ** 2 + 2 * dec * ds[i]))

    # Integración temporal
    dt = 1.0 / fps
    acc = 1.8
    t_s, t_v, t_i, t_halt, t_htag = [], [], [], [], []
    s, v, i = 0.0, 0.0, 0
    pending = sorted(halt_idx)
    s_end = s_arr[-1]
    while True:
        k = pending[0] if pending else None
        if k is not None and s >= s_arr[k] - 1e-6:
            pending.pop(0)
            for _ in range(int(halt_idx[k][1] * fps)):
                t_s.append(s); t_v.append(0.0); t_i.append(i); t_halt.append(True); t_htag.append(halt_idx[k][3])
            v = 0.0
            i = min(k + 1, n - 1)  # reanudar desde la muestra siguiente al punto de parada
            continue
        if s >= s_end - 1e-6:
            break
        v = max(min(v + acc * dt, vlim[i]), 0.05)
        s = min(s + v * dt, s_arr[k] if k is not None else s_end)
        while i < n - 1 and s_arr[i + 1] <= s:
            i += 1
        t_s.append(s); t_v.append(v); t_i.append(i); t_halt.append(False); t_htag.append("")

    F = len(t_s)
    ti = np.array(t_i)
    sv = np.array(t_s)
    v = np.array(t_v)
    is_halt = np.array(t_halt)
    htag = np.array(t_htag)
    xy = np.column_stack([np.interp(sv, s_arr, P[:, 0]), np.interp(sv, s_arr, P[:, 1])])
    yaw = np.interp(sv, s_arr, head)
    gait = g_arr[ti].astype(object)
    gait_true = np.where(is_halt, "halt", gait).astype(object)
    mov = m_arr[ti]
    tag = tag_arr[ti]

    # --- Fases de los miembros -------------------------------------------
    jit = _lowpass_noise(rng, F, dt, 2.0, 0.025)
    phase = np.zeros(F)
    offs = {leg: np.zeros(F) for leg in LEGS}
    cur = {leg: GAIT_PARAMS[gait[0]]["off"][leg] for leg in LEGS}
    par_arr = {k: np.zeros(F) for k in ("duty", "lift", "bob", "bob_k", "f")}
    sm = {k: GAIT_PARAMS[gait[0]][k] for k in ("duty", "lift", "bob")}
    for f in range(F):
        gp = GAIT_PARAMS[gait[f]]
        a = 1 - np.exp(-dt / 0.25)
        for leg in LEGS:
            d = (gp["off"][leg] - cur[leg] + 0.5) % 1.0 - 0.5
            cur[leg] = (cur[leg] + a * d) % 1.0
            offs[leg][f] = cur[leg]
        for k in ("duty", "lift", "bob"):
            sm[k] += a * (gp[k] - sm[k])
            par_arr[k][f] = sm[k]
        par_arr["bob_k"][f] = gp["bob_k"]
        fr = gp["f0"] * np.clip(max(v[f], 0.3) / gp["v0"], 0.6, 1.6) ** 0.35 * (1 + jit[f])
        par_arr["f"][f] = fr
        if f:
            phase[f] = phase[f - 1] + fr * dt
    duty, lift, f_str = par_arr["duty"], par_arr["lift"], par_arr["f"]
    move_k = np.clip(v / 0.5, 0, 1)

    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    fwd = np.column_stack([cos_y, sin_y, np.zeros(F)])
    left = np.column_stack([-sin_y, cos_y, np.zeros(F)])
    up = np.tile([0, 0, 1.0], (F, 1))
    origin = np.column_stack([xy, np.zeros(F)])

    bob = par_arr["bob"] * move_k * np.cos(2 * np.pi * par_arr["bob_k"] * phase)
    dz = bob[:, None] * up

    def body(x, y, z):
        return origin + x * fwd + y * left + z * up

    H = np.zeros((F, len(HORSE_KP), 3))
    hk = {name: i for i, name in enumerate(HORSE_KP)}

    def put(name, arr):
        H[:, hk[name]] = arr

    withers = body(0.55, 0, 1.62) + dz
    put("withers", withers)
    put("chest", body(0.85, 0, 1.25) + dz)
    put("mid_back", body(0.0, 0, 1.55) + dz)
    put("croup", body(-0.65, 0, 1.60) + dz)
    tail_base = body(-0.82, 0, 1.50) + dz
    put("tail_base", tail_base)
    swish = _lowpass_noise(rng, F, dt, 0.4, 0.08)
    put("tail_tip", tail_base - 0.12 * fwd - 0.62 * up + swish[:, None] * left)

    # Cabeza y cuello: ángulo de elevación del cuello y del plano nasal
    fam = np.array([family(g) for g in gait_true])
    hnp_target = np.array([profile["hnp"].get(fm, profile["hnp"].get("walk")) for fm in fam], float)
    for mv, val in profile["hnp_override"].items():
        hnp_target[mov == mv] = val
    hnp = _ema(hnp_target, dt, 1.2) + _lowpass_noise(rng, F, dt, 0.6, 1.2)
    nod = np.where(fam == "walk", 2.5 * np.sin(4 * np.pi * phase), 0.0) * move_k
    hnp = hnp + nod
    elev = np.radians(_ema(np.array([NECK_ELEV[g] for g in gait_true]), dt, 0.8))
    poll = withers + 0.95 * (np.cos(elev)[:, None] * fwd + np.sin(elev)[:, None] * up)
    ph = np.radians(hnp)[:, None]
    nasal = -np.cos(ph) * up + np.sin(ph) * fwd
    face_fwd = np.cos(ph) * fwd + np.sin(ph) * up
    put("poll", poll)
    put("nose", poll + 0.58 * nasal)
    put("eye", poll + 0.16 * nasal + 0.07 * face_fwd)
    put("throat", poll + 0.14 * nasal - 0.18 * face_fwd)

    # Miembros
    final_halt = (htag == "final")
    unsq = _ema(np.where(final_halt, profile["halt13_unsquare"], 0.0), dt, 0.3)
    for leg in LEGS:
        lp = (phase - offs[leg]) % 1.0
        sweep = v * duty / f_str  # recorrido del casco durante el apoyo
        stance = lp < duty
        u_st = np.where(stance, lp / duty, 0)
        u_sw = np.where(~stance, (lp - duty) / (1 - duty), 0)
        hx = np.where(stance, sweep * (0.5 - u_st), sweep * (-0.5 + (1 - np.cos(np.pi * u_sw)) / 2))
        hz = np.where(stance, 0.0, lift * (1.15 if leg[1] == "F" else 1.0) * np.sin(np.pi * u_sw)) * move_k
        x0, y0 = LEG_HOOF0[leg]
        if leg == "RH":
            hx = hx - unsq
        hoof = body(0, 0, 0) + (x0 + hx)[:, None] * fwd + y0 * left + (hz + 0.03)[:, None] * up
        ux, uy, uz = LEG_UPPER[leg]
        upper = body(ux, uy, uz) + dz
        fet = hoof + FETLOCK_H * up + np.where(stance, 0, 0.06 * np.sin(np.pi * u_sw))[:, None] * (fwd if leg[1] == "H" else -fwd)
        l1, l2 = LEG_LEN[leg[1]]
        hint = fwd if leg[1] == "F" else -fwd
        mid = _two_bone(upper, fet, l1, l2, hint)
        names = (f"{leg}_shoulder", f"{leg}_knee") if leg[1] == "F" else (f"{leg}_hip", f"{leg}_hock")
        put(names[0], upper)
        put(names[1], mid)
        put(f"{leg}_fetlock", fet)
        put(f"{leg}_hoof", hoof)

    # --- Jinete -------------------------------------------------------------
    Rr = np.zeros((F, len(RIDER_KP), 3))
    rk = {name: i for i, name in enumerate(RIDER_KP)}
    lean_t = np.where(fam == "canter", profile["lean_canter"], profile["lean_other"]).astype(float)
    lean = np.radians(_ema(lean_t, dt, 0.8) + _lowpass_noise(rng, F, dt, 0.5, 1.0))
    bounce_t = np.zeros(F)
    for mv, amp in profile["bounce_movs"].items():
        bounce_t[(mov == mv) & (fam == "trot")] = amp
    bounce = _ema(bounce_t, dt, 0.5) * move_k * (0.5 + 0.5 * np.cos(4 * np.pi * phase + 0.9))
    rdz = (bob * 0.9 + bounce)[:, None] * up
    pelvis = body(0.05, 0, 1.74) + rdz
    trunk = np.sin(lean)[:, None] * fwd + np.cos(lean)[:, None] * up
    neck = pelvis + 0.55 * trunk
    Rr[:, rk["neck"]] = neck
    Rr[:, rk["head"]] = pelvis + 0.80 * trunk
    for side, sg in (("L", 1), ("R", -1)):
        hip = pelvis + sg * 0.16 * left
        sh = neck - 0.05 * trunk + sg * 0.19 * left
        knee = hip + 0.24 * fwd + sg * 0.12 * left - 0.32 * up
        ankle = knee - 0.05 * fwd + sg * 0.02 * left - 0.42 * up
        Rr[:, rk[f"{side}_hip"]] = hip
        Rr[:, rk[f"{side}_shoulder"]] = sh
        Rr[:, rk[f"{side}_knee"]] = knee
        Rr[:, rk[f"{side}_ankle"]] = ankle
    hj = profile["hand_jitter"]
    jit_h = np.column_stack([_lowpass_noise(rng, F, dt, 0.25, hj) for _ in range(3)])
    l_hand = body(0.47, 0.03, 1.93) + 0.25 * np.sin(lean)[:, None] * fwd + rdz * 0.7 \
        + jit_h[:, :1] * fwd + jit_h[:, 1:2] * left + jit_h[:, 2:3] * up
    r_rest = Rr[:, rk["R_hip"]] + 0.14 * fwd - 0.07 * left + 0.06 * up
    # Caricias (mano derecha al cuello por delante de las riendas)
    w = np.zeros(F)
    tt = np.arange(F) * dt
    for mv, frac, dur in profile["caress"]:
        idx = np.where(mov == mv)[0]
        if len(idx) == 0:
            continue
        t0 = tt[idx[0]] + frac * (tt[idx[-1]] - tt[idx[0]])
        w = np.maximum(w, np.clip(np.minimum((tt - t0) / 0.3, (t0 + dur - tt) / 0.3), 0, 1))
    crest = withers + 0.38 * (poll - withers) + 0.06 * up - 0.06 * left
    r_hand = r_rest * (1 - w[:, None]) + crest * w[:, None]
    Rr[:, rk["L_hand"]] = l_hand
    Rr[:, rk["R_hand"]] = r_hand
    for side, hand in (("L", l_hand), ("R", r_hand)):
        sh = Rr[:, rk[f"{side}_shoulder"]]
        sg = 1 if side == "L" else -1
        Rr[:, rk[f"{side}_elbow"]] = _two_bone(sh, hand, 0.30, 0.30, -up - 0.3 * fwd + sg * 0.4 * left)

    # Pequeña variabilidad anatómica / de marcado
    H += rng.normal(0, 0.004, H.shape)
    Rr += rng.normal(0, 0.004, Rr.shape)

    return {
        "fps": fps, "t": tt, "horse": H, "rider": Rr,
        "gt": {"gait": gait_true, "mov": mov, "speed": v, "yaw": yaw, "hnp": hnp,
               "lean": np.degrees(lean), "caress": w > 0.5, "tag": tag},
    }


# ---------------------------------------------------------------------------
def make_rig(n_cams: int = 6, f_px: float = 2300.0, width: int = 3840, height: int = 2160, fps: float = 50.0) -> list[Camera]:
    K = intrinsics(f_px, width, height)
    placements = [
        ("CAM1 esquina A-F", (-9.0, -15.0, 5.5)), ("CAM2 esquina A-K", (-9.0, 15.0, 5.5)),
        ("CAM3 esquina C-M", (49.0, -15.0, 5.5)), ("CAM4 esquina C-H", (49.0, 15.0, 5.5)),
        ("CAM5 lateral B", (20.0, -23.0, 6.5)), ("CAM6 lateral E", (20.0, 23.0, 6.5)),
        ("CAM7 lateral A", (-14.0, 0.0, 6.0)), ("CAM8 lateral C", (54.0, 0.0, 6.0)),
    ][:n_cams]
    return [look_at(name, eye, (20.0, 0.0, 0.5), K, width, height, fps) for name, eye in placements]


def calibrate_rig(true_rig: list[Camera], rng, px_noise: float = 0.8) -> tuple[list[Camera], list[dict]]:
    """Simula la calibración extrínseca: detectar letras/esquinas en la imagen y resolver la pose."""
    names, W = arena.control_points_3d()
    est, report = [], []
    for cam in true_rig:
        uv, vis = cam.project(W)
        uv = uv + rng.normal(0, px_noise, uv.shape)
        R, t, rmse = calibrate_extrinsics_planar(cam.K, W[vis], uv[vis])
        c = Camera(cam.name, cam.K, R, t, cam.width, cam.height, cam.fps)
        pos_err = float(np.linalg.norm(c.center - cam.center))
        c.meta = {"calib_points": int(vis.sum()), "calib_rmse_px": round(rmse, 2), "pos_err_cm": round(pos_err * 100, 1)}
        est.append(c)
        report.append({"name": cam.name, **c.meta})
    return est, report


def observe(rig: list[Camera], X: np.ndarray, side_of: np.ndarray, yaw: np.ndarray, rng, px_sigma: float = 1.6,
            p_drop: float = 0.05, p_far: float = 0.35, p_outlier: float = 0.015) -> np.ndarray:
    """Simula detecciones 2D de un detector de keypoints.

    X (F,J,3); side_of (J,) +1 izquierdo / -1 derecho / 0 línea media; yaw (F,) rumbo del caballo.
    Devuelve uv (C, F, J, 2) con NaN donde no hay detección.
    """
    F, J, _ = X.shape
    out = np.full((len(rig), F, J, 2), np.nan)
    left_xy = np.column_stack([-np.sin(yaw), np.cos(yaw)])  # (F,2)
    for c, cam in enumerate(rig):
        uv, vis = cam.project(X)
        uv = uv + rng.normal(0, px_sigma, uv.shape)
        keep = vis & (rng.random((F, J)) > p_drop)
        # Oclusión del lado lejano: el cuerpo tapa los miembros del lado opuesto a la cámara
        cam_dir = cam.center[None, None, :2] - X[..., :2]
        same_side = np.einsum("fjk,fk->fj", cam_dir, left_xy) * side_of[None, :]
        far = (side_of[None, :] != 0) & (same_side < 0)
        keep &= ~(far & (rng.random((F, J)) < p_far))
        # Falsos positivos (p.ej. confusión de miembros del detector)
        outl = rng.random((F, J)) < p_outlier
        uv = uv + outl[..., None] * rng.normal(0, 80, uv.shape)
        uv[~keep] = np.nan
        out[c] = uv
    return out

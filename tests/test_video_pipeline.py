"""Tests del camino de vídeo real: calibración OpenCV, distorsión, sincronización,
traducción de esqueletos y la sesión completa (vídeos -> 3D -> nota) con un detector
simulado que devuelve la proyección de la verdad sintética.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = pytest.importorskip("cv2")

from roca import arena  # noqa: E402
from roca.calibration import (BoardSpec, board_image, calibrate_extrinsics, calibrate_intrinsics,  # noqa: E402
                              load_camera, save_camera, undistort_points)
from roca.camera import Camera  # noqa: E402
from roca.detectors.base import apply_mapping  # noqa: E402
from roca.detectors.rider import COCO17, COCO_TO_ROCA, select_rider  # noqa: E402
from roca.detectors.horse import SA_QUADRUPED, SA_TO_ROCA  # noqa: E402
from roca.pipeline import side_vector  # noqa: E402
from roca.protocol import DEMO_PROFILE  # noqa: E402
from roca.report import build_report  # noqa: E402
from roca.scoring import score  # noqa: E402
from roca.analysis import analyze  # noqa: E402
from roca.session import frame_plan, load_session, run_session  # noqa: E402
from roca.skeleton import HORSE_KP, RIDER_KP, hidx  # noqa: E402
from roca.synth import make_rig, simulate_ride  # noqa: E402
from roca.video import estimate_offsets, probe  # noqa: E402

DIST = np.array([-0.12, 0.03, 0.0005, -0.0004, 0.0])


def _rt(rx, ry, tz, tx=0.0, ty=0.0):
    R, _ = cv2.Rodrigues(np.array([rx, ry, 0.0]))
    return R, np.array([tx, ty, tz])


def test_charuco_intrinsics_recovers_focal():
    spec = BoardSpec(7, 5, 0.08, 0.06)
    pps, margin = 120, 30
    tex = board_image(spec, pps, margin)
    W, H, f = 1280, 720, 1000.0
    K = np.array([[f, 0, 640], [0, f, 360], [0, 0, 1.0]])
    s = spec.square_m / pps
    T = np.array([[s, 0, -margin * s], [0, s, -margin * s], [0, 0, 1.0]])
    views = []
    for rx, ry in [(0.3, 0), (-0.3, 0), (0, 0.35), (0, -0.35), (0.25, 0.25), (-0.2, 0.3), (0.35, -0.2), (0.1, 0.1)]:
        R, t = _rt(rx, ry, 0.9, -0.28, -0.2)
        Hm = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ T
        views.append(cv2.warpPerspective(tex, Hm, (W, H), borderValue=255))
    intr = calibrate_intrinsics(views, spec)
    assert intr["n_views"] >= 6
    assert intr["rms_px"] < 1.0
    assert abs(intr["K"][0, 0] - f) / f < 0.02 and abs(intr["K"][1, 1] - f) / f < 0.02


def test_undistort_roundtrip():
    K = np.array([[1800, 0, 1920], [0, 1800, 1080], [0, 0, 1.0]])
    rng = np.random.default_rng(0)
    Xc = np.column_stack([rng.uniform(-1, 1, 200), rng.uniform(-0.6, 0.6, 200), np.full(200, 2.0)])
    ideal = (Xc @ K.T)[:, :2] / 2.0
    dist, _ = cv2.projectPoints(Xc, np.zeros(3), np.zeros(3), K, DIST)
    dist = dist.reshape(-1, 2)
    dist[5] = np.nan
    und = undistort_points(dist, K, DIST)
    ok = ~np.isnan(und[:, 0])
    assert not ok[5] and ok.sum() == 199
    assert np.abs(und[ok] - ideal[ok]).max() < 0.05


def test_extrinsics_from_letters_with_distortion(tmp_path):
    names, Wp = arena.control_points_3d()
    for cam in make_rig(4):
        rv, _ = cv2.Rodrigues(cam.R)
        uv, _ = cv2.projectPoints(Wp, rv, cam.t, cam.K, DIST)
        uv = uv.reshape(-1, 2)
        _, vis = cam.project(Wp)
        pts = {n: uv[i].tolist() for i, n in enumerate(names) if vis[i]}
        R, t, rmse, n = calibrate_extrinsics(cam.K, DIST, pts)
        assert np.linalg.norm(-R.T @ t - cam.center) < 0.02 and rmse < 0.5
        c = Camera(cam.name, cam.K, R, t, cam.width, cam.height, 50.0, {"calib_points": n})
        save_camera(tmp_path / "c.json", c, DIST)
        c2, D2 = load_camera(tmp_path / "c.json")
        assert np.allclose(c2.P, c.P) and np.allclose(D2, DIST)


def test_flash_sync_offsets():
    rng = np.random.default_rng(3)
    base = 80 + rng.normal(0, 0.5, 400)
    sigs = []
    for flash in (100, 104, 97):
        s = base[:300] + rng.normal(0, 0.5, 300)
        s[flash:flash + 3] += 120
        sigs.append(s)
    assert estimate_offsets(sigs, max_lag=20) == [0, 4, -3]


def test_frame_plan_overlap():
    base, per = frame_plan([100, 100, 100], [0, 5, -2], stride=2)
    assert base[0] == 2 and base[-1] < 95
    assert (per[1] == base + 5).all() and (per[2] == base - 2).all()


def test_skeleton_mappings_are_complete():
    # Jinete: los 14 puntos de ROCA salen de COCO-17
    kp = np.random.default_rng(0).uniform(0, 100, (17, 2))
    out, c = apply_mapping(COCO17, kp, np.ones(17), COCO_TO_ROCA, RIDER_KP)
    assert not np.isnan(out).any() and c.min() == 1
    assert np.allclose(out[RIDER_KP.index("neck")], kp[[5, 6]].mean(0))
    # Caballo SuperAnimal: todos los usados por las métricas existen (sólo faltan menudillos)
    out, _ = apply_mapping(SA_QUADRUPED, np.zeros((39, 2)), np.ones(39), SA_TO_ROCA, HORSE_KP)
    missing = [n for n, p in zip(HORSE_KP, out) if np.isnan(p[0])]
    assert sorted(missing) == sorted(f"{leg}_fetlock" for leg in ("LF", "RF", "LH", "RH"))


def test_select_rider_prefers_person_on_horse():
    people = np.zeros((2, 17, 2))
    people[0, :, :] = [100, 400]   # persona de pie lejos
    people[1, :, :] = [520, 290]   # sobre la cruz
    horse = np.full((26, 2), np.nan)
    horse[:, :] = [600, 400]
    horse[hidx("withers")] = [500, 300]
    horse[hidx("LF_hoof")] = [500, 520]
    assert select_rider(people, np.ones((2, 17)), horse, np.ones(26)) == 1


# ----------------------------------------------------------------------------- sesión completa

class ProjectedDetector:
    """Detector simulado: proyección (con distorsión) de la verdad + ruido y huecos."""

    def __init__(self, names, uv_by_video, seed):
        self.names, self.uv, self.rng = names, uv_by_video, np.random.default_rng(seed)

    def detect_video(self, path, frames, context=None):
        uv = self.uv[str(Path(path).resolve())]
        t = np.asarray(frames)
        kp = uv[t] + self.rng.normal(0, 1.0, uv[t].shape)
        conf = np.where(np.isnan(kp[..., 0]), 0.0, 0.9)
        conf[self.rng.random(conf.shape) < 0.03] = 0.1  # detecciones dudosas
        return kp, conf


@pytest.fixture(scope="module")
def session_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("sesion")
    gt = simulate_ride(DEMO_PROFILE)
    X = np.concatenate([gt["horse"], gt["rider"]], axis=1)
    F = X.shape[0]
    rig = make_rig(4)
    offsets = [0, 7, -4, 12]
    lead = 60  # frames antes del inicio, con el flash
    cams_cfg, proj = [], {}
    w, h = 64, 36  # vídeo diminuto: el detector simulado no mira los píxeles
    nh = len(HORSE_KP)
    side = side_vector()
    for cam, off in zip(rig, offsets):
        name = cam.name.split()[0]
        n_frames = F + lead + 30
        video = d / f"{name}.avi"
        wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 50.0, (w, h))
        dark, bright = np.full((h, w, 3), 40, np.uint8), np.full((h, w, 3), 230, np.uint8)
        flash_at = lead // 2 + off
        for i in range(n_frames):
            wr.write(bright if flash_at <= i < flash_at + 3 else dark)
        wr.release()
        rv, _ = cv2.Rodrigues(cam.R)
        uv, _ = cv2.projectPoints(X.reshape(-1, 3), rv, cam.t, cam.K, DIST)
        uv = uv.reshape(F, -1, 2)
        _, vis = cam.project(X)
        # oclusión del lado lejano, como en el PoC
        left = np.column_stack([-np.sin(gt["gt"]["yaw"]), np.cos(gt["gt"]["yaw"])])
        same = np.einsum("fjk,fk->fj", cam.center[None, None, :2] - X[..., :2], left) * side[None]
        vis &= ~((side[None] != 0) & (same < 0) & (np.random.default_rng(1).random(vis.shape) < 0.35))
        uv[~vis] = np.nan
        full = np.full((n_frames, uv.shape[1], 2), np.nan)
        full[lead + off: lead + off + F] = uv[: n_frames - lead - off]
        proj[str(video.resolve())] = full
        calib = Camera(name, cam.K, cam.R, cam.t, w, h, 50.0, {"calib_points": 15, "calib_rmse_px": 0.5})
        save_camera(d / f"{name}.json", calib, DIST)
        cams_cfg.append({"name": name, "video": video.name, "calibration": f"{name}.json"})
    cfg = {"cameras": cams_cfg, "sync": {"method": "flash", "search_s": 1.2, "max_lag_s": 0.5},
           "start_frame": lead, "stop_frame": lead + F, "min_conf": 0.4,
           "detectors": {"horse": {"type": "test"}, "rider": {"type": "test"}}, "work_dir": "work"}
    (d / "session.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d, gt, proj, offsets, nh


def test_session_end_to_end(session_dir):
    d, gt, proj, offsets, nh = session_dir
    cfg = load_session(d / "session.json")
    # `proj` está indexado por frame de vídeo de cada cámara (ya incluye su desfase)
    dets = {"horse": ProjectedDetector(HORSE_KP, {k: uv[:, :nh] for k, uv in proj.items()}, 1),
            "rider": ProjectedDetector(RIDER_KP, {k: uv[:, nh:] for k, uv in proj.items()}, 2)}
    logs = []
    out = run_session(cfg, detectors=dets, log=logs.append)
    assert any("desfases" in m and "+7" in m and "-4" in m and "+12" in m for m in logs), logs

    X_gt = np.concatenate([gt["horse"], gt["rider"]], axis=1)
    err = np.linalg.norm(out["X"] - X_gt, axis=-1)
    assert np.nanmedian(err) < 0.03, np.nanmedian(err)
    q = out["validation"]
    assert q["real"] and q["raw_coverage_pct"] > 90 and q["reproj_px"] < 3

    truth = score(analyze(gt["horse"], gt["rider"], gt["fps"]))
    assert abs(out["score"]["percent"] - truth["percent"]) < 2.0

    rep = build_report(out, {"synthetic": False, "rider": "R", "horse": "H"})
    assert rep["meta"]["synthetic"] is False and rep["validation"]["real"]
    # Segunda ejecución: usa la caché de detecciones (los detectores no se llaman)
    out2 = run_session(cfg, detectors={"horse": None, "rider": None}, log=lambda m: None)
    assert np.allclose(np.nan_to_num(out2["X"]), np.nan_to_num(out["X"]))

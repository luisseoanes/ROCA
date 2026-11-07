"""Tests del acceso a PFERD con un mini-dataset sintético con la misma estructura y convenciones
(npz sin pickle, prefijo IDn_ en los marcadores, kp2d = cv2.projectPoints(mocap3d, R, T, K, D))."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = pytest.importorskip("cv2")

from roca.detectors import NoRider, build  # noqa: E402
from roca.pferd import (DEFAULT_MAPPING, GroundTruthHorse, find_trials, load_kp2d, load_mapping,  # noqa: E402
                        normalize_label, to_roca, write_session)
from roca.session import load_session, reconstruct  # noqa: E402
from roca.skeleton import HORSE_KP  # noqa: E402

TRIAL = "20201128_ID_9_0001"
SERIALS = ["11111", "22222", "33333", "44444"]


def _marker_names(mapping):
    names = set()
    for v in mapping.values():
        if isinstance(v, str):
            names.add(v)
        elif isinstance(v, dict):
            names.update(v["mean"])
    return sorted(names)


def _fake_dataset(root: Path, F: int = 60, W: int = 1920, H: int = 1088):
    mapping = load_mapping()
    labels = _marker_names(mapping) + ["C_Belly_1_49"]
    rng = np.random.default_rng(0)
    base = rng.uniform([-1.2, -0.3, 0.0], [1.2, 0.3, 1.9], (len(labels), 3))
    t = np.arange(F) / 30.0
    mocap = base[None] + np.stack([0.8 * t, 0 * t, 0.02 * np.sin(6 * t)], -1)[:, None]
    mocap[5, 3] = np.nan  # un marcador perdido
    subj = root / "ID_9"
    K = np.array([[1600.0, 0, W / 2], [0, 1600.0, H / 2], [0, 0, 1]])
    D = np.array([-0.04, 0.13, 0.0006, 0.0002, 0.0])
    for k, s in enumerate(SERIALS):
        ang = k * np.pi / 2 + 0.3
        C = np.array([9 * np.cos(ang), 9 * np.sin(ang), 2.0])
        z = -C / np.linalg.norm(C)
        x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        T = -R @ C
        (subj / "CAM_DATA").mkdir(parents=True, exist_ok=True)
        np.savez(subj / "CAM_DATA" / f"Camera_Miqus_Video_{s}.npz", R=R, T=T[None], K=K, D=D, T_ori=T[None] * 1000)
        uv, _ = cv2.projectPoints(np.nan_to_num(mocap.reshape(-1, 3)), cv2.Rodrigues(R)[0], T, K, D)
        uv = uv.reshape(F, len(labels), 2)
        uv[np.isnan(mocap[..., 0])] = np.nan
        cam = f"Miqus_{40 + k}_{s}"
        kd = subj / "KP2D_DATA" / TRIAL
        kd.mkdir(parents=True, exist_ok=True)
        np.savez(kd / f"{TRIAL}_{cam}_2Dkp.npz", kp2d=uv, mocap3d=mocap,
                 labels=np.array([f"ID9_{n}" for n in labels]), videoFps=np.int64(30), videoFrameNum=np.int64(F))
        vd = subj / "VIDEO_DATA" / TRIAL
        vd.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(vd / f"{TRIAL}_{cam}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (W, H))
        for _ in range(F):
            vw.write(np.zeros((H, W, 3), np.uint8))
        vw.release()
    return mocap, labels


def test_mapping_file_covers_roca_skeleton():
    m = load_mapping(DEFAULT_MAPPING)
    assert set(m) == set(HORSE_KP)
    assert m["tail_tip"] is None  # PFERD no tiene marcador en la punta de la cola
    assert all(not n.startswith("ID") for n in _marker_names(m))


def test_normalize_label():
    assert normalize_label("ID1_L_HockJoint_30") == "L_HockJoint_30"
    assert normalize_label("ID12_C_Back_1_10") == "C_Back_1_10"
    assert normalize_label("C_Back_1_10") == "C_Back_1_10"


def test_load_and_map(tmp_path):
    mocap, labels = _fake_dataset(tmp_path)
    trials = find_trials(tmp_path)
    assert len(trials) == 4 and {t["serial"] for t in trials} == set(SERIALS)
    assert all(t["video"].exists() and t["calib"].exists() for t in trials)
    d = load_kp2d(trials[0]["kp2d"])
    assert d["labels"][0] == labels[0] and d["kp"].shape == (60, len(labels), 2)
    G = to_roca(d["labels"], d["mocap"])
    assert G.shape == (60, len(HORSE_KP), 3)
    assert np.isnan(G[:, HORSE_KP.index("tail_tip")]).all()
    nose = load_mapping()["nose"]["mean"]
    np.testing.assert_allclose(G[0, 0], mocap[0, [labels.index(n) for n in nose]].mean(0))


def test_session_reconstructs_mocap(tmp_path):
    """Calibración PFERD + triangulación de ROCA recuperan el mocap con 2D exacto."""
    _fake_dataset(tmp_path)
    sp = write_session(tmp_path, "ID_9", TRIAL, tmp_path / "sesion")
    cfg = load_session(sp)
    assert cfg["detectors"]["rider"] == {"type": "none"}
    assert isinstance(build("rider", {"type": "none"}, tmp_path), NoRider)
    kp2d = {c["video"]: c["kp2d"] for c in json.loads(sp.read_text())["cameras"]}
    rec = reconstruct(cfg, {"horse": GroundTruthHorse(kp2d), "rider": NoRider()}, log=lambda *a: None)
    d = load_kp2d(next(iter(kp2d.values())))
    G = to_roca(d["labels"], d["mocap"])[rec["frames"]]
    e = np.linalg.norm(rec["X"][:, :len(HORSE_KP)] - G, axis=-1)
    assert np.nanmean(e) < 0.01  # < 1 cm (sólo el suavizado)
    assert np.isnan(rec["X"][:, len(HORSE_KP):]).all()  # sin jinete

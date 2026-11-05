"""Tests del núcleo: geometría de cámaras, triangulación, aires y puntuación.

Ejecutar: python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cohnetic import arena  # noqa: E402
from cohnetic.analysis import analyze  # noqa: E402
from cohnetic.camera import calibrate_extrinsics_planar  # noqa: E402
from cohnetic.metrics.geometry import fit_circle  # noqa: E402
from cohnetic.protocol import DEMO_PROFILE, IDEAL_PROFILE  # noqa: E402
from cohnetic.scoring import BASE, score  # noqa: E402
from cohnetic.synth import make_rig, simulate_ride  # noqa: E402
from cohnetic.triangulation import triangulate  # noqa: E402


@pytest.fixture(scope="module")
def demo():
    gt = simulate_ride(DEMO_PROFILE)
    return gt, analyze(gt["horse"], gt["rider"], gt["fps"])


@pytest.fixture(scope="module")
def ideal():
    gt = simulate_ride(IDEAL_PROFILE)
    return gt, analyze(gt["horse"], gt["rider"], gt["fps"])


def test_calibration_from_letters_recovers_pose():
    rng = np.random.default_rng(0)
    _, W = arena.control_points_3d()
    for cam in make_rig(6):
        uv, vis = cam.project(W)
        R, t, rmse = calibrate_extrinsics_planar(cam.K, W[vis], uv[vis] + rng.normal(0, 0.5, uv[vis].shape))
        center = -R.T @ t
        assert np.linalg.norm(center - cam.center) < 0.05  # < 5 cm
        assert rmse < 1.5


def test_triangulation_rejects_outliers():
    rng = np.random.default_rng(1)
    rig = make_rig(6)
    X = np.column_stack([rng.uniform(2, 38, 500), rng.uniform(-9, 9, 500), rng.uniform(0, 2.5, 500)])
    uv = np.stack([c.project(X)[0] for c in rig]) + rng.normal(0, 1.0, (6, 500, 2))
    uv[0, :50] += 150  # cámara 0 con falsos positivos groseros en 50 puntos
    uv[3, 100:150] = np.nan  # detecciones ausentes
    Ps = np.stack([c.P for c in rig])
    Xh, n_in, _ = triangulate(Ps, uv)
    err = np.linalg.norm(Xh - X, axis=1)
    assert np.nanmedian(err) < 0.03
    assert np.nanmax(err[:50]) < 0.10  # el outlier se descarta
    assert (n_in[:50] <= 5).all()


def test_fit_circle():
    a = np.linspace(0, 2 * np.pi, 200)
    c, r, rms = fit_circle(np.column_stack([3 + 5 * np.cos(a), -1 + 5 * np.sin(a)]))
    assert np.allclose(c, [3, -1], atol=1e-6) and abs(r - 5) < 1e-6 and rms < 1e-6


def test_gait_detection_accuracy(demo):
    gt, an = demo
    lab, g = an["series"]["labels"], gt["gt"]["gait"]
    moving = g != "halt"
    assert np.mean(lab[moving] == g[moving]) > 0.92


def test_ideal_ride_has_no_deductions(ideal):
    _, an = ideal
    sc = score(an)
    assert all(m["note"] == BASE for m in sc["movements"]), [(m["n"], m["deductions"]) for m in sc["movements"] if m["deductions"]]
    assert all(c["note"] == BASE for c in sc["collective"])
    assert not sc["penalties"]


def test_demo_faults_are_detected(demo):
    _, an = demo
    sc = score(an)
    by_n = {m["n"]: m for m in sc["movements"]}
    reasons = lambda n: " ".join(d["reason"] for d in by_n[n]["deductions"])  # noqa: E731
    assert "Círculo de 11." in reasons(3)              # círculo de 10 m demasiado grande
    assert "tarde en M" in reasons(5)                    # transición al paso tardía
    assert by_n[8]["note"] <= 5 and "Rompe el aire" in reasons(8)  # ruptura del galope -> firewall
    assert "Alargamiento insuficiente" in reasons(12)
    assert "Inmovilidad insuficiente" in reasons(13) and "no cuadrada" in reasons(13)
    assert len(sc["penalties"]) == 1                     # una caricia por delante de las riendas
    col = {c["key"]: c for c in sc["collective"]}
    assert any("nariz" in d["reason"] for d in col["C"]["deductions"])
    assert any("Tronco" in d["reason"] for d in col["D"]["deductions"])

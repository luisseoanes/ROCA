"""Modelo de cámara pinhole y calibración extrínseca con las letras de la pista.

Convención: x_cam = R @ X_world + t ; píxel = K @ x_cam (sin distorsión; en el
sistema real se corrige la distorsión antes con los coeficientes intrínsecos).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass
class Camera:
    name: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int = 3840
    height: int = 2160
    fps: float = 50.0
    meta: dict = field(default_factory=dict)

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def to_cam(self, X: np.ndarray) -> np.ndarray:
        return X @ self.R.T + self.t

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Proyecta puntos (..., 3) -> píxeles (..., 2) y máscara de visibilidad."""
        Xc = self.to_cam(X)
        z = Xc[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uvw = Xc @ self.K.T
            uv = uvw[..., :2] / uvw[..., 2:3]
        vis = (z > 0.1) & (uv[..., 0] >= 0) & (uv[..., 0] < self.width) & (uv[..., 1] >= 0) & (uv[..., 1] < self.height)
        return uv, vis

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "K": self.K.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "center": self.center.tolist(),
            "width": self.width,
            "height": self.height,
            **self.meta,
        }


def intrinsics(f_px: float, width: int, height: int) -> np.ndarray:
    return np.array([[f_px, 0, width / 2], [0, f_px, height / 2], [0, 0, 1.0]])


def look_at(name: str, eye, target, K: np.ndarray, width: int, height: int, fps: float = 50.0) -> Camera:
    """Cámara situada en `eye` apuntando a `target` (z arriba en el mundo)."""
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])  # filas: ejes x, y, z de la cámara
    t = -R @ eye
    return Camera(name, K, R, t, width, height, fps)


def _homography(world_xy: np.ndarray, img: np.ndarray) -> np.ndarray:
    A = []
    for (X, Y), (u, v) in zip(world_xy, img):
        A.append([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u])
        A.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    return Vt[-1].reshape(3, 3)


def calibrate_extrinsics_planar(K: np.ndarray, world_pts: np.ndarray, img_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Estima R, t a partir de puntos del suelo (z=0) visibles: letras y esquinas.

    1) Homografía suelo->imagen y descomposición H = K [r1 r2 t].
    2) Refinamiento no lineal minimizando el error de reproyección.
    Devuelve (R, t, rmse_px).
    """
    H = _homography(world_pts[:, :2], img_pts)
    M = np.linalg.inv(K) @ H
    lam = 1.0 / np.linalg.norm(M[:, 0])
    if (lam * M[:, 2])[2] < 0:  # la pista debe quedar delante de la cámara
        lam = -lam
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    R0 = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R0)
    R0 = U @ Vt
    if np.linalg.det(R0) < 0:
        R0 = -R0

    def residual(p):
        R = Rotation.from_rotvec(p[:3]).as_matrix()
        Xc = world_pts @ R.T + p[3:]
        uv = Xc @ K.T
        return (uv[:, :2] / uv[:, 2:3] - img_pts).ravel()

    p0 = np.concatenate([Rotation.from_matrix(R0).as_rotvec(), t])
    sol = least_squares(residual, p0, method="lm")
    R = Rotation.from_rotvec(sol.x[:3]).as_matrix()
    rmse = float(np.sqrt(np.mean(sol.fun.reshape(-1, 2) ** 2) * 2))
    return R, sol.x[3:], rmse

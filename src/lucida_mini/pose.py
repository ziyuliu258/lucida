from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _rotation(axis: str, degrees: float) -> FloatArray:
    angle = np.deg2rad(degrees)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"unknown axis: {axis}")


def rotation_zxy(delta_zxy_deg: FloatArray) -> FloatArray:
    """Return the intrinsic local-frame Z-X-Y rotation used by Lucida Eq. (5)."""
    dz, dx, dy = np.asarray(delta_zxy_deg, dtype=np.float64)
    return _rotation("z", dz) @ _rotation("x", dx) @ _rotation("y", dy)


@dataclass(frozen=True)
class Pose:
    position: FloatArray
    rotation: FloatArray
    size: FloatArray

    def __post_init__(self) -> None:
        p = np.asarray(self.position, dtype=np.float64)
        r = np.asarray(self.rotation, dtype=np.float64)
        s = np.asarray(self.size, dtype=np.float64)
        if p.shape != (3,) or r.shape != (3, 3) or s.shape != (3,):
            raise ValueError("pose shapes must be position=(3,), rotation=(3,3), size=(3,)")
        if not (np.isfinite(p).all() and np.isfinite(r).all() and np.isfinite(s).all()):
            raise ValueError("pose values must be finite")
        if np.any(s <= 0):
            raise ValueError("pose size must be strictly positive")
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(r), 1.0, atol=1e-6
        ):
            raise ValueError("rotation must be a right-handed orthonormal matrix")
        object.__setattr__(self, "position", p)
        object.__setattr__(self, "rotation", r)
        object.__setattr__(self, "size", s)

    @classmethod
    def identity(cls, size: FloatArray | None = None) -> "Pose":
        return cls(
            position=np.zeros(3),
            rotation=np.eye(3),
            size=np.ones(3) if size is None else np.asarray(size, dtype=np.float64),
        )

    def update(
        self,
        *,
        rotate_zxy_deg: FloatArray | None = None,
        translate_local_fraction: FloatArray | None = None,
        scale_fraction: FloatArray | None = None,
    ) -> "Pose":
        delta_r = np.zeros(3) if rotate_zxy_deg is None else np.asarray(rotate_zxy_deg)
        delta_p = (
            np.zeros(3)
            if translate_local_fraction is None
            else np.asarray(translate_local_fraction)
        )
        delta_s = np.zeros(3) if scale_fraction is None else np.asarray(scale_fraction)
        for name, value in (("rotation", delta_r), ("translation", delta_p), ("scale", delta_s)):
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} delta must contain three finite values")

        next_rotation = self.rotation @ rotation_zxy(delta_r)
        next_size = self.size * (1.0 + delta_s)
        if np.any(next_size <= 0):
            raise ValueError("scale update must keep every size component positive")
        next_position = self.position + next_rotation @ (delta_p * self.size)
        return Pose(next_position, next_rotation, next_size)

    def permute(self, matrix: FloatArray) -> "Pose":
        matrix = np.asarray(matrix, dtype=np.float64)
        return Pose(self.position, self.rotation @ matrix, self.size)

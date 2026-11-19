"""Shared data types and SE(2) helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scan:
    """One LiDAR revolution, exactly as the sensor reported it.

    angle : raw sensor angle, rad (sensor convention; config maps it to robot frame)
    range : metres, 0 == no return
    tof   : per-point time offset from t_start, seconds
    Times are time.monotonic() seconds on the host.
    """
    seq: int
    t_start: float
    t_end: float
    t_recv: float
    angle: np.ndarray
    range: np.ndarray
    tof: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.angle.shape[0])


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wrap_arr(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def compose(p1, p2) -> np.ndarray:
    """p1 ∘ p2 for poses [x, y, th]."""
    x1, y1, t1 = p1
    x2, y2, t2 = p2
    c, s = math.cos(t1), math.sin(t1)
    return np.array([x1 + c * x2 - s * y2, y1 + s * x2 + c * y2, wrap(t1 + t2)])


def inverse(p) -> np.ndarray:
    x, y, t = p
    c, s = math.cos(t), math.sin(t)
    return np.array([-c * x - s * y, s * x - c * y, wrap(-t)])


def relative(p_from, p_to) -> np.ndarray:
    """Pose of p_to expressed in the frame of p_from."""
    return compose(inverse(p_from), p_to)


def transform(pts: np.ndarray, pose) -> np.ndarray:
    x, y, t = pose
    c, s = math.cos(t), math.sin(t)
    out = np.empty_like(pts)
    out[:, 0] = c * pts[:, 0] - s * pts[:, 1] + x
    out[:, 1] = s * pts[:, 0] + c * pts[:, 1] + y
    return out

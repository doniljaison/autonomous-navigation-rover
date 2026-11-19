"""
Phase-2 pipeline, reproduced for before/after comparison on the same logs.
==========================================================================
Faithful to phase2_autonomous/slam.py + hybrid.py:
  * PyLidar3-style scan: dict keyed by INTEGER degree (last sample wins), mm,
    150 < r <= 8000  (RC7: <= 360 points)
  * frame-to-frame point-to-point ICP (KDTree, 30 iters, 0.30 m gate), result
    always accepted when finite (RC1/RC2), no de-skew (RC3)
  * complementary filter pulling heading 30 % toward the firmware yaw "y",
    polled at 10 Hz (RC4)
  * occupancy update from every frame, +0.9 / -0.4, clamp +/-5 (RC5)
Only the raycast is jitted (same arithmetic) so replays run fast.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

from phase3_onboard.common import Scan, compose, transform, wrap
from phase3_onboard.slam.grid import OccupancyGrid


def legacy_points(scan: Scan, angle_sign=1.0, angle_offset=0.0) -> np.ndarray:
    d = {}
    for a, r in zip(scan.angle, scan.range):
        deg = int(math.degrees(angle_sign * float(a) + angle_offset)) % 360
        d[deg] = float(r) * 1000.0
    pts = []
    for deg, mm in d.items():
        if 150 < mm <= 8000:
            rad = math.radians(deg)
            pts.append([mm * math.cos(rad) / 1000.0, mm * math.sin(rad) / 1000.0])
    return np.array(pts) if pts else np.empty((0, 2))


def icp(source, target, max_iter=30, tolerance=1e-5, max_dist=0.30):
    if len(source) < 5 or len(target) < 5:
        return np.zeros(3), float("inf")
    src = source.copy()
    tree = cKDTree(target)
    R_acc = np.eye(2)
    t_acc = np.zeros(2)
    prev_err = float("inf")
    for _ in range(max_iter):
        dists, idxs = tree.query(src, k=1)
        mask = dists < max_dist
        if mask.sum() < 5:
            break
        S = src[mask]
        T = target[idxs[mask]]
        mu_s, mu_t = S.mean(axis=0), T.mean(axis=0)
        H = (S - mu_s).T @ (T - mu_t)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = mu_t - R @ mu_s
        src = src @ R.T + t
        R_acc = R @ R_acc
        t_acc = R @ t_acc + t
        err = dists[mask].mean()
        if abs(prev_err - err) < tolerance:
            break
        prev_err = err
    dtheta = math.atan2(R_acc[1, 0], R_acc[0, 0])
    return np.array([t_acc[0], t_acc[1], dtheta]), prev_err


class LegacyEngine:
    def __init__(self, P, imu_alpha=0.7, imu_poll_hz=10.0):
        self.P = P
        self.grid = OccupancyGrid(P.MAP_SIZE_M, P.MAP_RES_M, 0.9, -0.4, 5.0)
        self.pose = np.zeros(3)
        self.prev = None
        self.alpha = imu_alpha
        self.imu_period = 1.0 / imu_poll_hz
        self._imu_last_poll = -1e9
        self._imu_offset = None
        self.imu_yaw = None
        self.path = deque(maxlen=100000)
        self.frames = 0
        self.icp_log = []      # (t, icp dtheta, err)

    def feed_yaw(self, t, yaw_rad):
        """Mimics the 10 Hz HTTP poll: only every imu_period a sample is taken."""
        if yaw_rad is None or t - self._imu_last_poll < self.imu_period:
            return
        self._imu_last_poll = t
        if self._imu_offset is None:
            self._imu_offset = yaw_rad
        self.imu_yaw = wrap(yaw_rad - self._imu_offset)

    def step(self, scan: Scan):
        pts = legacy_points(scan, self.P.LIDAR_ANGLE_SIGN, self.P.LIDAR_ANGLE_OFFSET_RAD)
        if len(pts) < 30:
            return None
        dth, err = None, float("inf")
        if self.prev is not None and len(self.prev) >= 30:
            delta, err = icp(pts, self.prev)
            if err < float("inf"):
                self.pose = compose(self.pose, delta)
                dth = float(delta[2])
        if self.imu_yaw is not None:
            diff = wrap(self.imu_yaw - self.pose[2])
            self.pose[2] = wrap(self.pose[2] + (1.0 - self.alpha) * diff)
        world = transform(pts, self.pose)
        self.grid.update((self.pose[0], self.pose[1]), world, None)
        self.prev = pts
        self.frames += 1
        self.path.append((scan.t_end, *self.pose))
        self.icp_log.append((scan.t_end, dth, err, len(pts)))
        return self.pose.copy()

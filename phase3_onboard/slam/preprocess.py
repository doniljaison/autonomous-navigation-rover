"""
Scan preprocessing: validity + spike filtering, frame conversion, de-skew,
down-sampling.

Output points are in the ROBOT frame at the scan's END time (t_end), which is
the time stamp of the pose the SLAM engine estimates for the scan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phase3_onboard.common import Scan
from phase3_onboard.slam.jit import njit


@dataclass
class Prepared:
    t: float                 # pose time (scan end)
    pts: np.ndarray          # Nx2 robot-frame hits (full resolution, de-skewed)
    pts_t: np.ndarray        # per-point absolute time
    free_pts: np.ndarray     # Mx2 endpoints for free-only rays (beyond map range / no return)
    match_pts: np.ndarray    # Kx2 down-sampled subset for matching
    n_raw: int
    n_valid: int
    n_spikes: int
    n_zero: int


@njit
def _spike_mask(r, valid, abs_thr, rel_thr):
    """Isolated-point test in scan order: a valid point whose range differs from
    BOTH its valid angular neighbours by more than the threshold is a spike."""
    n = r.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    idx = np.empty(n, dtype=np.int64)
    m = 0
    for i in range(n):
        if valid[i]:
            idx[m] = i
            m += 1
    if m < 3:
        return out
    for k in range(m):
        i = idx[k]
        ip = idx[k - 1] if k > 0 else idx[m - 1]
        inx = idx[k + 1] if k < m - 1 else idx[0]
        thr = max(abs_thr, rel_thr * r[i])
        if abs(r[i] - r[ip]) > thr and abs(r[i] - r[inx]) > thr:
            out[i] = True
    return out


def sensor_to_robot(ang_sensor: np.ndarray, rng: np.ndarray, P) -> np.ndarray:
    a = ang_sensor + P.LIDAR_MOUNT_YAW_RAD
    xs = rng * np.cos(a) + P.LIDAR_MOUNT_X_M
    ys = rng * np.sin(a) + P.LIDAR_MOUNT_Y_M
    return np.stack([xs, ys], axis=1)


def deskew(pts: np.ndarray, pts_t: np.ndarray, t_ref: float, yaw_rel: np.ndarray,
           v_body: tuple) -> np.ndarray:
    """Move each point from the robot frame at its own time to the robot frame at
    t_ref.  yaw_rel[i] = yaw(t_ref) - yaw(t_i) (CCW rad); v_body = (vx, vy) m/s."""
    c = np.cos(-yaw_rel)
    s = np.sin(-yaw_rel)
    dt = t_ref - pts_t
    x = c * pts[:, 0] - s * pts[:, 1] - v_body[0] * dt
    y = s * pts[:, 0] + c * pts[:, 1] - v_body[1] * dt
    return np.stack([x, y], axis=1)


def voxel_downsample(pts: np.ndarray, voxel: float, max_pts: int) -> np.ndarray:
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    k = keys[:, 0] * 100003 + keys[:, 1]
    _, first = np.unique(k, return_index=True)
    first.sort()                       # keep angular order
    sel = pts[first]
    if len(sel) > max_pts:
        idx = np.linspace(0, len(sel) - 1, max_pts).astype(np.int64)
        sel = sel[idx]
    return sel


def prepare(scan: Scan, P, yaw_fn=None, v_body=(0.0, 0.0), w_fallback: float = 0.0) -> Prepared:
    """yaw_fn(ts) -> relative integrated yaw at absolute times ts (gyro), or None
    to use constant angular velocity w_fallback for de-skew."""
    raw_a = scan.angle.astype(np.float64)
    r = scan.range.astype(np.float64)
    ang = P.LIDAR_ANGLE_SIGN * raw_a + P.LIDAR_ANGLE_OFFSET_RAD
    t_pts = scan.t_start + scan.tof.astype(np.float64)
    finite = np.isfinite(r)
    r = np.where(finite, r, 0.0)
    zero = r <= 1e-6
    lo = max(P.PRE_MIN_RANGE_M, P.LIDAR_MIN_RANGE_M)
    valid = (~zero) & (r >= lo) & (r <= P.LIDAR_MAX_RANGE_M)
    spikes = _spike_mask(r, valid, float(P.PRE_SPIKE_ABS_M), float(P.PRE_SPIKE_REL))
    good = valid & ~spikes

    pts_all = sensor_to_robot(ang, r, P)
    # robot self-hits
    self_hit = np.hypot(pts_all[:, 0], pts_all[:, 1]) < P.SELF_FILTER_RADIUS_M
    good &= ~self_hit

    hit_mask = good & (r <= P.PRE_MAX_RANGE_M)
    far_mask = good & (r > P.PRE_MAX_RANGE_M)

    t_ref = scan.t_end
    if P.DESKEW:
        if yaw_fn is not None:
            yaw_rel = yaw_fn(np.array([t_ref]))[0] - yaw_fn(t_pts)
        else:
            yaw_rel = w_fallback * (t_ref - t_pts)
        pts_ds = deskew(pts_all, t_pts, t_ref, yaw_rel, v_body)
    else:
        pts_ds = pts_all

    pts = pts_ds[hit_mask]
    pts_t = t_pts[hit_mask]

    free_list = []
    if np.any(far_mask):
        far = pts_ds[far_mask]
        d = np.hypot(far[:, 0], far[:, 1])[:, None]
        free_list.append(far / d * min(P.MAP_MAX_RAY_M, P.PRE_MAX_RANGE_M))
    if P.MAP_FREE_RAY_NO_RETURN and np.any(zero):
        a = ang[zero]
        free_list.append(sensor_to_robot(a, np.full(a.shape, P.MAP_MAX_RAY_M), P))
    free = np.concatenate(free_list) if free_list else np.empty((0, 2))

    rr = np.hypot(pts[:, 0], pts[:, 1])
    mp = pts[rr <= P.MATCH_MAX_RANGE_M]
    match = voxel_downsample(mp, P.MATCH_VOXEL_M, P.MATCH_MAX_POINTS)
    return Prepared(t_ref, pts, pts_t, free, match, scan.n, int(valid.sum()),
                    int(spikes.sum()), int(zero.sum()))


"""
SLAM v3 engine.
===============
Per LiDAR revolution:
  1. preprocess (validity, spikes, self-hits)            -> preprocess.prepare
  2. de-skew with gyro yaw + constant body velocity      -> preprocess.deskew
  3. motion prior: gyro yaw + constant velocity (zero when commands are idle)
  4. scan-to-MAP match: correlative window + Gauss-Newton  -> matcher.match
  5. gate: inliers / RMS / |match-gyro| yaw / speed / ambiguity / degeneracy
     rejected -> pose = prior, map NOT updated, counted; K in a row -> LOST
  6. fuse: gyro-propagated heading corrected by the match, weighted by quality;
     weak translational direction falls back to the prior
  7. map update from accepted frames only (numba raycast)
  8. snapshot for readers (lock-protected copy)

`step()` takes everything it needs as arguments so the identical code runs
live (main.py) and offline (tools/replay.py).
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from phase3_onboard.common import Scan, relative, transform, wrap
from phase3_onboard.imu import GyroYaw
from phase3_onboard.slam import matcher as M
from phase3_onboard.slam.grid import OccupancyGrid
from phase3_onboard.slam.preprocess import prepare


@dataclass
class StepResult:
    t: float
    pose: np.ndarray
    prior: np.ndarray
    accepted: bool
    reason: str
    lost: bool
    gyro_dth: Optional[float]
    match_dth: Optional[float]
    inliers: int = 0
    inlier_ratio: float = 0.0
    rms: float = 0.0
    ambiguity: float = 0.0
    eig_min: float = 0.0
    n_pts: int = 0
    n_match: int = 0
    n_spikes: int = 0
    timing_ms: dict = field(default_factory=dict)

    def as_log(self) -> dict:
        return {
            "pose": [round(float(v), 5) for v in self.pose],
            "prior": [round(float(v), 5) for v in self.prior],
            "acc": self.accepted, "why": self.reason, "lost": self.lost,
            "gdth": None if self.gyro_dth is None else round(self.gyro_dth, 5),
            "mdth": None if self.match_dth is None else round(self.match_dth, 5),
            "inl": self.inliers, "ir": round(self.inlier_ratio, 3), "rms": round(self.rms, 4),
            "amb": round(self.ambiguity, 3), "eig": round(self.eig_min, 3),
            "npts": self.n_pts, "nm": self.n_match, "nsp": self.n_spikes,
            "ms": {k: round(v, 2) for k, v in self.timing_ms.items()},
        }


class SlamEngine:
    def __init__(self, P, grid: Optional[OccupancyGrid] = None, init_pose=(0.0, 0.0, 0.0)):
        self.P = P
        self.grid = grid or OccupancyGrid(P.MAP_SIZE_M, P.MAP_RES_M, P.MAP_HIT, P.MAP_MISS, P.MAP_CLAMP)
        self.pose = np.array(init_pose, dtype=np.float64)
        self.gyro = GyroYaw(P.YAW_SOURCE, bias_window_s=P.GYRO_BIAS_WINDOW_S)
        self.v_body = np.zeros(2)
        self.w_est = 0.0
        self.last_t: Optional[float] = None
        self.last_accept_t: Optional[float] = None
        self.last_accept_pose: Optional[np.ndarray] = None
        self.frames = 0
        self.accepted = 0
        self.rejected = 0
        self.consecutive_rej = 0
        self.lost = False
        self.skipped = 0
        self.bootstrap_frames = 3 if grid is None else 0
        self.recent = deque(maxlen=30)   # accept flags for rejection-rate
        self.path = deque(maxlen=3000)
        self.last_result: Optional[StepResult] = None
        self.last_scan_world: np.ndarray = np.empty((0, 2))
        self.last_pts_robot: np.ndarray = np.empty((0, 2))
        self.relocalize_first = False    # set when starting on a loaded map
        self.gyro_sign_suspect = 0
        self.lock = threading.Lock()

    # ── helpers ────────────────────────────────────────────────────────────────
    def reject_rate(self) -> float:
        return 0.0 if not self.recent else 1.0 - (sum(self.recent) / len(self.recent))

    @staticmethod
    def _still_enough(dth, dt, cmd_idle) -> bool:
        """Re-localise only when (nearly) stationary: a wrong gyro also corrupts
        de-skew, so matching a spinning rover's scans is how bad poses get in."""
        rate = abs(dth) / dt if dt > 0 else 0.0
        return cmd_idle or rate < math.radians(10.0)

    def clear_lost(self):
        self.lost = False
        self.consecutive_rej = 0

    def _prior(self, t: float, cmd_idle: bool, cmd=None):
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        if self.gyro.available and self.last_t is not None:
            dth = self.gyro.delta(self.last_t, t)
            gyro_dth = dth
        else:
            dth = self.w_est * dt
            gyro_dth = None
        if cmd_idle:
            v = np.zeros(2)
            if gyro_dth is None:
                dth = 0.0
        else:
            v = self.v_body.copy()
            if cmd is not None and self.P.DUTY_TO_MPS:
                # lightly-trusted calibrated model blended with constant velocity
                v_cal = self.P.DUTY_TO_MPS * 0.5 * (cmd[0] + cmd[1])
                v = np.array([0.5 * v[0] + 0.5 * v_cal, 0.5 * v[1]])
            if gyro_dth is None and cmd is not None and self.P.DUTY_TO_RADPS:
                dth = 0.5 * dth + 0.5 * self.P.DUTY_TO_RADPS * 0.5 * (cmd[1] - cmd[0]) * dt
        th_mid = self.pose[2] + 0.5 * dth
        c, s = math.cos(th_mid), math.sin(th_mid)
        dx = (c * v[0] - s * v[1]) * dt
        dy = (s * v[0] + c * v[1]) * dt
        prior = np.array([self.pose[0] + dx, self.pose[1] + dy, wrap(self.pose[2] + dth)])
        return prior, gyro_dth, dth, dt, v

    # ── main entry ─────────────────────────────────────────────────────────────
    def step(self, scan: Scan, cmd_idle: bool = False, cmd=None) -> Optional[StepResult]:
        """cmd_idle: no non-zero motor command for PRIOR_STOP_AFTER_S.
        cmd: latest (L, R) duty, only used if DUTY_TO_MPS / DUTY_TO_RADPS are set."""
        P = self.P
        T = {}
        t0 = time.perf_counter()
        t = scan.t_end
        prior, gyro_dth, dth_used, dt, v_prior = self._prior(t, cmd_idle, cmd)

        yaw_fn = self.gyro.yaw_at_many if self.gyro.available else None
        w_fb = (dth_used / dt) if (dt > 0 and yaw_fn is None) else 0.0
        prep = prepare(scan, P, yaw_fn=yaw_fn, v_body=tuple(v_prior), w_fallback=w_fb)
        T["pre"] = (time.perf_counter() - t0) * 1e3
        self.frames += 1

        if len(prep.pts) < P.PRE_MIN_POINTS:
            self.skipped += 1
            self.pose = prior
            self.last_t = t
            res = StepResult(t, prior.copy(), prior, False, "few_points", self.lost, gyro_dth, None,
                             n_pts=len(prep.pts), n_spikes=prep.n_spikes, timing_ms=T)
            self._publish(res, prep)
            return res

        # ── bootstrap: first frames build the map at the prior ────────────────
        if self.accepted < self.bootstrap_frames:
            self.pose = prior
            t1 = time.perf_counter()
            self._map_update(prep, self.pose)
            T["map"] = (time.perf_counter() - t1) * 1e3
            self.accepted += 1
            self.recent.append(1)
            self.last_accept_t, self.last_accept_pose = t, self.pose.copy()
            self.last_t = t
            res = StepResult(t, self.pose.copy(), prior, True, "bootstrap", False, gyro_dth, None,
                             n_pts=len(prep.pts), n_match=len(prep.match_pts),
                             n_spikes=prep.n_spikes, timing_ms=T)
            T["total"] = (time.perf_counter() - t0) * 1e3
            self._publish(res, prep)
            return res

        # ── LOST and still moving: do not even try (rover is being stopped) ──
        lost_mode = self.consecutive_rej >= P.GATE_LOST_AFTER
        if lost_mode and not self._still_enough(dth_used, dt, cmd_idle):
            self.pose = prior
            self.rejected += 1
            self.consecutive_rej += 1
            self.recent.append(0)
            self.last_t = t
            T["total"] = (time.perf_counter() - t0) * 1e3
            res = StepResult(t, prior.copy(), prior, False, "lost_moving", self.lost, gyro_dth, None,
                             n_pts=len(prep.pts), n_spikes=prep.n_spikes, timing_ms=T)
            self._publish(res, prep)
            return res

        # ── match against the map ───────────────────────────────────────────────
        t1 = time.perf_counter()
        g = self.grid
        cx, cy = g.w2c(prior[0], prior[1])
        half = int(P.MATCH_LOCAL_HALF_M / g.res)
        D, wx0, wy0, n_occ = g.distance_field(int(cx), int(cy), half, P.MAP_OCC_THRESH, P.MATCH_FIELD_MAX_M)
        T["field"] = (time.perf_counter() - t1) * 1e3
        t1 = time.perf_counter()
        rot_since = abs(gyro_dth) if gyro_dth is not None else abs(dth_used)
        th_half = math.radians(P.MATCH_COARSE_TH_DEG) + P.MATCH_COARSE_TH_PER_RAD * rot_since
        MP = P
        mpts = prep.match_pts
        reloc = self.relocalize_first or self.consecutive_rej >= P.GATE_LOST_AFTER
        gyro_dth_gate = gyro_dth
        if reloc:
            # (a) first scan on a loaded map: the start pose is only the taped mark;
            # (b) LOST: the pose has drifted off the map.  Search a wide window,
            # skip the gyro/speed checks, but demand a clearly better match.
            from types import SimpleNamespace
            MP = SimpleNamespace(**vars(P))
            MP.MATCH_COARSE_XY_M = P.RELOC_XY_M
            MP.MATCH_COARSE_TH_STEP_DEG = 1.0
            th_half = math.radians(P.RELOC_TH_DEG)
            gyro_dth_gate = None
            if len(mpts) > 160:
                mpts = mpts[np.linspace(0, len(mpts) - 1, 160).astype(np.int64)]
        mr = M.match(D, wx0 - g.orig, wy0 - g.orig, g.res, mpts, prior, th_half, MP)
        T["match"] = (time.perf_counter() - t1) * 1e3

        # ── gate ───────────────────────────────────────────────────────────────
        match_dth = wrap(mr.pose[2] - self.pose[2])
        reason = "ok"
        min_inl = max(P.GATE_MIN_INLIERS, 0)
        if n_occ < 20:
            reason = "empty_map"
        elif mr.inliers < min_inl or mr.inlier_ratio < P.GATE_MIN_INLIER_RATIO:
            reason = "inliers"
        elif mr.rms > P.GATE_MAX_RMS_M:
            reason = "rms"
        elif gyro_dth_gate is not None and abs(wrap(match_dth - gyro_dth_gate)) > (
                math.radians(P.GATE_YAW_TOL_DEG) + P.GATE_YAW_TOL_FRAC * abs(gyro_dth_gate)):
            reason = "yaw_vs_gyro"
        elif dt > 0 and not reloc and math.hypot(mr.pose[0] - self.pose[0], mr.pose[1] - self.pose[1]) > \
                P.GATE_MAX_SPEED_MPS * dt + 0.05:
            reason = "speed"
        elif mr.ambiguity > (P.RELOC_MAX_AMBIGUITY if reloc else P.GATE_AMBIGUITY):
            reason = "ambiguous"
        elif reloc and (mr.inlier_ratio < P.RELOC_MIN_INLIER_RATIO or mr.rms > 0.8 * P.GATE_MAX_RMS_M):
            reason = "reloc_weak"
        accepted = reason == "ok"
        # inverted-gyro detector: a good match that agrees with MINUS the gyro
        if gyro_dth is not None and abs(gyro_dth) > math.radians(3.0) and mr.inlier_ratio > 0.7:
            tol = math.radians(P.GATE_YAW_TOL_DEG) + P.GATE_YAW_TOL_FRAC * abs(gyro_dth)
            if abs(wrap(match_dth + gyro_dth)) < tol < abs(wrap(match_dth - gyro_dth)):
                self.gyro_sign_suspect += 1

        if accepted:
            # ── fuse ─────────────────────────────────────────────────────────
            q = min(1.0, max(0.0, (mr.inlier_ratio - P.GATE_MIN_INLIER_RATIO) /
                             max(1e-6, 1.0 - P.GATE_MIN_INLIER_RATIO)))
            w_th = P.FUSE_YAW_WEIGHT_MIN + (1.0 - P.FUSE_YAW_WEIGHT_MIN) * q
            th = wrap(prior[2] + w_th * wrap(mr.pose[2] - prior[2]))
            xy = mr.pose[:2].copy()
            if mr.degenerate:
                # keep the prior along the weak (unobservable) direction
                e = mr.eig_vec_weak / (np.linalg.norm(mr.eig_vec_weak) + 1e-12)
                d = xy - prior[:2]
                xy = xy - e * float(d @ e)
                reason = "ok_degenerate"
            new_pose = np.array([xy[0], xy[1], th])
            # velocity estimate for the next prior / de-skew
            if self.last_accept_pose is not None and self.last_accept_t is not None:
                dta = t - self.last_accept_t
                if 0.02 < dta < 1.0:
                    rel = relative(self.last_accept_pose, new_pose)
                    vb = rel[:2] / dta
                    a = P.PRIOR_VEL_ALPHA
                    self.v_body = (1 - a) * self.v_body + a * vb
                    self.w_est = (1 - a) * self.w_est + a * rel[2] / dta
            if cmd_idle:
                self.v_body[:] = 0.0
                self.w_est = 0.0
            # bounded online gyro scale correction from good, clearly-turning frames
            if (P.GYRO_SCALE_ADAPT and not reloc and gyro_dth is not None
                    and abs(gyro_dth) > math.radians(5.0) and mr.inlier_ratio > 0.7):
                ratio = wrap(mr.pose[2] - self.pose[2]) / gyro_dth
                if 0.7 < ratio < 1.3:
                    c = self.gyro.scale_corr * (1.0 + P.GYRO_SCALE_ADAPT_RATE * (ratio - 1.0))
                    self.gyro.scale_corr = min(1.0 + P.GYRO_SCALE_ADAPT_MAX,
                                               max(1.0 - P.GYRO_SCALE_ADAPT_MAX, c))
            if reloc:
                reason = "relocalized"
                self.v_body[:] = 0.0
                self.w_est = 0.0
                self.lost = False
            self.pose = new_pose
            self.relocalize_first = False
            self.accepted += 1
            self.consecutive_rej = 0
            self.last_accept_t, self.last_accept_pose = t, new_pose.copy()
            if not reloc:   # a re-localising frame never writes the map
                t1 = time.perf_counter()
                self._map_update(prep, self.pose)
                T["map"] = (time.perf_counter() - t1) * 1e3
        else:
            self.pose = prior
            self.rejected += 1
            self.consecutive_rej += 1
            if self.consecutive_rej >= P.GATE_LOST_AFTER:
                self.lost = True
        self.recent.append(1 if accepted else 0)
        self.last_t = t
        T["total"] = (time.perf_counter() - t0) * 1e3
        res = StepResult(t, self.pose.copy(), prior, accepted, reason, self.lost, gyro_dth, match_dth,
                         inliers=mr.inliers, inlier_ratio=mr.inlier_ratio, rms=mr.rms,
                         ambiguity=mr.ambiguity, eig_min=mr.eig_min, n_pts=len(prep.pts),
                         n_match=len(prep.match_pts), n_spikes=prep.n_spikes, timing_ms=T)
        self._publish(res, prep)
        return res

    def _map_update(self, prep, pose):
        P = self.P
        hits = transform(prep.pts, pose)
        free = transform(prep.free_pts, pose) if len(prep.free_pts) else None
        # sensor origin in world
        c, s = math.cos(pose[2]), math.sin(pose[2])
        ox = pose[0] + c * P.LIDAR_MOUNT_X_M - s * P.LIDAR_MOUNT_Y_M
        oy = pose[1] + s * P.LIDAR_MOUNT_X_M + c * P.LIDAR_MOUNT_Y_M
        if P.MAP_RAY_STRIDE > 1:
            # all hits still mark cells; only every Nth ray carves free space
            with self.lock:
                self.grid.update((ox, oy), hits[::P.MAP_RAY_STRIDE], free)
                rest = np.delete(hits, np.s_[::P.MAP_RAY_STRIDE], axis=0)
                if len(rest):
                    cx, cy = self.grid.w2c(rest[:, 0], rest[:, 1])
                    ok = (cx >= 0) & (cx < self.grid.n) & (cy >= 0) & (cy < self.grid.n)
                    np.add.at(self.grid.log, (cy[ok], cx[ok]), np.float32(self.grid.hit))
                    np.clip(self.grid.log, -self.grid.clamp, self.grid.clamp, out=self.grid.log)
        else:
            with self.lock:
                self.grid.update((ox, oy), hits, free)

    def _publish(self, res: StepResult, prep):
        with self.lock:
            self.last_result = res
            self.path.append((res.t, float(res.pose[0]), float(res.pose[1]), float(res.pose[2])))
            self.last_pts_robot = prep.pts
            self.last_scan_world = transform(prep.pts[::3], res.pose) if len(prep.pts) else np.empty((0, 2))

    # ── reader API (other threads) ──────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self.lock:
            r = self.last_result
            return {
                "pose": self.pose.copy(),
                "lost": self.lost,
                "frames": self.frames,
                "accepted": self.accepted,
                "rejected": self.rejected,
                "skipped": self.skipped,
                "reject_rate": self.reject_rate(),
                "last": r.as_log() if r else None,
                "map_version": self.grid.version,
                "scan_world": self.last_scan_world.copy(),
                "pts_robot": self.last_pts_robot,
                "gyro_bias": self.gyro.bias,
                "gyro_ready": self.gyro.bias_ready,
                "gyro_scale_corr": self.gyro.scale_corr,
                "gyro_sign_suspect": self.gyro_sign_suspect,
            }

    def map_copy(self) -> np.ndarray:
        with self.lock:
            return self.grid.log.copy()


class SlamInputs:
    """Feeds raw rover traffic into the engine: IMU samples -> gyro integrator,
    motor commands -> "idle" flag (used for the zero-velocity prior and for
    standstill gyro-bias estimation).  Live, the SLAM thread drains a queue into
    this; in replay, the log's rx/tx events go straight in.  Same code path."""

    def __init__(self, P, engine: SlamEngine):
        from phase3_onboard.imu import ImuParser
        self.P = P
        self.eng = engine
        self.parser = ImuParser(P)
        self.last_nonzero_cmd_t = -1e9
        self.last_cmd = (0.0, 0.0)
        self.first_t: Optional[float] = None
        self.n_imu = 0
        self.last_yaw_abs: Optional[float] = None

    def on_tx(self, t: float, line: str):
        msg = self.parser.parse_line(line)
        if msg and msg.get("T") == 1:
            try:
                self.last_cmd = (float(msg.get("L", 0)), float(msg.get("R", 0)))
                if abs(float(msg.get("L", 0))) > 1e-3 or abs(float(msg.get("R", 0))) > 1e-3:
                    self.last_nonzero_cmd_t = t
            except (TypeError, ValueError):
                pass

    def on_rx(self, t: float, line: str):
        if self.first_t is None:
            self.first_t = t
        msg = self.parser.parse_line(line)
        if not msg:
            return None
        s = self.parser.sample(t, msg)
        if s is None:
            return None
        self.n_imu += 1
        if s.yaw is not None:
            self.last_yaw_abs = s.yaw
        g = self.eng.gyro
        still = self.idle_for(t) > 0.5
        if s.rate is not None and g.bias_ready:
            still = still and abs(s.rate - g.bias) < math.radians(3.0)
        elif s.rate is not None:
            still = still and abs(s.rate) < math.radians(5.0)
        g.add(s, stationary=still)
        return s

    def idle_for(self, t: float) -> float:
        return t - self.last_nonzero_cmd_t

    def cmd_idle(self, t: float) -> bool:
        return self.idle_for(t) > self.P.PRIOR_STOP_AFTER_S


def warmup_jit(P):
    """Compile numba kernels before going real-time (first compile on a Pi 4B can
    take 20-60 s; results are cached in __pycache__ for later starts)."""
    from phase3_onboard.slam.grid import OccupancyGrid as G
    g = G(2.0, P.MAP_RES_M)
    g.update((0.0, 0.0), np.array([[0.5, 0.2], [0.7, -0.1]]), np.array([[0.9, 0.0]]))
    g.distance_field(g.orig, g.orig, 5, 0.1, 0.3)
    M.warmup()
    from phase3_onboard.slam.preprocess import _spike_mask
    _spike_mask(np.ones(5), np.ones(5, dtype=np.bool_), 0.1, 0.05)


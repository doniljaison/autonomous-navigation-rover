"""
Synthetic rover + G4 + IMU simulator.
=====================================
Purpose: develop and regression-test SLAM offline (it must reproduce the
old pipeline's deformation), and run the full on-board runtime without
hardware (``python -m phase3_onboard.main --sim``).

Models (deliberately pessimistic where the real rover is known to be bad):
  * skid-steer kinematics from *duty* commands, first-order lag, dead band,
    random per-run gain error and slowly varying slip (no encoders exist)
  * firmware 3 s command timeout
  * G4: 10 Hz, ~900 points / revolution, per-point timestamps (scan skew),
    range noise, 0 returns (dropouts), random spikes
  * IMU: gyro z in deg/s with bias + noise, firmware yaw "y" with drift and
    motor-induced magnetic disturbance, replies to {"T":126}
  * chassis feedback {"T":1001} at 20 Hz when {"T":131,"cmd":1} is on

Everything leaves the simulator as raw scans and raw JSON lines, i.e. in the
exact form the logger records for the real rover.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from phase3_onboard.common import Scan, wrap


# ─────────────────────────────────────────────────────────────────────────────
#  World
# ─────────────────────────────────────────────────────────────────────────────

def box_segments(cx, cy, w, h, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    pts = [(cx + c * x - s * y, cy + s * x + c * y) for x, y in pts]
    return [(pts[i], pts[(i + 1) % 4]) for i in range(4)]


def square_room(side=4.0, clutter=True):
    """Square room centred on the origin.  `clutter` adds a few small objects
    (a real room is never perfectly empty; set False for the worst case)."""
    segs = box_segments(0, 0, side, side)
    if clutter:
        segs += box_segments(side / 2 - 0.35, side / 2 - 0.25, 0.5, 0.35)      # cabinet in a corner
        segs += box_segments(-side / 2 + 0.2, 0.6, 0.25, 0.25)                  # bin against a wall
        segs += [((-0.3, -side / 2), (-0.3, -side / 2 + 0.06))]                 # door frame stub
    return np.array(segs, dtype=np.float64)


def raycast_segments(ox, oy, ang, segs, max_r):
    """Vectorised ray/segment intersection.  Returns range per ray (inf = none)."""
    dx, dy = np.cos(ang)[:, None], np.sin(ang)[:, None]
    ax, ay = segs[:, 0, 0][None, :], segs[:, 0, 1][None, :]
    bx, by = segs[:, 1, 0][None, :], segs[:, 1, 1][None, :]
    ex, ey = bx - ax, by - ay
    ox = np.asarray(ox)[:, None] if np.ndim(ox) else ox
    oy = np.asarray(oy)[:, None] if np.ndim(oy) else oy
    den = dx * ey - dy * ex
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((ax - ox) * ey - (ay - oy) * ex) / den
        u = ((ax - ox) * dy - (ay - oy) * dx) / den
    ok = (np.abs(den) > 1e-12) & (t > 1e-6) & (u >= 0) & (u <= 1)
    t = np.where(ok, t, np.inf)
    r = t.min(axis=1)
    r[r > max_r] = np.inf
    return r


# ─────────────────────────────────────────────────────────────────────────────
#  Simulator core
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimParams:
    seed: int = 1
    room_side: float = 4.0
    clutter: bool = True
    start_pose: tuple = (-0.9, -0.8, 0.0)
    # kinematics (duty -> motion)
    k_v: float = 1.0           # m/s per unit mean duty
    k_w: float = 4.4           # rad/s per unit half-difference duty
    gain_err: float = 0.08     # +/- random per-run gain error
    slip_noise: float = 0.10   # slowly varying multiplicative noise
    deadband: float = 0.05
    tau: float = 0.12
    # LiDAR
    scan_hz: float = 10.0
    sample_hz: float = 9000.0
    range_sigma: float = 0.008
    range_sigma_rel: float = 0.004
    dropout: float = 0.04
    spike_rate: float = 0.006
    max_range: float = 12.0
    lidar_mount: tuple = (0.0, 0.0, 0.0)
    lidar_angle_sign: float = 1.0      # raw = sign * (robot angle - mount yaw)
    # IMU
    gyro_bias_dps: float = 0.4
    gyro_noise_dps: float = 0.35
    gyro_scale_err: float = 0.01
    imu_stream_hz: float = 25.0        # used offline (no request/response)
    feedback_hz: float = 20.0
    yaw_drift_dps: float = 0.3
    yaw_mag_disturb_deg: float = 8.0
    physics_hz: float = 1000.0


class SimCore:
    """Time-stepped simulator.  Call `advance(dt)`; it fills the out-queues."""

    def __init__(self, p: SimParams | None = None):
        self.p = p or SimParams()
        self.rng = np.random.default_rng(self.p.seed)
        self.segs = square_room(self.p.room_side, self.p.clutter)
        self.t = 0.0
        self.pose = np.array(self.p.start_pose, dtype=np.float64)
        self.v = 0.0
        self.w = 0.0
        g = self.p.gain_err
        self.kv = self.p.k_v * (1 + self.rng.uniform(-g, g))
        self.kw = self.p.k_w * (1 + self.rng.uniform(-g, g))
        self.slip = 1.0
        self.cmd = (0.0, 0.0)
        self.cmd_t = -1e9
        self.override: Optional[Callable[[float], tuple]] = None  # hand-carry: t -> (v, w)
        self.feedback_on = False
        # LiDAR state
        self._rev_period = 1.0 / self.p.scan_hz
        self._pts_per_rev = int(round(self.p.sample_hz / self.p.scan_hz))
        self._rev_t0 = 0.0
        self._rev_buf_t: list = []
        self._rev_buf_pose: list = []
        self._seq = 0
        # IMU state
        self._gyro_bias = math.radians(self.p.gyro_bias_dps) * self.rng.choice([-1, 1])
        self._fw_yaw = 0.0
        self._next_imu = 0.0
        self._next_fb = 0.0
        self._pending_imu_req: list = []
        # outputs
        self.scans: list = []          # Scan
        self.lines: list = []          # (t, line)
        self.truth: list = []          # (t, pose)
        self.stream_imu = True         # offline: emit IMU at imu_stream_hz without requests
        self._next_truth = 0.0

    # -- inputs -------------------------------------------------------------------
    def handle_tx(self, line: str):
        try:
            d = json.loads(line)
        except Exception:
            return
        T = d.get("T")
        if T == 1:
            self.cmd = (float(d.get("L", 0)), float(d.get("R", 0)))
            self.cmd_t = self.t
        elif T == 126:
            self._pending_imu_req.append(self.t + 0.004 + self.rng.uniform(0, 0.004))
        elif T == 131:
            self.feedback_on = bool(d.get("cmd", 0))

    # -- physics ------------------------------------------------------------------
    def _duty(self, x):
        x = max(-0.5, min(0.5, x))
        if abs(x) < self.p.deadband:
            return 0.0
        return x

    def _step_physics(self, dt):
        if self.override is not None:
            v_t, w_t = self.override(self.t)
            self.v, self.w = v_t, w_t
        else:
            L, R = self.cmd
            if self.t - self.cmd_t > 3.0:           # firmware timeout
                L = R = 0.0
            L, R = self._duty(L), self._duty(R)
            v_t = self.kv * 0.5 * (L + R) * self.slip
            w_t = self.kw * 0.5 * (R - L) * self.slip
            a = dt / self.p.tau
            self.v += a * (v_t - self.v)
            self.w += a * (w_t - self.w)
            # slip random walk
            self.slip += self.rng.normal(0, self.p.slip_noise * math.sqrt(dt) * 0.5)
            self.slip = min(1 + self.p.slip_noise, max(1 - self.p.slip_noise, self.slip))
        x, y, th = self.pose
        th_mid = th + 0.5 * self.w * dt
        self.pose = np.array([x + self.v * math.cos(th_mid) * dt,
                              y + self.v * math.sin(th_mid) * dt,
                              wrap(th + self.w * dt)])
        # firmware yaw drifts + magnetic disturbance when motors run
        self._fw_yaw = wrap(self._fw_yaw + self.w * dt + math.radians(self.p.yaw_drift_dps) * dt)

    def _imu_line(self, t):
        gz = (self.w * (1 + self.p.gyro_scale_err) + self._gyro_bias
              + self.rng.normal(0, math.radians(self.p.gyro_noise_dps)))
        moving = abs(self.cmd[0]) + abs(self.cmd[1]) > 0.05 and t - self.cmd_t < 3.0
        mag = math.radians(self.p.yaw_mag_disturb_deg) * (math.sin(t * 0.7) if moving else 0.0)
        msg = {"T": 1002, "r": round(float(self.rng.normal(0, 0.3)), 2), "p": round(float(self.rng.normal(0, 0.3)), 2),
               "y": round(math.degrees(wrap(self._fw_yaw + mag)), 2),
               "ax": round(float(self.rng.normal(0, 30)), 1), "ay": round(float(self.rng.normal(0, 30)), 1), "az": 1024.0,
               "gx": round(float(self.rng.normal(0, 0.3)), 3), "gy": round(float(self.rng.normal(0, 0.3)), 3),
               "gz": round(math.degrees(gz), 3),
               "mx": 120, "my": -40, "mz": 300, "temp": 31.5}
        self.lines.append((t, json.dumps(msg, separators=(",", ":"))))

    def _fb_line(self, t):
        msg = {"T": 1001, "L": round(self.cmd[0], 3), "R": round(self.cmd[1], 3),
               "r": 0.1, "p": -0.2, "v": round(11.9 - 0.0005 * t, 2)}
        self.lines.append((t, json.dumps(msg, separators=(",", ":"))))

    def _lidar_pose(self, pose):
        mx, my, myaw = self.p.lidar_mount
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return (pose[0] + c * mx - s * my, pose[1] + s * mx + c * my, pose[2] + myaw)

    def _finish_revolution(self):
        n = len(self._rev_buf_t)
        if n == 0:
            return
        ts = np.array(self._rev_buf_t)
        poses = np.array(self._rev_buf_pose)
        k = np.arange(n)
        sensor_ang = 2 * np.pi * k / n + self.rng.normal(0, 0.0005, n)
        world_ang = poses[:, 2] + sensor_ang
        r = raycast_segments(poses[:, 0], poses[:, 1], world_ang, self.segs, self.p.max_range)
        sig = self.p.range_sigma + self.p.range_sigma_rel * np.where(np.isfinite(r), r, 0)
        r = r + self.rng.normal(0, 1, n) * sig
        r = np.where(np.isfinite(r), r, 0.0)
        drop = self.rng.random(n) < self.p.dropout
        r[drop] = 0.0
        spike = self.rng.random(n) < self.p.spike_rate
        r[spike] = self.rng.uniform(0.3, self.p.max_range, spike.sum())
        r = np.round(r, 3)  # mm resolution
        raw = wrap_np(self.p.lidar_angle_sign * sensor_ang)
        t0 = self._rev_t0
        s = Scan(self._seq, t0, float(ts[-1]), float(ts[-1]) + 0.002,
                 raw.astype(np.float32), r.astype(np.float32), (ts - t0).astype(np.float32))
        self._seq += 1
        self.scans.append(s)

    def advance(self, duration: float):
        dt = 1.0 / self.p.physics_hz
        steps = max(1, int(round(duration / dt)))
        pts_per_step = self.p.sample_hz * dt
        acc = getattr(self, "_pt_acc", 0.0)
        for _ in range(steps):
            self._step_physics(dt)
            self.t += dt
            # LiDAR samples during this step (pose interpolation is unnecessary at 1 kHz)
            acc += pts_per_step
            lp = self._lidar_pose(self.pose)
            while acc >= 1.0:
                acc -= 1.0
                self._rev_buf_t.append(self.t)
                self._rev_buf_pose.append(lp)
                if len(self._rev_buf_t) >= self._pts_per_rev:
                    self._finish_revolution()
                    self._rev_buf_t, self._rev_buf_pose = [], []
                    self._rev_t0 = self.t
            # IMU
            if self.stream_imu and self.t >= self._next_imu:
                self._imu_line(self.t + 0.003)
                self._next_imu += 1.0 / self.p.imu_stream_hz
            while self._pending_imu_req and self._pending_imu_req[0] <= self.t:
                self._imu_line(self._pending_imu_req.pop(0))
            if self.feedback_on and self.t >= self._next_fb:
                self._fb_line(self.t)
                self._next_fb += 1.0 / self.p.feedback_hz
            if self.t >= self._next_truth:
                self.truth.append((self.t, self.pose.copy()))
                self._next_truth += 0.02
        self._pt_acc = acc


def wrap_np(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ─────────────────────────────────────────────────────────────────────────────
#  Scripted runs (offline log generation)
# ─────────────────────────────────────────────────────────────────────────────

class _WaypointDriver:
    """Drives the sim along waypoints using *true* pose (sim only), with
    rotate-in-place corners: the classic manual square-loop pattern."""

    def __init__(self, wps, fwd, turn, tol=0.06):
        self.wps, self.fwd, self.turn, self.tol = wps, fwd, turn, tol
        self.i = 0
        self.pause_until = None
        self.final_yaw = None

    def __call__(self, t, pose):
        if self.pause_until is not None:
            if t < self.pause_until:
                return 0.0, 0.0
            self.pause_until = None
        if self.i >= len(self.wps):
            if self.final_yaw is not None:
                e = wrap(self.final_yaw - pose[2])
                if abs(e) > math.radians(0.8):
                    d = min(self.turn, max(0.07, abs(e) * 0.4))
                    d = d if e > 0 else -d
                    return -d, d
            return None
        gx, gy = self.wps[self.i]
        dx, dy = gx - pose[0], gy - pose[1]
        dist = math.hypot(dx, dy)
        if dist < self.tol:
            self.i += 1
            self.pause_until = t + 0.4
            return 0.0, 0.0
        e = wrap(math.atan2(dy, dx) - pose[2])
        if abs(e) > math.radians(8):
            d = min(self.turn, max(0.12, abs(e) * 0.5))
            d = d if e > 0 else -d
            return -d, d
        steer = max(-0.08, min(0.08, e * 0.6))
        f = self.fwd if dist > 0.25 else max(0.12, self.fwd * dist / 0.25)
        return f - steer, f + steer


def _spin_script(duty, turns=2.0):
    seq = []
    seq.append((1.5, (0.0, 0.0)))
    seq.append((None, ("spin", +duty, turns * 2 * math.pi)))
    seq.append((1.0, (0.0, 0.0)))
    seq.append((None, ("spin", -duty, turns * 2 * math.pi)))
    seq.append((1.5, (0.0, 0.0)))
    return seq


def generate_run(pattern: str, out_root: str, params: SimParams | None = None, tag: str | None = None,
                 imu_poll: bool = False) -> str:
    """Generate a synthetic run directory in the real logger format.
    pattern: handrot | spin | slow_square | fast_square | wander
    """
    from phase3_onboard.logger import RunLogger
    p = params or SimParams()
    sim = SimCore(p)
    sim.stream_imu = not imu_poll
    sim.feedback_on = True
    t_offset = 1000.0   # make timestamps look like monotonic()
    lg = RunLogger(out_root, tag or f"sim_{pattern}", meta={"sim": True, "pattern": pattern,
                                                            "sim_params": p.__dict__})
    period = 0.02
    still = 2.0   # standstill at start for gyro bias

    def send(L, R):
        line = json.dumps({"T": 1, "L": round(L, 3), "R": round(R, 3)}, separators=(",", ":"))
        sim.handle_tx(line)
        lg.event("tx", t=sim.t + t_offset, line=line)

    def flush():
        for s in sim.scans:
            s.t_start += t_offset
            s.t_end += t_offset
            s.t_recv += t_offset
            lg.scan(s)
        for t, line in sim.lines:
            lg.event("rx", t=t + t_offset, line=line)
        for t, pose in sim.truth:
            lg.event("truth", t=t + t_offset, pose=[round(float(v), 5) for v in pose])
        sim.scans.clear(); sim.lines.clear(); sim.truth.clear()

    # standstill
    while sim.t < still:
        send(0, 0)
        sim.advance(period)
        flush()

    x0, y0, _ = p.start_pose
    if pattern == "handrot":
        # rover lifted/rotated by hand: +90 deg, hold, -90 deg (smooth profile)
        def prof(t):
            tt = t - still
            def bump(t0, dur, ang):
                if t0 <= tt <= t0 + dur:
                    return ang * (math.pi / dur) * 0.5 * math.sin(math.pi * (tt - t0) / dur)
                return 0.0
            return 0.0, bump(0.0, 3.0, math.pi / 2) + bump(5.0, 3.0, -math.pi / 2)
        sim.override = prof
        while sim.t < still + 10.0:
            sim.advance(period)
            flush()
        sim.override = None
    elif pattern == "spin":
        duty = 0.35
        for dur, act in _spin_script(duty):
            if dur is not None:
                t_end = sim.t + dur
                while sim.t < t_end:
                    send(*act)
                    sim.advance(period)
                    flush()
            else:
                _, d, total = act
                y_acc, last = 0.0, sim.pose[2]
                while y_acc < total:
                    send(-d, d)
                    sim.advance(period)
                    y_acc += abs(wrap(sim.pose[2] - last))
                    last = sim.pose[2]
                    flush()
    elif pattern in ("slow_square", "fast_square", "wander"):
        side = 1.6
        if pattern == "wander":
            wps = [(x0 + 1.8, y0), (x0 + 1.8, y0 + 1.5), (x0 + 0.4, y0 + 2.2), (x0 + 0.2, y0 + 0.7),
                   (x0 + 1.2, y0 + 0.9), (x0, y0)]
            fwd, turn, loops = 0.30, 0.30, 1
        else:
            wps = [(x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side), (x0, y0)]
            fwd, turn = (0.18, 0.15) if pattern == "slow_square" else (0.42, 0.38)
            loops = 2
        drv = _WaypointDriver(wps * loops, fwd, turn)
        drv.final_yaw = p.start_pose[2]
        while True:
            u = drv(sim.t, sim.pose)
            if u is None:
                break
            send(*u)
            sim.advance(period)
            flush()
            if sim.t > 400:
                break
    else:
        raise ValueError(pattern)

    end = sim.t + 1.5
    while sim.t < end:
        send(0, 0)
        sim.advance(period)
        flush()
    lg.close()
    return lg.dir


# ─────────────────────────────────────────────────────────────────────────────
#  Live backend (real time) for main.py --sim
# ─────────────────────────────────────────────────────────────────────────────

class LiveSim:
    """Runs SimCore in real time on a thread.  Provides a scan queue and an rx
    queue, and accepts tx lines — used by SimTransport and SimLidarSource."""

    def __init__(self, params: SimParams | None = None):
        self.core = SimCore(params)
        self.core.stream_imu = False   # behave like the real firmware: reply to T:126
        self.scan_q: queue.Queue = queue.Queue(maxsize=50)
        self.rx_q: queue.Queue = queue.Queue(maxsize=5000)
        self._lock = threading.Lock()
        self._running = False
        self._t0 = None
        self.truth_pose = self.core.pose.copy()

    def start(self):
        if self._running:
            return
        self._running = True
        self._t0 = time.monotonic()
        threading.Thread(target=self._loop, daemon=True, name="sim").start()

    def stop(self):
        self._running = False

    def tx(self, line: str):
        with self._lock:
            self.core.handle_tx(line)

    def _loop(self):
        step = 0.01
        next_t = time.monotonic()
        while self._running:
            with self._lock:
                self.core.advance(step)
                off = self._t0
                for s in self.core.scans:
                    s.t_start += off; s.t_end += off; s.t_recv += off
                    try:
                        self.scan_q.put_nowait(s)
                    except queue.Full:
                        pass
                for t, line in self.core.lines:
                    try:
                        self.rx_q.put_nowait((t + off, line))
                    except queue.Full:
                        pass
                self.core.scans.clear(); self.core.lines.clear(); self.core.truth.clear()
                self.truth_pose = self.core.pose.copy()
            next_t += step
            d = next_t - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                next_t = time.monotonic()


_LIVE: Optional[LiveSim] = None


def live_instance(params: SimParams | None = None) -> LiveSim:
    global _LIVE
    if _LIVE is None:
        _LIVE = LiveSim(params)
    return _LIVE

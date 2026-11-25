"""
Obstacle guard: the last check on EVERY motion command before it reaches the motors.
================================================================================
Installed by wrapping RoverLink.set_intent, so manual (keys / GUI), autonomous goals,
console `spin` / `fwd` tests and the stuck back-up all pass through it.  stop_now() is
never touched.

For each command it predicts where the rover's footprint will go and how far it can
travel before touching anything seen by the LiDAR:
  * straight / curved driving (forward or reverse): the footprint rectangle is swept
    along the commanded arc up to GUARD_LOOKAHEAD_M;
  * turning in place: the footprint is rotated up to GUARD_SPIN_DEG (corners swing out).
Then:
  free < GUARD_STOP_M                 -> that motion is blocked (a pure turn is still
                                         allowed if the turn itself is clear)
  GUARD_STOP_M .. GUARD_SLOW_M        -> speed ramps down (never below GUARD_MIN_DUTY,
                                         which keeps the heavy rover actually moving)
  autonomous goals only: when the path ahead is short, nearby curvatures are tried and
  the one closest to the follower's choice that is clear is used (steer around), and
  the planner is asked to re-plan.
Spinning in place is all-or-nothing (a slower spin just stalls this rover).

Obstacle memory (ObstacleLayer): the G4 sees nothing closer than ~0.26 m, and thin things
(chair legs) can also be wiped from the occupancy map by later scans.  Every point seen
within OBST_KEEP_RADIUS_M is remembered on a 3 cm grid in map coordinates.  A remembered
point is forgotten only when (a) the LiDAR has looked straight through it OBST_CLEAR_MISSES
scans in a row (beam returns from further away in that direction and both neighbours), or
(b) it has not been seen for OBST_TTL_S (OBST_TTL_BLOCK_S for points that stopped the rover).
Points inside the blind zone cannot be looked through, so they stay until the TTL.

Re-routing (autonomous goals): when the guard blocks or steers, the points in front of the
rover are marked as "blocking", and an immediate re-plan is forced.  The planner is wrapped
so every plan sees the remembered points as occupied cells, so the new path goes AROUND
them instead of back through them.  A successful forced re-plan resets the stuck monitor,
so the rover follows the new path instead of backing up.  Fallbacks if the planner finds
nothing: plan with points near the rover left out, then the plain map.
If no new scan has been processed for GUARD_STALE_S, all motion is blocked.

Robot frame: x forward, y left, origin at the LiDAR (LIDAR_MOUNT_* are 0 on this rover).
Footprint is measured from the LiDAR centre and must include the payload:
  ROBOT_FRONT_M, ROBOT_BACK_M, ROBOT_HALF_WIDTH_M
  (+ GUARD_MARGIN_FRONT_M in front and behind, + GUARD_MARGIN_SIDE_M left and right)
Stopping distance grows with speed: the scan is up to GUARD_LATENCY_S old when the motors
react, and the rover needs v^2 / (2 GUARD_BRAKE_MPS2) to stop, so both are added to the
stop / slow distances (free_eff = free - v*latency - v^2/2a).
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque

import numpy as np

log = logging.getLogger("guard")

DEFAULTS = dict(
    GUARD_ENABLED=True,
    ROBOT_FRONT_M=0.15,
    ROBOT_BACK_M=0.15,
    ROBOT_HALF_WIDTH_M=0.12,
    GUARD_MARGIN_FRONT_M=0.05,  # added in front and behind
    GUARD_MARGIN_SIDE_M=0.0,    # added left and right (the planner keeps PLAN_MARGIN_M on the sides)
    GUARD_LATENCY_S=0.15,       # scan age + SLAM + nav period + motor response
    GUARD_BRAKE_MPS2=0.6,       # deceleration of the loaded rover after the duty drops (motors coast)
    GUARD_MIN_RANGE_M=0.24,     # stop with the obstacle >= this far from the LiDAR after braking (G4 blind < 0.26 m)
    GUARD_SPEED_TAU_S=0.35,     # how slowly the real speed follows the command
    GUARD_SPIN_DUTY=0.45,       # auto, blocked: turn in place toward the path at this duty
    GUARD_MPS_PER_DUTY=0.8,     # speed per unit duty (tools.calibrate_motion: 0.81)
    GUARD_STAT_HZ=5.0,          # 'gstat' log events while moving (diagnostics)
    GUARD_STOP_M=0.04,
    GUARD_SLOW_M=0.20,
    GUARD_LOOKAHEAD_M=0.45,
    GUARD_MIN_DUTY=0.16,
    GUARD_SPIN_DEG=40.0,
    GUARD_KAPPA_GAIN=5.0,      # curvature 1/m per unit (R-L)/(R+L); ~2/track, reduced for skid
    GUARD_SLIP=0.0,            # skid steer turns far less than commanded: also sweep kappa*GUARD_SLIP (0 = straight)
    GUARD_SPIN_MARGIN_M=0.03,  # spinning in place: margin around the swinging corners
    GUARD_STEER_AUTO=False,    # the planner re-routes instead (steering fought the path follower)
    GUARD_STALE_S=0.6,
    GUARD_MEMORY_RADIUS_M=0.35, # the guard uses remembered points only this close: where the LiDAR
                                #   is blind; beyond it the live scan is trusted (memory smears with pose error)
    DEADEND_CONFIRM_S=0.4,      # dead end must hold this long before backing up
    OBST_ENABLED=True,
    OBST_CELL_M=0.03,
    OBST_KEEP_RADIUS_M=1.5,     # only points this close are remembered
    OBST_TTL_S=15.0,
    OBST_TTL_BLOCK_S=40.0,
    OBST_CLEAR_MISSES=4,
    OBST_CLEAR_MIN_R_M=0.32,    # closer than this: never cleared by see-through (blind zone)
    OBST_FORCE_REPLAN_S=1.0,    # at most one forced re-plan per this many seconds
    GUARD_BLOCK_REPLAN_S=0.15,  # auto: re-plan after being blocked this long
    GUARD_HOLD_STUCK_S=3.0,     # auto: no stuck back-up for this long after the guard acted
    GUARD_REROUTE_TRIES=3,      # back-up allowed only after this many re-routes in a row did not help
    PLANNER_V2=True,            # nav/planner2.ClearancePlanner instead of the stock planner
    GUARD_HEADON_ONLY=True,     # slow / stop only for things in the rover's straight-ahead lane
    FOLLOW_SKID=True,           # auto: use the follower below instead of the stock one
    FOLLOW_MODE="smooth",       # smooth: steer along curves (yaw-rate feedback); skid: straight + spin at corners
    FOLLOW_SMOOTH_LOOKAHEAD_M=0.45,
    FOLLOW_SMOOTH_SPIN_DEG=70.0,    # smooth mode: turn in place only if the path is more than this off
    FOLLOW_SMOOTH_SPIN_STOP_DEG=20.0,
    FOLLOW_TURN_KFF=0.6,        # turn duty per rad/s wanted (feed-forward); the integral adds what skid eats
    FOLLOW_TURN_KI=1.0,         # turn duty per rad of accumulated yaw-rate shortfall
    FOLLOW_TURN_MAX=0.45,       # max turn duty while driving
    FOLLOW_TURN_SLEW=0.06,      # max change of turn duty per nav step (smooth steering)
    FOLLOW_LOOKAHEAD_M=0.30,
    FOLLOW_SPIN_START_DEG=30.0, # turn in place when the path ahead is more than this off the heading
    CREEP_M=0.10,               # cannot spin here: move this far (forward if clear, else back) and try again
    FOLLOW_SPIN_STOP_DEG=6.0,   # ...until it is within this
    DEADEND_SIDE_FREE_M=0.35,   # a side is "open" if, after turning 35/60/90 deg, the lane is free this far
    DEADEND_REPEAT_M=0.35,      # blocked again within this distance of an earlier block (after a re-route)
    DEADEND_REPEAT_S=40.0,      #   ... within this time -> treated as a dead end too
    ESCAPE_BACK_M=0.30,         # dead end: back up this far (rear checked by the guard)
    ESCAPE_DUTY=0.30,
    ESCAPE_MAX_S=3.0,
    OBST_TTL_VIRTUAL_S=60.0,    # dead-end marks: the planner avoids that gap for this long
    PATH_SWITCH_GAIN=0.85,      # keep the current path unless a new one is < 85 % as long
    PATH_CHECK_S=0.3,           # auto: how often the path ahead is checked against new obstacles
)


class ObstacleLayer:
    """Short-term memory of obstacle points (map frame), with see-through clearing."""

    def __init__(self, P):
        g = lambda k: getattr(P, k, DEFAULTS[k])
        self.cell = g("OBST_CELL_M")
        self.keep_r = g("OBST_KEEP_RADIUS_M")
        self.ttl = g("OBST_TTL_S")
        self.ttl_block = g("OBST_TTL_BLOCK_S")
        self.misses = int(g("OBST_CLEAR_MISSES"))
        self.clear_min_r = g("OBST_CLEAR_MIN_R_M")
        self.K = np.zeros(0, np.int64)          # cell keys
        self.XY = np.zeros((0, 2))              # world points
        self.T = np.zeros(0)                    # last seen
        self.MISS = np.zeros(0, np.int16)
        self.BLOCK = np.zeros(0, bool)
        self.VXY = np.zeros((0, 2))             # dead-end marks (planner only, never block the rover)
        self.VEND = np.zeros(0)                 # expiry time
        self.lock = threading.Lock()
        self.version = 0

    def _keys(self, w):
        q = np.round(w / self.cell).astype(np.int64)
        return (q[:, 0] + (1 << 20)) * (1 << 21) + (q[:, 1] + (1 << 20))

    @staticmethod
    def _to_robot(w, pose):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return (w - pose[:2]) @ np.array([[c, -s], [s, c]])

    def update(self, pts_robot, pose, now):
        pose = np.asarray(pose, float)
        r2 = (pts_robot ** 2).sum(1)
        near = pts_robot[(r2 > 1e-4) & (r2 < self.keep_r ** 2)]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        w = near @ np.array([[c, s], [-s, c]]) + pose[:2]
        with self.lock:
            seen = np.zeros(len(self.K), bool)
            if len(w):
                k = self._keys(w)
                uk, ui = np.unique(k, return_index=True)
                hit = np.isin(self.K, uk)
                seen |= hit
                self.T[hit] = now
                self.MISS[hit] = 0
                new = ~np.isin(uk, self.K)
                if new.any():
                    self.K = np.concatenate([self.K, uk[new]])
                    self.XY = np.concatenate([self.XY, w[ui[new]]])
                    self.T = np.concatenate([self.T, np.full(new.sum(), now)])
                    self.MISS = np.concatenate([self.MISS, np.zeros(new.sum(), np.int16)])
                    self.BLOCK = np.concatenate([self.BLOCK, np.zeros(new.sum(), bool)])
                    seen = np.concatenate([seen, np.ones(new.sum(), bool)])
            # see-through clearing, 0.5 deg bins of the current scan (all ranges)
            if len(self.K) and len(pts_robot):
                rs = np.sqrt(r2)
                ok = rs > 1e-2
                bs = np.arctan2(pts_robot[ok, 1], pts_robot[ok, 0])
                nb = 720
                binmin = np.full(nb, np.inf)
                np.minimum.at(binmin, ((bs + math.pi) / (2 * math.pi) * nb).astype(int) % nb, rs[ok])
                pr = self._to_robot(self.XY, pose)
                rm = np.hypot(pr[:, 0], pr[:, 1])
                bm = ((np.arctan2(pr[:, 1], pr[:, 0]) + math.pi) / (2 * math.pi) * nb).astype(int) % nb
                thr = rm + 0.12
                through = (binmin[bm] > thr) & (binmin[(bm - 1) % nb] > thr) & (binmin[(bm + 1) % nb] > thr)
                through &= np.isfinite(binmin[bm]) & np.isfinite(binmin[(bm - 1) % nb]) \
                    & np.isfinite(binmin[(bm + 1) % nb])
                cand = ~seen & (rm > self.clear_min_r) & (rm < 4.0) & through
                self.MISS[cand] += 1
            ttl = np.where(self.BLOCK, self.ttl_block, self.ttl)
            keep = (now - self.T <= ttl) & (self.MISS < self.misses)
            if not keep.all():
                self.K, self.XY, self.T, self.MISS, self.BLOCK = (a[keep] for a in
                                                                (self.K, self.XY, self.T, self.MISS, self.BLOCK))
            self.version += 1

    def near_robot(self, pose, radius):
        with self.lock:
            if not len(self.K):
                return np.zeros((0, 2))
            pr = self._to_robot(self.XY, np.asarray(pose, float))
        return pr[(pr ** 2).sum(1) < radius ** 2]

    def mark_blocking(self, pose, x0, x1, half_w, now):
        """Points in the robot-frame box x0..x1, |y|<=half_w become 'blocking' (longer TTL)."""
        with self.lock:
            if not len(self.K):
                return 0
            pr = self._to_robot(self.XY, np.asarray(pose, float))
            m = (pr[:, 0] >= x0) & (pr[:, 0] <= x1) & (np.abs(pr[:, 1]) <= half_w)
            self.BLOCK |= m
            self.T[m] = now
            return int(m.sum())

    def add_virtual(self, pts_world, now, ttl, max_patches=4):
        with self.lock:
            self._vid = getattr(self, "_vid", 0) + 1
            pts = np.asarray(pts_world, float).reshape(-1, 2)
            self.VXY = np.concatenate([self.VXY, pts])
            self.VEND = np.concatenate([self.VEND, np.full(len(pts), now + ttl)])
            self.VID = np.concatenate([getattr(self, "VID", np.zeros(0, np.int64)), np.full(len(pts), self._vid)])
            ids = np.unique(self.VID)
            if len(ids) > max_patches:                     # keep only the newest dead-end marks
                keep = self.VID > ids[-max_patches - 1]
                self.VXY, self.VEND, self.VID = self.VXY[keep], self.VEND[keep], self.VID[keep]

    def world_points(self, now=None, virtual=True, not_near=None, radius=0.0):
        """Remembered points (+ dead-end marks), optionally without marks within radius of not_near."""
        now = time.monotonic() if now is None else now
        with self.lock:
            if not hasattr(self, "VID"):
                self.VID = np.zeros(len(self.VXY), np.int64)
            keep = self.VEND > now
            self.VXY, self.VEND, self.VID = self.VXY[keep], self.VEND[keep], self.VID[keep]
            v = self.VXY
            if virtual and not_near is not None and len(v):
                v = v[((v - np.asarray(not_near, float)[:2]) ** 2).sum(1) > radius ** 2]
            if not virtual or not len(v):
                return self.XY.copy(), self.BLOCK.copy()
            return (np.concatenate([self.XY, v]), np.concatenate([self.BLOCK, np.ones(len(v), bool)]))

    def clear(self):
        with self.lock:
            self.K, self.XY, self.T = self.K[:0], self.XY[:0], self.T[:0]
            self.MISS, self.BLOCK = self.MISS[:0], self.BLOCK[:0]


class ObstacleGuard:
    def __init__(self, P):
        g = lambda k: getattr(P, k, DEFAULTS[k])
        self.enabled = bool(g("GUARD_ENABLED"))
        mf = getattr(P, "GUARD_MARGIN_FRONT_M", getattr(P, "GUARD_MARGIN_M", DEFAULTS["GUARD_MARGIN_FRONT_M"]))
        ms = getattr(P, "GUARD_MARGIN_SIDE_M", DEFAULTS["GUARD_MARGIN_SIDE_M"])
        self.front = g("ROBOT_FRONT_M") + mf
        self.back = g("ROBOT_BACK_M") + mf
        self.hw = g("ROBOT_HALF_WIDTH_M") + ms
        self.lat = g("GUARD_LATENCY_S")
        self.brake = g("GUARD_BRAKE_MPS2")
        self.mps = g("GUARD_MPS_PER_DUTY")
        self.free_eff = None
        self._vout = 0.0
        self._vout_t = -1e9
        self.stop_m = g("GUARD_STOP_M")
        self.slow_m = g("GUARD_SLOW_M")
        self.look = g("GUARD_LOOKAHEAD_M")
        self.min_duty = g("GUARD_MIN_DUTY")
        self.spin = math.radians(g("GUARD_SPIN_DEG"))
        self.kgain = g("GUARD_KAPPA_GAIN")
        self.slip = g("GUARD_SLIP")
        sm = g("GUARD_SPIN_MARGIN_M")
        self.base_hw = g("ROBOT_HALF_WIDTH_M")
        self.sfront = g("ROBOT_FRONT_M") + sm
        self.sback = g("ROBOT_BACK_M") + sm
        self.shw = self.base_hw + sm
        self.steer_auto = bool(g("GUARD_STEER_AUTO"))
        self.headon = bool(g("GUARD_HEADON_ONLY"))
        self.side_free = g("DEADEND_SIDE_FREE_M")
        self.mem_r = g("GUARD_MEMORY_RADIUS_M")
        self.stale_s = g("GUARD_STALE_S")
        self.layer = ObstacleLayer(P) if g("OBST_ENABLED") else None
        self.max_duty = 0.5
        self.reach = self.look + math.hypot(max(self.front, self.back), self.hw) + 0.05
        self._s = np.arange(0.0, self.look + 1e-9, 0.02)
        self._frames = None
        self._frames_t = 0.0
        self._pts = np.zeros((0, 2))
        self._pose = np.zeros(3)
        self._scan_near = np.zeros((0, 2))
        self.trapped = False
        self.min_range = g("GUARD_MIN_RANGE_M")
        self.tau = g("GUARD_SPEED_TAU_S")
        self._v_cmd_est = 0.0
        self._v_t = None
        self._ph = deque(maxlen=12)
        self.lock = threading.Lock()
        # status for UI / log
        self.state = "clear"
        self.free_m = None
        self.events = 0

    # ── perception ─────────────────────────────────────────────────────────────
    def update(self, snap, now=None):
        """Feed the latest SLAM snapshot (needs pts_robot, pose, frames, lost)."""
        now = time.monotonic() if now is None else now
        pts = np.asarray(snap.get("pts_robot", np.zeros((0, 2))), dtype=float).reshape(-1, 2)
        pose = np.asarray(snap["pose"], dtype=float)
        frames = snap.get("frames")
        lost = bool(snap.get("lost"))
        new = frames != self._frames
        if new:
            self._frames, self._frames_t = frames, now
            if self.layer is not None:
                if not lost:                        # while LOST: keep memory, add nothing
                    self.layer.update(pts, pose, now)
        r2 = (pts ** 2).sum(1)
        near = pts[(r2 > 1e-4) & (r2 < self.reach ** 2)]
        if self.layer is not None and not lost:
            near = np.concatenate([near, self.layer.near_robot(pose, self.mem_r)])
        if new and not lost:
            self._ph.append((now, pose[0], pose[1]))
        with self.lock:
            self._pts = near
            self._scan_near = pts[(r2 > 1e-4) & (r2 < self.reach ** 2)]
            self._pose = pose

    def stale(self, now=None):
        now = time.monotonic() if now is None else now
        return self._frames is None or now - self._frames_t > self.stale_s

    # ── geometry ───────────────────────────────────────────────────────────────
    def _inside(self, xr, yr):
        return (xr <= self.front) & (xr >= -self.back) & (np.abs(yr) <= self.hw)

    def free_along(self, pts, kappa, direction):
        """Arc length the footprint can travel along curvature kappa (direction +1 fwd / -1 rev)."""
        if len(pts) == 0:
            return self.look
        x0, y0 = pts[:, 0], pts[:, 1]
        in0 = self._inside(x0, y0)
        ahead0 = in0 & (x0 * direction > 0)
        if ahead0.any():
            return 0.0
        p = pts[~in0]
        if len(p) == 0:
            return self.look
        s = self._s[1:] * direction
        if abs(kappa) < 1e-6:
            px, py, th = s, np.zeros_like(s), np.zeros_like(s)
        else:
            th = kappa * s
            px, py = np.sin(th) / kappa, (1 - np.cos(th)) / kappa
        c, sn = np.cos(th)[:, None], np.sin(th)[:, None]
        dx, dy = p[None, :, 0] - px[:, None], p[None, :, 1] - py[:, None]
        hit = self._inside(c * dx + sn * dy, -sn * dx + c * dy).any(1)
        if not hit.any():
            return self.look
        i = int(np.argmax(hit))
        return float(self._s[i])            # last clear sample

    def spin_clear(self, pts, sign, angle=None):
        if len(pts) == 0:
            return True
        ins = lambda x, y: (x <= self.sfront) & (x >= -self.sback) & (np.abs(y) <= self.shw)
        p = pts[~ins(pts[:, 0], pts[:, 1])]
        if len(p) == 0:
            return True
        a = np.linspace(0, sign * (self.spin if angle is None else angle), 13)[1:]
        c, s = np.cos(a)[:, None], np.sin(a)[:, None]
        X, Y = p[None, :, 0], p[None, :, 1]
        return not ins(c * X + s * Y, -s * X + c * Y).any()

    def open_sides(self, pts=None):
        """(left_open, right_open): can the rover turn 35/60/90 deg that way and then drive
        DEADEND_SIDE_FREE_M straight?"""
        if pts is None:
            with self.lock:
                pts = self._pts
        res = []
        for sign in (1.0, -1.0):
            ok = False
            for deg in (35.0, 60.0, 90.0):
                a = math.radians(deg) * sign
                if not self.spin_clear(pts, sign, abs(a)):
                    break                              # cannot turn further that way either
                c, s = math.cos(a), math.sin(a)
                rp = pts @ np.array([[c, -s], [s, c]])    # points in the turned robot frame
                if self.free_along(rp, 0.0, 1.0) >= min(self.side_free, self.look):
                    ok = True
                    break
            res.append(ok)
        return res[0], res[1]

    # ── the filter ─────────────────────────────────────────────────────────────
    def filter(self, L, R, source="", now=None):
        now = time.monotonic() if now is None else now
        L2, R2, st = self._filter(L, R, source, now)
        self._track_cmd((L2 + R2) / 2.0, now)
        self._vout = (L2 + R2) / 2.0            # what the rover is now being told: its speed soon
        self._vout_t = now
        return L2, R2, st

    def speed_now(self, direction, now):
        """Estimated real speed (m/s): the larger of (a) the sent duty passed through a first-order
        lag (the rover does not slow down the instant the duty drops) and (b) SLAM pose speed."""
        v = self._v_cmd_est * direction if self._v_cmd_est * direction > 0 else 0.0
        h = [e for e in self._ph if now - e[0] < 0.6]
        if len(h) >= 2 and h[-1][0] - h[0][0] > 0.2:
            vp = math.hypot(h[-1][1] - h[0][1], h[-1][2] - h[0][2]) / (h[-1][0] - h[0][0])
            v = max(v, min(vp, 0.6))
        return v

    def _track_cmd(self, vout, now):
        if self._v_t is None:
            self._v_t = now
        dt = min(0.5, max(0.0, now - self._v_t))
        self._v_t = now
        a = min(1.0, dt / self.tau)
        self._v_cmd_est += (vout * self.mps - self._v_cmd_est) * a

    def _filter(self, L, R, source="", now=None):
        """Returns (L, R, state) with state in clear / slow / steer / blocked / stale / off."""
        if not self.enabled:
            return L, R, "off"
        if abs(L) < 1e-3 and abs(R) < 1e-3:
            return L, R, "clear"
        if self.stale(now):
            return 0.0, 0.0, "stale"
        with self.lock:
            pts = self._pts
        vd, td = (L + R) / 2.0, (R - L) / 2.0
        # turning in place (or nearly): all-or-nothing
        if abs(vd) < 0.3 * abs(td):
            ok = self.spin_clear(pts, 1.0 if td > 0 else -1.0)
            self.free_m = None
            return (L, R, "clear") if ok else (0.0, 0.0, "blocked")
        d = 1.0 if vd > 0 else -1.0
        kappa = float(np.clip(self.kgain * td / vd, -6.0, 6.0))
        if self.headon:
            free = self.free_along(pts, 0.0, d)      # only the straight lane ahead: side objects never slow it
        else:
            free = self.free_along(pts, kappa, d)
            if abs(kappa) > 1e-6:                    # the real arc is much straighter (skid)
                free = min(free, self.free_along(pts, kappa * self.slip, d))
        state = "clear"
        if free < self.slow_m and source == "auto" and self.steer_auto and d > 0:
            best_k, best_f = kappa, free
            for k in sorted(np.linspace(-4.0, 4.0, 17), key=lambda k: abs(k - kappa)):
                f = self.free_along(pts, k, d)
                if f >= self.slow_m:
                    best_k, best_f = k, f
                    break
                if f > best_f + 0.05:
                    best_k, best_f = k, f
            if best_k != kappa:
                kappa, free, state = best_k, best_f, "steer"
                td = kappa * vd / self.kgain
                lim = max(abs(L), abs(R))
                m = max(abs(vd - td), abs(vd + td))
                if m > lim:
                    vd, td = vd * lim / m, td * lim / m
        self.free_m = free
        v = self.speed_now(d, now if now is not None else time.monotonic())
        free = free - v * self.lat - v * v / (2.0 * self.brake)   # distance left after reacting + braking
        self.free_eff = free
        # stop while the obstacle is still outside the LiDAR blind zone (memory is only a backup)
        stop_m = max(self.stop_m, self.min_range - (self.front if d > 0 else self.back))
        slow_m = stop_m + (self.slow_m - self.stop_m)
        if free <= stop_m:
            self.trapped = not (self.spin_clear(pts, 1.0) or self.spin_clear(pts, -1.0))
            if abs(td) > 1e-3 and self.spin_clear(pts, 1.0 if td > 0 else -1.0):
                return float(-td), float(td), "blocked"   # keep only the turn
            return 0.0, 0.0, "blocked"
        if free < slow_m:
            f = (free - stop_m) / (slow_m - stop_m)
            v2 = max(self.min_duty, abs(vd) * f)
            if v2 < abs(vd):
                td *= v2 / abs(vd)
                vd = d * v2
                if state == "clear":
                    state = "slow"
        return float(vd - td), float(vd + td), state

    def diag(self, direction):
        """Nearest point ahead (or behind) inside the footprint's width: live scan vs memory."""
        with self.lock:
            sc, allp = self._scan_near, self._pts
        def near(p):
            if not len(p):
                return None
            x = p[:, 0] * direction
            m = (np.abs(p[:, 1]) <= self.hw) & (x > 0)
            return None if not m.any() else round(float(x[m].min()), 3)
        return {"scan_ahead": near(sc), "all_ahead": near(allp), "n_scan": int(len(sc)),
                "n_mem": int(len(allp) - len(sc))}

    def as_dict(self):
        return {"state": self.state, "free_m": None if self.free_m is None else round(self.free_m, 2),
                "events": self.events, "points": int(len(self._pts))}


def install_guard(rt):
    """Wrap rt.link.set_intent (every motion command is checked) and rt.planner.plan
    (clearance planner + remembered obstacles + path stability). Returns the guard."""
    P = rt.P
    gp = lambda k: getattr(P, k, DEFAULTS[k])
    guard = ObstacleGuard(P)
    if not guard.enabled:
        rt.note("obstacle guard OFF (GUARD_ENABLED=False)")
        return guard
    link = rt.link
    raw_set_intent = link.set_intent
    last_note = [0.0, "clear"]
    S = {"escape": None, "blocks": [], "reroutes": 0, "spinning": False, "guard_t": -1e9, "tries": 0, "fail": False, "forced_t": -1e9, "pending": None, "blocked_since": None, "check_t": 0.0, "reroute_note_t": 0.0}
    pl = None
    if gp("PLANNER_V2") and getattr(rt, "planner", None) is not None:
        from phase3_onboard.nav.planner2 import ClearancePlanner, path_length, trim_path
        pl = ClearancePlanner(P, guard.base_hw, spin_radius=math.hypot(guard.sfront, guard.shw) + 0.02)
        t0 = time.monotonic()
        pl.warmup()                                   # numba compile (cached after the first run)
        rt.note(f"clearance planner ready ({time.monotonic() - t0:.1f} s): keeps >= {pl.r_hard:.2f} m "
                f"from obstacle centres, prefers >= {pl.r_hard + pl.soft:.2f} m")
    guard.planner2 = pl

    def force_replan(why, now):
        if now - S["forced_t"] < gp("OBST_FORCE_REPLAN_S"):
            return
        S["forced_t"] = now
        S["pending"] = why
        try:
            rt.last_plan_t = -1e9                     # re-plan on the next nav step
        except Exception:
            pass

    def path_blocked_ahead():
        """Does the path ahead (1.2 m) pass closer than the hard clearance to a remembered point?"""
        if pl is None or guard.layer is None:
            return False
        with rt.lock:
            path = list(rt.path)
        if len(path) < 2:
            return False
        pose = guard._pose
        rem, d = trim_path(path, pose[:2])
        P_ = np.asarray([np.asarray(p, float)[:2] for p in rem])
        seg = np.hypot(*np.diff(P_, axis=0).T) if len(P_) > 1 else np.zeros(0)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        P_ = P_[cum <= 1.2]
        if len(P_) < 2:
            return False
        dense = [P_[0]]
        for a, b in zip(P_[:-1], P_[1:]):
            n = max(1, int(np.hypot(*(b - a)) / 0.03))
            dense += [a + (b - a) * k / n for k in range(1, n + 1)]
        dense = np.array(dense)
        mem = guard.layer.near_robot(pose, 1.6)
        if not len(mem):
            return False
        c, s_ = math.cos(pose[2]), math.sin(pose[2])
        memw = mem @ np.array([[c, s_], [-s_, c]]) + pose[:2]
        dmin = np.sqrt(((dense[:, None, :] - memw[None, :, :]) ** 2).sum(2)).min()
        return dmin < pl.r_hard - 0.05

    def turn_to_path():
        """Blocked in auto: turn in place toward the path ~0.4 m ahead, at a duty that really
        turns this rover, if the corners can swing clear.  Returns the signed turn duty or None."""
        if pl is None:
            return None
        with rt.lock:
            path = list(rt.path)
        if len(path) < 2:
            return None
        pose = guard._pose
        rem, _ = trim_path(path, pose[:2])
        P_ = np.asarray([np.asarray(p, float)[:2] for p in rem])
        seg = np.hypot(*np.diff(P_, axis=0).T) if len(P_) > 1 else np.zeros(0)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        k = int(np.searchsorted(cum, 0.4))
        tgt = P_[min(k, len(P_) - 1)]
        err = math.atan2(tgt[1] - pose[1], tgt[0] - pose[0]) - pose[2]
        err = math.atan2(math.sin(err), math.cos(err))
        if abs(err) < math.radians(12):
            return None                                 # facing the path and still blocked: re-plan
        sign = 1.0 if err > 0 else -1.0
        with guard.lock:
            pts = guard._pts
        if not guard.spin_clear(pts, sign):
            return None
        return sign * gp("GUARD_SPIN_DUTY")

    def maybe_escape(why, now):
        """Back up only if the dead end persists for DEADEND_CONFIRM_S (one noisy scan is not enough)."""
        if S.get("de_t0") is None or now - S.get("de_last", -1e9) > 0.35:
            S["de_t0"] = now
        S["de_last"] = now
        if now - S["de_t0"] >= gp("DEADEND_CONFIRM_S"):
            S["de_t0"] = None
            start_escape(why, now)
        return S["escape"] is not None

    def start_escape(why, now):
        """Dead end: mark the gap ahead for the planner (not for the guard), back up, re-plan."""
        if S["escape"] is not None:
            return
        pose = guard._pose.copy()
        if guard.layer is not None:
            xs = np.arange(guard.front + 0.02, guard.front + 0.27, 0.04)
            ys = np.arange(-(guard.hw + 0.20), guard.hw + 0.20 + 1e-9, 0.04)
            X, Y = np.meshgrid(xs, ys)
            c, s_ = math.cos(pose[2]), math.sin(pose[2])
            w = np.c_[X.ravel(), Y.ravel()] @ np.array([[c, s_], [-s_, c]]) + pose[:2]
            guard.layer.add_virtual(w, now, gp("OBST_TTL_VIRTUAL_S"))
        S["escape"] = {"t0": now, "p0": pose, "why": why}
        S["blocks"] = []
        rt.note(f"dead end ({why}): backing up {gp('ESCAPE_BACK_M'):.2f} m, then another route")

    def escape_step(now):
        """Returns the duty while backing up, or None when the back-up is over."""
        e = S["escape"]
        moved = float(np.hypot(*(guard._pose[:2] - e["p0"][:2])))
        if moved >= gp("ESCAPE_BACK_M") or now - e["t0"] > gp("ESCAPE_MAX_S"):
            S["escape"] = None
            S["pending"] = None
            S["forced_t"] = -1e9
            force_replan("dead end", now)
            return None
        return -gp("ESCAPE_DUTY")

    def guarded(L, R, source="", *a, **kw):
        now = time.monotonic()
        esc = None
        cr = S.get("creep")
        if source == "auto" and cr is not None and S["escape"] is None:
            moved = float(np.hypot(*(guard._pose[:2] - cr["p0"][:2])))
            if moved >= gp("CREEP_M") or now - cr["t0"] > 1.2:
                S["creep"] = None
            else:
                L = R = cr["d"] * gp("ESCAPE_DUTY")
        if source == "auto" and S["escape"] is not None:
            try:
                guard.update(rt.eng.snapshot(), now)
            except Exception:
                pass
            esc = escape_step(now)
            if esc is not None:
                L, R = esc, esc
        try:
            guard.update(rt.eng.snapshot(), now)
            L2, R2, st = guard.filter(float(L), float(R), source, now)
        except Exception as e:                      # never let a guard bug drive the rover blind
            log.exception("guard: %s", e)
            L2, R2, st = 0.0, 0.0, "error"
        if S.get("creep") is not None and esc is None and source == "auto":
            if st == "blocked":
                S["creep"] = None
            elif abs(L2 + R2) > 0:
                st = "creep"
        if esc is not None:
            if st == "blocked":                     # something behind too: stop backing up, re-plan
                S["escape"] = None
                force_replan("dead end", now)
            else:
                st = "backup"
        elif source == "auto" and st == "blocked" and abs(L + R) < 0.3 * abs(R - L):
            # turning in place is blocked that way: turn the long way round, or escape if boxed in
            sign = 1.0 if (R - L) > 0 else -1.0
            with guard.lock:
                pts_ = guard._pts
            if guard.spin_clear(pts_, -sign):
                L2, R2, st = float(R), float(L), "turn"
            else:
                fwd_free = guard.free_along(pts_, 0.0, 1.0)
                back_free = guard.free_along(pts_, 0.0, -1.0)
                room = gp("CREEP_M") + max(guard.stop_m, guard.min_range - guard.front)
                if fwd_free >= room or back_free >= gp("CREEP_M") + guard.stop_m:
                    d = 1.0 if fwd_free >= room else -1.0
                    S["creep"] = {"d": d, "p0": guard._pose.copy(), "t0": now}
                    L2 = R2 = d * gp("ESCAPE_DUTY")
                    st = "creep"
                else:
                    maybe_escape("boxed in: cannot turn or move", now)
        elif source == "auto" and st == "blocked" and (L + R) > 0:
            try:
                pos = guard._pose[:2].copy()
                repeat = any(now - t < gp("DEADEND_REPEAT_S") and n < S["reroutes"]
                             and np.hypot(*(pos - q)) < gp("DEADEND_REPEAT_M") for t, q, n in S["blocks"])
                if not S["blocks"] or now - S["blocks"][-1][0] > 1.0:
                    S["blocks"].append((now, pos, S["reroutes"]))
                    S["blocks"] = S["blocks"][-10:]
                left, right = guard.open_sides()
                if not left and not right:
                    maybe_escape("left and right blocked", now)
                elif repeat:
                    maybe_escape("blocked again at the same place after a re-route", now)
                else:
                    t = turn_to_path()
                    if t is not None:
                        L2, R2, st = -t, t, "turn"
            except Exception as e:
                log.exception("blocked handling: %s", e)
        guard.state = st
        if st in ("blocked", "stale", "steer", "error"):
            if st != last_note[1] or now - last_note[0] > 3.0:
                guard.events += 1
                last_note[0], last_note[1] = now, st
                fm = "" if guard.free_m is None else f", free {guard.free_m:.2f} m"
                rt.note(f"obstacle guard: {st} ({source}{fm})")
        elif st == "clear":
            last_note[1] = "clear"
        if source == "auto":
            if st in ("blocked", "turn") or (st == "slow" and guard.free_eff is not None
                                   and guard.free_eff < guard.stop_m + 0.05):
                S["guard_t"] = now                    # the guard is handling an obstacle
            if st in ("blocked", "turn"):
                if S["blocked_since"] is None:
                    S["blocked_since"] = now
                    if guard.layer is not None:
                        d = 1.0 if (L + R) >= 0 else -1.0
                        x0, x1 = (0.0, guard.front + guard.slow_m) if d > 0 else (-guard.back - guard.slow_m, 0.0)
                        guard.layer.mark_blocking(guard._pose, x0, x1, guard.hw + 0.10, now)
                elif now - S["blocked_since"] > gp("GUARD_BLOCK_REPLAN_S"):
                    force_replan("blocked", now)
            else:
                S["blocked_since"] = None
            if now - S["check_t"] > gp("PATH_CHECK_S"):
                S["check_t"] = now
                try:
                    if path_blocked_ahead():
                        force_replan("new obstacle on the path", now)
                except Exception as e:
                    log.exception("path check: %s", e)
        moving = abs(L2) + abs(R2) > 0.05
        if moving and rt.logger and now - S.get("stat_t", 0.0) > 1.0 / gp("GUARD_STAT_HZ"):
            S["stat_t"] = now
            try:
                d = 1.0 if (L2 + R2) >= 0 else -1.0
                dg = guard.diag(d)
                rt.logger.event("gstat", src=source, st=st, out=[round(L2, 3), round(R2, 3)],
                                free=None if guard.free_m is None else round(guard.free_m, 3),
                                feff=None if guard.free_eff is None else round(guard.free_eff, 3), **dg)
            except Exception:
                pass
        # sanity: does "forward" make the rover move forward? (checked over ~1 s of straight driving)
        vd_out = (L2 + R2) / 2.0
        if abs(vd_out) > 0.12 and abs(R2 - L2) < 0.5 * abs(vd_out):
            if S.get("dir_t0") is None:
                S["dir_t0"], S["dir_p0"], S["dir_s"] = now, guard._pose.copy(), np.sign(vd_out)
            elif np.sign(vd_out) != S["dir_s"]:
                S["dir_t0"] = None
            elif now - S["dir_t0"] > 1.0:
                p0 = S["dir_p0"]
                dx = (guard._pose[0] - p0[0]) * math.cos(p0[2]) + (guard._pose[1] - p0[1]) * math.sin(p0[2])
                if dx * S["dir_s"] < -0.05 and not S.get("dir_warned"):
                    S["dir_warned"] = True
                    rt.note("WARNING obstacle guard: commanded forward but the rover moved backward "
                            "(or the reverse) -- the guard is checking the wrong side! Check INVERT_FWD.")
                S["dir_t0"] = None
        else:
            S["dir_t0"] = None
        if rt.logger and (L2 != L or R2 != R):
            try:
                rt.logger.event("guard", st=st, src=source, cmd=[round(L, 3), round(R, 3)],
                                out=[round(L2, 3), round(R2, 3)],
                                free=None if guard.free_m is None else round(guard.free_m, 3))
            except Exception:
                pass
        return raw_set_intent(L2, R2, source, *a, **kw)

    link.set_intent = guarded

    if pl is not None:
        raw_plan = rt.planner.plan

        def plan_v2(lo, res, orig, start, goal, *a, **kw):
            mem = guard.layer.world_points()[0] if guard.layer is not None else None
            mem_real = guard.layer.world_points(virtual=False)[0] if guard.layer is not None else None
            why = S["pending"]
            S["pending"] = None
            try:
                t_pl = time.monotonic()
                st_xy, g_xy = np.asarray(start, float)[:2], np.asarray(goal, float)[:2]
                new = pl.plan(lo, res, orig, st_xy, g_xy, mem)
                if new is None and mem is not None and mem_real is not None and len(mem) != len(mem_real):
                    mem2 = guard.layer.world_points(not_near=st_xy, radius=0.6)[0]
                    if len(mem2) != len(mem):                  # marks right next to the rover may trap it
                        new = pl.plan(lo, res, orig, st_xy, g_xy, mem2)
                    if new is None:
                        new = pl.plan(lo, res, orig, st_xy, g_xy, mem_real)
                        if new is not None:
                            rt.note("no other route: trying the marked dead end again")
                ms = (time.monotonic() - t_pl) * 1e3
                if rt.logger:
                    try:
                        rt.logger.event("plan", ms=round(ms, 1), why=S["pending"] if why is None else why,
                                        ok=new is not None, n_mem=0 if mem is None else int(len(mem)))
                    except Exception:
                        pass
                if ms > 150 and time.monotonic() - S.get("slow_plan_note", 0) > 30:
                    S["slow_plan_note"] = time.monotonic()
                    rt.note(f"planner took {ms:.0f} ms (map {pl.last['h']}x{pl.last['w']} cells)")
            except Exception as e:
                log.exception("planner2 failed, using the stock planner: %s", e)
                return raw_plan(lo, res, orig, start, goal, *a, **kw)
            info = pl.info
            # path stability: keep the current path while it is still clear and not much longer
            with rt.lock:
                old = list(rt.path)
                gold = None if rt.goal is None else np.asarray(rt.goal, float)
            if old and gold is not None and np.allclose(gold, np.asarray(goal, float)[:2], atol=1e-3):
                rem, dist = trim_path(old, np.asarray(start, float)[:2])
                if dist < 0.25 and len(rem) >= 2:
                    mc = pl.path_min_clearance(rem)
                    still_ok = mc is not None and mc >= pl.r_hard - 0.03
                    much_better = new is not None and path_length(new) < gp("PATH_SWITCH_GAIN") * path_length(rem) - 0.05
                    if still_ok and not much_better and why is None:
                        return rem
            if why is not None:
                if new is not None:
                    S["reroutes"] += 1
                    S["tries"] += 1 if why == "blocked" else 0
                    S["fail"] = False
                    try:
                        rt.stuck.reset(rt.eng.pose)       # follow the new path, don't back up
                    except Exception:
                        pass
                    if time.monotonic() - S["reroute_note_t"] > 2.0:
                        S["reroute_note_t"] = time.monotonic()
                        rt.note(f"re-routed ({why}): {info}")
                else:
                    S["fail"] = True
                    rt.note(f"re-route failed ({why}): {info}")
            elif new is None:
                rt.note(f"planner: {info}")
            return new

        rt.planner.plan = plan_v2

    stuck = getattr(rt, "stuck", None)
    if stuck is not None and hasattr(stuck, "check"):
        raw_check = stuck.check

        def check(pose, *a, **kw):
            """Back-ups are run by the guard (dead ends only).  The stock monitor's own back-up
            becomes a dead-end escape too (mark the gap, back up, re-plan), so the rover never
            backs up and then drives into the same place again."""
            now = time.monotonic()
            th = float(np.asarray(pose, float)[2])
            dth = abs(math.atan2(math.sin(th - S.get("chk_th", th)), math.cos(th - S.get("chk_th", th))))
            if S["spinning"] or dth > math.radians(15):    # turning in place is progress, not "stuck"
                S["chk_th"] = th
                try:
                    stuck.reset(pose)
                except Exception:
                    pass
                return None
            if S["escape"] is not None or now - S["guard_t"] < gp("GUARD_HOLD_STUCK_S"):
                try:
                    stuck.reset(pose)
                except Exception:
                    pass
                return None
            r = raw_check(pose, *a, **kw)
            if r == "backup":
                start_escape("no progress (something the LiDAR cannot see?)", now)
                return None
            return r

        stuck.check = check

    follower = getattr(rt, "follower", None)
    if gp("FOLLOW_SKID") and pl is not None and follower is not None and hasattr(follower, "compute"):
        raw_compute = follower.compute

        F = {"t": None, "th": None, "wf": 0.0, "I": 0.0, "td": 0.0, "spin": False}

        def smooth_cmd(pose, P_, cum, a, kw, out):
            """Pure pursuit with yaw-rate feedback: the wanted turn rate is v*curvature; the
            measured rate (SLAM heading) is compared and the shortfall integrated, so the turn
            duty grows until this skid-steer rover really follows the curve."""
            now = time.monotonic()
            th = float(pose[2])
            if F["t"] is not None and now - F["t"] > 1e-3:
                dth = math.atan2(math.sin(th - F["th"]), math.cos(th - F["th"]))
                dt = now - F["t"]
                F["wf"] += (dth / dt - F["wf"]) * min(1.0, dt / 0.25)
            else:
                dt = 0.1
            F["t"], F["th"] = now, th
            Ld = gp("FOLLOW_SMOOTH_LOOKAHEAD_M")
            k = int(np.searchsorted(cum, Ld))
            tgt = P_[min(k, len(P_) - 1)]
            dist = float(np.hypot(*(tgt - pose[:2])))
            if dist < 0.05:
                return out
            alpha = math.atan2(tgt[1] - pose[1], tgt[0] - pose[0]) - th
            alpha = math.atan2(math.sin(alpha), math.cos(alpha))
            spin_on, spin_off = math.radians(gp("FOLLOW_SMOOTH_SPIN_DEG")), math.radians(gp("FOLLOW_SMOOTH_SPIN_STOP_DEG"))
            if abs(alpha) > spin_on or (F["spin"] and abs(alpha) > spin_off):
                F["spin"], F["I"], F["td"] = True, 0.0, 0.0
                S["spinning"] = True
                t = float(getattr(P, "AUTO_TURN_DUTY_MAX", 0.45)) * (1.0 if alpha > 0 else -1.0)
                return (-t, t, "turn") + tuple(out[3:])
            F["spin"] = False
            S["spinning"] = False
            scale = float(a[0]) if a else float(kw.get("scale", 1.0))
            v = float(getattr(P, "AUTO_FWD_DUTY", 0.22)) * scale * max(0.6, math.cos(alpha))
            vm = v * guard.mps
            w_want = vm * 2.0 * math.sin(alpha) / max(dist, 0.2)
            F["I"] = float(np.clip(F["I"] + gp("FOLLOW_TURN_KI") * (w_want - F["wf"]) * dt, -0.3, 0.3))
            if w_want * F["I"] < 0 and abs(w_want) > 0.05:          # turning the other way now: drop it
                F["I"] = 0.0
            td = gp("FOLLOW_TURN_KFF") * w_want + F["I"]
            tmax = gp("FOLLOW_TURN_MAX")
            td = float(np.clip(td, -tmax, tmax))
            sl = gp("FOLLOW_TURN_SLEW")
            td = float(np.clip(td, F["td"] - sl, F["td"] + sl))
            F["td"] = td
            return (v - td, v + td, "drive") + tuple(out[3:])

        def compute(pose, *a, **kw):
            """Skid-steer path following: straight lines, sharp turns in place at the corners.
            The stock follower still decides when the goal is reached."""
            out = raw_compute(pose, *a, **kw)
            L, R, phase = out[0], out[1], out[2]
            if phase == "arrived":
                S["spinning"] = False
                return out
            with rt.lock:
                path = list(rt.path)
            if len(path) < 2:
                return out
            pose = np.asarray(pose, float)
            rem, _ = trim_path(path, pose[:2])
            P_ = np.asarray([np.asarray(p, float)[:2] for p in rem])
            if len(P_) < 2:
                return out
            seg = np.hypot(*np.diff(P_, axis=0).T)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            k = int(np.searchsorted(cum, gp("FOLLOW_LOOKAHEAD_M")))
            tgt = P_[min(k, len(P_) - 1)]
            if np.hypot(*(tgt - pose[:2])) < 0.05:
                return out
            err = math.atan2(tgt[1] - pose[1], tgt[0] - pose[0]) - pose[2]
            err = math.atan2(math.sin(err), math.cos(err))
            if gp("FOLLOW_MODE") == "smooth":
                return smooth_cmd(pose, P_, cum, a, kw, out)
            start, stop = math.radians(gp("FOLLOW_SPIN_START_DEG")), math.radians(gp("FOLLOW_SPIN_STOP_DEG"))
            if abs(err) > start or (S["spinning"] and abs(err) > stop):
                S["spinning"] = True
                t = float(getattr(P, "AUTO_TURN_DUTY_MAX", 0.45)) * (1.0 if err > 0 else -1.0)
                return (-t, t, "turn") + tuple(out[3:])
            S["spinning"] = False
            scale = float(a[0]) if a else float(kw.get("scale", 1.0))
            v = float(getattr(P, "AUTO_FWD_DUTY", 0.22)) * scale
            tc = float(np.clip(0.8 * err * v, -0.08, 0.08))      # small trim while driving straight
            return (v - tc, v + tc, "drive") + tuple(out[3:])

        follower.compute = compute

    f = guard
    rt.note(f"obstacle guard ON: footprint +{f.front:.2f}/-{f.back:.2f} x +/-{f.hw:.2f} m (incl. margin), "
            f"stop {f.stop_m:.2f} m, slow {f.slow_m:.2f} m; obstacle memory "
            f"{'ON' if f.layer is not None else 'OFF'}; planner {'v2' if pl is not None else 'stock'}")
    return guard

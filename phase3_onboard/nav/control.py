"""
Path following and safety, all in DUTY units (the rover has no encoders).

PathFollower : pure pursuit with rotate-in-place for big heading errors.
               Output (L, R) duty.  Turn duty capped so the matcher keeps up.
Proximity    : LiDAR-sector minimum ranges from the latest raw scan points;
               blocks forward (or reverse) motion that would close a gap < stop.
StuckMonitor : no progress while commanding motion -> back up, replan, give up.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from phase3_onboard.common import wrap


def mix(fwd: float, turn: float, lim: float) -> tuple:
    """fwd, turn in duty; +turn = CCW (right side faster)."""
    L = fwd - turn
    R = fwd + turn
    m = max(abs(L), abs(R))
    if m > lim:
        L, R = L * lim / m, R * lim / m
    return L, R


class PathFollower:
    def __init__(self, P):
        self.P = P
        self.path: list = []
        self.idx = 0

    def set_path(self, path):
        self.path = [np.asarray(p, dtype=float) for p in (path or [])]
        self.idx = 0

    def done(self, pose) -> bool:
        if not self.path:
            return True
        return math.hypot(*(self.path[-1] - pose[:2])) < self.P.GOAL_TOLERANCE_M

    def compute(self, pose, speed_scale: float = 1.0):
        P = self.P
        if not self.path:
            return 0.0, 0.0, "idle"
        pos = np.asarray(pose[:2])
        while self.idx < len(self.path) - 1 and math.hypot(*(self.path[self.idx] - pos)) < P.LOOKAHEAD_M:
            self.idx += 1
        goal = self.path[-1]
        dist_goal = math.hypot(*(goal - pos))
        if dist_goal < P.GOAL_TOLERANCE_M:
            return 0.0, 0.0, "arrived"
        target = self.path[self.idx]
        alpha = wrap(math.atan2(target[1] - pos[1], target[0] - pos[0]) - pose[2])
        tmax = P.AUTO_TURN_DUTY_MAX
        if abs(alpha) > math.radians(P.ROTATE_FIRST_DEG):
            t = min(tmax, max(P.AUTO_TURN_DUTY_MIN, P.AUTO_TURN_GAIN * abs(alpha)))
            t = t if alpha > 0 else -t
            return -t, t, "rotate"
        fwd = P.AUTO_FWD_DUTY * speed_scale
        if dist_goal < 0.5:
            fwd *= max(0.55, dist_goal / 0.5)
        turn = max(-tmax * 0.6, min(tmax * 0.6, P.AUTO_TURN_GAIN * alpha * 1.2))
        L, R = mix(fwd, turn, P.MAX_DUTY)
        return L, R, "track"


class Proximity:
    def __init__(self, P):
        self.P = P
        self.front = math.inf
        self.rear = math.inf
        self.left = math.inf
        self.right = math.inf

    def update(self, pts_robot: Optional[np.ndarray]):
        if pts_robot is None or len(pts_robot) == 0:
            self.front = self.rear = self.left = self.right = math.inf
            return
        a = np.degrees(np.arctan2(pts_robot[:, 1], pts_robot[:, 0]))
        d = np.hypot(pts_robot[:, 0], pts_robot[:, 1])
        h = self.P.PROX_SECTOR_HALF_DEG

        def sector(c):
            m = np.abs((a - c + 180) % 360 - 180) <= h
            return float(d[m].min()) if m.any() else math.inf
        self.front, self.left, self.rear, self.right = sector(0), sector(90), sector(180), sector(-90)

    def limit(self, L: float, R: float):
        """Remove the forward/backward component that would hit something;
        keep the turning component.  Returns (L, R, blocked_flag)."""
        P = self.P
        fwd = 0.5 * (L + R)
        turn = 0.5 * (R - L)
        blocked = False
        if fwd > 0:
            if self.front < P.PROX_STOP_M:
                fwd, blocked = 0.0, True
            elif self.front < P.PROX_SLOW_M:
                fwd *= max(0.3, (self.front - P.PROX_STOP_M) / (P.PROX_SLOW_M - P.PROX_STOP_M))
        elif fwd < 0:
            if self.rear < P.PROX_STOP_M:
                fwd, blocked = 0.0, True
        return fwd - turn, fwd + turn, blocked

    def as_dict(self):
        f = lambda v: None if not math.isfinite(v) else round(v, 3)
        return {"front": f(self.front), "left": f(self.left), "rear": f(self.rear), "right": f(self.right)}


class StuckMonitor:
    def __init__(self, P):
        self.P = P
        self.reset(None)

    def reset(self, pose):
        self.ref = None if pose is None else np.array(pose, dtype=float)
        self.t_ref = time.monotonic()
        self.attempts = 0
        self.backup_until = 0.0

    def check(self, pose, commanding_motion: bool) -> str:
        """Returns 'ok' | 'backup' | 'give_up'."""
        now = time.monotonic()
        if now < self.backup_until:
            return "backup"
        if self.ref is None or not commanding_motion:
            self.ref, self.t_ref = np.array(pose, dtype=float), now
            return "ok"
        moved = math.hypot(pose[0] - self.ref[0], pose[1] - self.ref[1])
        turned = abs(wrap(pose[2] - self.ref[2]))
        if moved > 0.05 or turned > math.radians(10):
            self.ref, self.t_ref = np.array(pose, dtype=float), now
            return "ok"
        if now - self.t_ref > self.P.STUCK_TIMEOUT_S:
            self.attempts += 1
            self.ref, self.t_ref = np.array(pose, dtype=float), now
            if self.attempts > self.P.STUCK_MAX_ATTEMPTS:
                return "give_up"
            self.backup_until = now + self.P.STUCK_BACKUP_S
            return "backup"
        return "ok"

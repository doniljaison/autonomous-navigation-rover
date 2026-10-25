"""
Navigation Utilities — A* Planner, Pure-Pursuit, PreconnectPool
===============================================================
Reusable building blocks for autonomous path planning and motor control.
No entry point — imported by hybrid.py.
"""

import json
import math
import heapq
import socket
import threading
import logging
import urllib.parse
from typing import Optional

import numpy as np

from config import (
    ROVER_IP, ROVER_PORT, ROVER_ENDPOINT,
    MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED,
    GOAL_TOLERANCE_M, LOOKAHEAD_M,
    INFLATION_RADIUS_M,
)

log = logging.getLogger("nav")

# ─────────────────────────────────────────────────────────────────────────────
#  HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_http_request(ip: str, json_str: str) -> bytes:
    """Build a raw HTTP/1.0 GET request for the rover's /js endpoint."""
    encoded = urllib.parse.quote(json_str, safe="")
    req = (
        f"GET {ROVER_ENDPOINT}?json={encoded} HTTP/1.0\r\n"
        f"Host: {ip}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    )
    return req.encode("ascii")


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


# ─────────────────────────────────────────────────────────────────────────────
#  PreconnectPool — fire-and-forget TCP for low-latency motor commands
# ─────────────────────────────────────────────────────────────────────────────

class PreconnectPool:
    """
    Maintains a single pre-connected TCP socket to the rover.
    Each send() uses the socket then immediately reconnects in background,
    giving minimal latency for the next command.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._reconnect()

    def _reconnect(self):
        threading.Thread(target=self._do_connect, daemon=True).start()

    def _do_connect(self):
        try:
            s = socket.create_connection((self.host, self.port), timeout=2.0)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                if self._sock:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                self._sock = s
        except Exception:
            pass

    def send(self, data: bytes) -> bool:
        with self._lock:
            s, self._sock = self._sock, None

        if s is None:
            self._reconnect()
            return False

        try:
            s.sendall(data)
        except Exception:
            pass
        finally:
            try:
                s.close()
            except Exception:
                pass

        self._reconnect()
        return True

    def close(self):
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None


# ─────────────────────────────────────────────────────────────────────────────
#  A* Planner (8-connected, with obstacle inflation)
# ─────────────────────────────────────────────────────────────────────────────

class AStarPlanner:
    def __init__(self, inflation_cells: int):
        self.inflation = max(0, inflation_cells)

    def plan(self, grid, start_world: np.ndarray, goal_world: np.ndarray) -> Optional[list]:
        # Try with inflation first, fallback to no inflation
        path = self._plan_inner(grid, start_world, goal_world, self.inflation)
        if path is not None:
            return path
        if self.inflation > 0:
            log.info("A*: retrying without inflation...")
            return self._plan_inner(grid, start_world, goal_world, 0)
        return None

    def _plan_inner(self, grid, start_world, goal_world, inflate) -> Optional[list]:
        from scipy.ndimage import binary_dilation

        n, orig, res = grid.n, grid.orig, grid.res

        # Only treat STRONGLY occupied cells as obstacles (threshold 0.5 not 0.3)
        # This filters out noisy single-hit readings
        occ = (grid.log > 0.5).astype(np.uint8)

        if inflate > 0:
            r = inflate
            y_, x_ = np.ogrid[-r:r+1, -r:r+1]
            struct = (x_*x_ + y_*y_ <= r*r)
            occ = binary_dilation(occ, structure=struct).astype(bool)
        else:
            occ = occ.astype(bool)

        def w2c(wx, wy):
            return int(round(wx / res)) + orig, int(round(wy / res)) + orig

        def c2w(cx, cy):
            return np.array([(cx - orig) * res, (cy - orig) * res])

        sx, sy = w2c(*start_world[:2])
        gx, gy = w2c(*goal_world[:2])

        if not (0 <= sx < n and 0 <= sy < n):
            log.warning("A*: start (%d,%d) is outside grid.", sx, sy)
            return None
        if not (0 <= gx < n and 0 <= gy < n):
            log.warning("A*: goal (%d,%d) is outside grid.", gx, gy)
            return None

        # If start cell is occupied, nudge to nearest free cell
        if occ[sy, sx]:
            sx, sy = self._nudge(occ, sx, sy, n, 6)
            if sx is None:
                log.warning("A*: start stuck in obstacles.")
                return None

        # If goal cell is occupied, nudge to nearest free cell
        if occ[gy, gx]:
            gx, gy = self._nudge(occ, gx, gy, n, 10)
            if gx is None:
                log.warning("A*: goal stuck in obstacles.")
                return None

        SQRT2 = math.sqrt(2)
        NEIGHBOURS = ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))

        open_ = [(0.0, (sx, sy))]
        came: dict = {}
        g_cost: dict = {(sx, sy): 0.0}

        while open_:
            _, cur = heapq.heappop(open_)
            cx, cy = cur
            if cx == gx and cy == gy:
                path = []
                node = (gx, gy)
                while node in came:
                    path.append(c2w(*node))
                    node = came[node]
                path.append(c2w(sx, sy))
                path.reverse()
                return path
            for dx, dy in NEIGHBOURS:
                nx_, ny_ = cx+dx, cy+dy
                if not (0 <= nx_ < n and 0 <= ny_ < n):
                    continue
                if occ[ny_, nx_]:
                    continue
                step = SQRT2 if (dx and dy) else 1.0
                new_g = g_cost[cur] + step
                nb = (nx_, ny_)
                if new_g < g_cost.get(nb, 1e18):
                    came[nb] = cur
                    g_cost[nb] = new_g
                    f = new_g + math.hypot(nx_-gx, ny_-gy)
                    heapq.heappush(open_, (f, nb))

        log.warning("A*: no path found from (%.2f,%.2f) to (%.2f,%.2f) (inflate=%d).",
                    *start_world[:2], *goal_world[:2], inflate)
        return None

    @staticmethod
    def _nudge(occ, x, y, n, max_r):
        """Find nearest free cell within max_r of (x, y)."""
        for r in range(1, max_r + 1):
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    nx_, ny_ = x+dx, y+dy
                    if 0 <= nx_ < n and 0 <= ny_ < n and not occ[ny_, nx_]:
                        return nx_, ny_
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Simple rotate-and-drive path tracker
#  Step 1: Spin in place to face the next waypoint
#  Step 2: Drive straight to it
#  Repeat for each waypoint
# ─────────────────────────────────────────────────────────────────────────────

FACE_THRESHOLD = math.radians(15)   # within 15° = "facing the target"

class RotateAndDriveController:
    def __init__(self):
        self.path: list = []
        self._idx = 0

    def set_path(self, path: list):
        self.path = path
        self._idx = 0

    def is_done(self) -> bool:
        return self._idx >= len(self.path)

    def compute(self, pose: np.ndarray):
        """Return (v, omega) for the given SE2 pose [x, y, theta]."""
        if not self.path or self.is_done():
            return 0.0, 0.0

        pos = pose[:2]

        # Advance past reached waypoints
        while self._idx < len(self.path) - 1:
            if np.hypot(*(self.path[self._idx] - pos)) < GOAL_TOLERANCE_M:
                self._idx += 1
            else:
                break

        if self.is_done():
            return 0.0, 0.0

        target = self.path[self._idx]
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return 0.0, 0.0

        # Angle error: how far off are we from facing the target?
        desired_heading = math.atan2(dy, dx)
        angle_err = (desired_heading - pose[2] + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_err) > FACE_THRESHOLD:
            # ── ROTATE: spin in place to face the target ──────────────
            # Proportional spin speed: faster when far off, slower when close
            spin_speed = MAX_ANGULAR_SPEED * min(1.0, abs(angle_err) / (math.pi / 2))
            spin_speed = max(spin_speed, 0.4)   # minimum spin speed
            omega = spin_speed if angle_err > 0 else -spin_speed
            return 0.0, omega
        else:
            # ── DRIVE: go straight with minor heading correction ──────
            # Small proportional steering to stay on course
            correction = angle_err * 0.5   # gentle correction
            v = MAX_LINEAR_SPEED * min(1.0, dist / 0.3)   # slow down near target
            omega = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, correction))
            return v, omega
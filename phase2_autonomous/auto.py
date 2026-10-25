"""
Waveshare 4WD Rover — Autonomous Frontier Exploration
======================================================
Requires slam.py (SLAMEngine, OccupancyGrid, MAP_RES_M, MAP_SIZE_M,
                   OBSTACLE_CRIT_M, OBSTACLE_WARN_M)

⚠️  SIMULATION MODE IS NOT SUPPORTED.
    Real LiDAR (RPLiDAR / YDLIDAR G4) is required.
    If the sensor cannot be opened the process exits immediately.

Architecture
------------
  SLAMEngine (slam.py)
      ↓ read-only from navigator thread, written only by visualiser (main thread)
  FrontierDetector     — finds unexplored grid boundaries
  AStarPlanner         — collision-free 8-connected A* on inflated occupancy grid
  PurePursuitController — path → (v, ω)
  WaveshareHTTPDriver   — (v, ω) → HTTP GET to rover
  AutonomousNavigator   — FSM tying everything together
  NavVisualiser         — extends SLAMVisualiser with nav overlay

Waveshare HTTP API  (same protocol as controller_based_controls.py)
------------------
  GET http://<ROVER_IP>/js?json={"T":1,"L":<left>,"R":<right>}
  L, R are floats (e.g. -0.30 .. +0.30), URL-encoded.
  Transport: raw TCP socket via PreconnectPool (fire-and-forget).

Dependencies
------------
  pip install numpy scipy matplotlib pyserial

Run
---
  python auto.py
"""

import json
import math
import socket
import sys
import time
import heapq
import threading
import logging
import urllib.parse
from collections import deque
from typing import Optional

import numpy as np

# ── Import SLAM module ────────────────────────────────────────────────────────
import slam as _slam_module
from slam import (
    SLAMEngine,
    OccupancyGrid,
    SLAMVisualiser,
    MAP_RES_M,
    MAP_SIZE_M,
    OBSTACLE_CRIT_M,
    OBSTACLE_WARN_M,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nav")

# ─────────────────────────────────────────────────────────────────────────────
#  !! CHANGE THESE TO MATCH YOUR SETUP !!
# ─────────────────────────────────────────────────────────────────────────────

# LiDAR serial port  (overrides slam.py's PORT constant)
LIDAR_PORT = "COM5"          # Windows: "COM5"  |  Linux: "/dev/ttyUSB0"

# Waveshare rover network settings
ROVER_IP   = "172.25.239.53"  # rover IP — must match your network
ROVER_PORT = 80
ROVER_ENDPOINT = "/js"        # Waveshare JSON command endpoint

# ─────────────────────────────────────────────────────────────────────────────
#  Motion / navigation configuration
# ─────────────────────────────────────────────────────────────────────────────

MAX_LINEAR_SPEED   = 0.45    # m/s — matches controller max speed
MAX_ANGULAR_SPEED  = 1.5     # rad/s — faster turning
ANGULAR_SCALE      = 0.35    # turning sensitivity (matches controller)
LOOKAHEAD_M        = 0.50    # pure-pursuit lookahead distance (metres)
GOAL_TOLERANCE_M   = 0.30    # distance to consider waypoint reached
FRONTIER_MIN_SIZE  = 8       # minimum frontier cluster size (cells)
REPLAN_INTERVAL_S  = 3.0     # re-run A* every N seconds while navigating
STUCK_TIMEOUT_S    = 6.0     # declare stuck after N seconds without movement
STUCK_ESCAPE_S     = 1.5     # reverse / spin duration when stuck
INFLATION_RADIUS_M = 0.20    # obstacle inflation radius for path planning (m)
WARMUP_FRAMES      = 40      # wait for this many SLAM frames before moving

# ── Apply LiDAR port override ─────────────────────────────────────────────────
_slam_module.PORT = LIDAR_PORT

# ─────────────────────────────────────────────────────────────────────────────
#  PreconnectPool  (copied from controller_based_controls.py)
# ─────────────────────────────────────────────────────────────────────────────

class PreconnectPool:
    """Fire-and-forget TCP pool that pre-connects the next socket in background."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._sock = None
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


def _build_http_request(ip: str, json_str: str) -> bytes:
    """Build a raw HTTP/1.0 GET request for the rover's /js endpoint."""
    encoded = urllib.parse.quote(json_str, safe="")
    req = (
        f"GET {ROVER_ENDPOINT}?json={encoded} HTTP/1.0\r\n"
        f"Host: {ip}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    )
    return req.encode("ascii")


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


# ─────────────────────────────────────────────────────────────────────────────
#  Waveshare HTTP driver  (same protocol as controller_based_controls.py)
# ─────────────────────────────────────────────────────────────────────────────

class WaveshareHTTPDriver:
    """
    Translates (linear m/s, angular rad/s) → Waveshare motor L/R values.

    Uses the same protocol as controller_based_controls.py:
      GET /js?json={"T":1,"L":<left>,"R":<right>}
    over a raw TCP PreconnectPool for minimal latency.

    Differential drive mixing:
        left  = throttle - turn
        right = throttle + turn
    clamped to ±MAX_LINEAR_SPEED.
    """

    def __init__(
        self,
        ip: str = ROVER_IP,
        port: int = ROVER_PORT,
        wheel_base_m: float = 0.17,
    ):
        self._ip        = ip
        self._pool      = PreconnectPool(ip, port)
        self.wheel_base = wheel_base_m
        self._last_cmd  = (None, None)
        self._consecutive_failures = 0
        self._max_failures         = 10
        log.info("Rover driver → http://%s:%d%s  (wheelbase=%.3f m)",
                 ip, port, ROVER_ENDPOINT, wheel_base_m)

    def set_velocity(self, linear: float, angular: float):
        """Convert (v, ω) to differential L/R and send to rover."""
        b = self.wheel_base
        lv = linear - angular * b / 2.0
        rv = linear + angular * b / 2.0
        left  = round(_clamp(lv, MAX_LINEAR_SPEED), 3)
        right = round(_clamp(rv, MAX_LINEAR_SPEED), 3)
        self._send(left, right)

    def stop(self):
        self._send(0.0, 0.0)

    def spin_in_place(self, direction: int = 1, speed: float = 0.15):
        """direction +1 = CCW/left, -1 = CW/right."""
        self._send(round(-direction * speed, 3), round(direction * speed, 3))

    def reverse(self, speed: float = 0.15):
        self._send(round(-speed, 3), round(-speed, 3))

    def _send(self, left: float, right: float):
        payload = json.dumps({"T": 1, "L": left, "R": right})
        data = _build_http_request(self._ip, payload)
        sent = self._pool.send(data)
        if sent:
            self._last_cmd = (left, right)
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures == 1 or \
               self._consecutive_failures % self._max_failures == 0:
                log.warning(
                    "Rover unreachable (failure #%d). Check IP / network.",
                    self._consecutive_failures,
                )

    def close(self):
        self._pool.close()

# ─────────────────────────────────────────────────────────────────────────────
#  Frontier detector
# ─────────────────────────────────────────────────────────────────────────────

class Frontier:
    __slots__ = ("centroid_world", "centroid_cell", "size")

    def __init__(self, centroid_world: np.ndarray, centroid_cell: tuple, size: int):
        self.centroid_world = centroid_world   # np.array [x, y] metres
        self.centroid_cell  = centroid_cell    # (cx, cy) grid coords
        self.size           = size

    def distance_to(self, pose: np.ndarray) -> float:
        return float(np.hypot(self.centroid_world[0] - pose[0],
                              self.centroid_world[1] - pose[1]))


class FrontierDetector:
    """
    A frontier is a FREE cell that has at least one UNKNOWN 4-connected neighbour.
    Adjacent frontier cells are clustered with BFS.
    """

    def detect(self, grid: OccupancyGrid) -> list:
        from scipy.ndimage import convolve

        log_g = grid.log
        n     = grid.n

        free_mask    = log_g < -0.3
        unknown_mask = np.abs(log_g) < 0.1

        kernel      = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        nbr_unknown = convolve(
            unknown_mask.astype(np.uint8), kernel, mode="constant", cval=0
        ) > 0
        frontier_mask = free_mask & nbr_unknown

        visited   = np.zeros((n, n), dtype=bool)
        frontiers = []
        fy_arr, fx_arr = np.where(frontier_mask)

        for sy, sx in zip(fy_arr, fx_arr):
            if visited[sy, sx]:
                continue
            cluster: list = []
            queue = deque([(sy, sx)])
            visited[sy, sx] = True

            while queue:
                cy, cx = queue.popleft()
                cluster.append((cx, cy))
                for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                    if 0 <= ny < n and 0 <= nx < n:
                        if frontier_mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))

            if len(cluster) < FRONTIER_MIN_SIZE:
                continue

            mcx = int(np.mean([c[0] for c in cluster]))
            mcy = int(np.mean([c[1] for c in cluster]))
            wx  = (mcx - grid.orig) * grid.res
            wy  = (mcy - grid.orig) * grid.res
            frontiers.append(Frontier(np.array([wx, wy]), (mcx, mcy), len(cluster)))

        return frontiers

# ─────────────────────────────────────────────────────────────────────────────
#  A* planner  (8-connected, with obstacle inflation)
# ─────────────────────────────────────────────────────────────────────────────

class AStarPlanner:
    def __init__(self, inflation_cells: int):
        self.inflation = max(0, inflation_cells)

    def plan(
        self,
        grid: OccupancyGrid,
        start_world: np.ndarray,
        goal_world: np.ndarray,
    ) -> Optional[list]:
        from scipy.ndimage import binary_dilation

        n, orig, res = grid.n, grid.orig, grid.res

        occ = (grid.log > 0.3).astype(np.uint8)
        if self.inflation > 0:
            r = self.inflation
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
        if occ[sy, sx]:
            log.warning("A*: start cell is occupied — using nearest free cell.")
            # nudge start to nearest free cell in a small search radius
            found = False
            for r in range(1, 5):
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        nx_, ny_ = sx+dx, sy+dy
                        if 0 <= nx_ < n and 0 <= ny_ < n and not occ[ny_, nx_]:
                            sx, sy = nx_, ny_
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                return None

        SQRT2 = math.sqrt(2)
        NEIGHBOURS = ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))

        open_  = [(0.0, (sx, sy))]
        came: dict = {}
        g_cost: dict = {(sx, sy): 0.0}

        while open_:
            _, cur = heapq.heappop(open_)
            cx, cy = cur
            if cx == gx and cy == gy:
                # Reconstruct path
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
                step   = SQRT2 if (dx and dy) else 1.0
                new_g  = g_cost[cur] + step
                nb     = (nx_, ny_)
                if new_g < g_cost.get(nb, 1e18):
                    came[nb]    = cur
                    g_cost[nb]  = new_g
                    f           = new_g + math.hypot(nx_-gx, ny_-gy)
                    heapq.heappush(open_, (f, nb))

        log.warning("A*: no path found from (%.2f,%.2f) to (%.2f,%.2f).",
                    *start_world[:2], *goal_world[:2])
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  Pure-pursuit path tracker
# ─────────────────────────────────────────────────────────────────────────────

class PurePursuitController:
    def __init__(self, lookahead: float = LOOKAHEAD_M):
        self.lookahead = lookahead
        self.path: list = []
        self._idx = 0

    def set_path(self, path: list):
        self.path = path
        self._idx = 0

    def is_done(self) -> bool:
        return self._idx >= len(self.path)

    def compute(self, pose: np.ndarray):
        """Return (v, omega) for the given SE2 pose."""
        if not self.path or self.is_done():
            return 0.0, 0.0

        pos = pose[:2]

        # Advance waypoint index
        while self._idx < len(self.path) - 1:
            if np.hypot(*(self.path[self._idx] - pos)) < GOAL_TOLERANCE_M:
                self._idx += 1
            else:
                break

        if self.is_done():
            return 0.0, 0.0

        # Pick lookahead target
        target = self.path[self._idx]
        for i in range(self._idx, len(self.path)):
            if np.hypot(*(self.path[i] - pos)) >= self.lookahead:
                target = self.path[i]
                break

        dx   = target[0] - pos[0]
        dy   = target[1] - pos[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return 0.0, 0.0

        # Steering angle relative to heading
        alpha = (math.atan2(dy, dx) - pose[2] + math.pi) % (2 * math.pi) - math.pi
        kappa = 2.0 * math.sin(alpha) / max(dist, 0.01)
        turn  = max(0.2, 1.0 - abs(alpha) / math.pi)   # slow down on sharp turns
        v     = MAX_LINEAR_SPEED * turn
        omega = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, v * kappa))
        return v, omega

# ─────────────────────────────────────────────────────────────────────────────
#  Navigator finite-state machine
# ─────────────────────────────────────────────────────────────────────────────

class NavState:
    WARMUP        = "WARMUP"
    DETECTING     = "DETECTING"
    PLANNING      = "PLANNING"
    NAVIGATING    = "NAVIGATING"
    OBSTACLE_STOP = "OBSTACLE_STOP"
    ESCAPING      = "ESCAPING"
    COMPLETE      = "COMPLETE"
    SENSOR_LOST   = "SENSOR_LOST"      # new — handles sensor disconnect


class AutonomousNavigator:
    """
    Runs in a background thread.
    eng.step() (SLAM) is called ONLY from the visualiser's main-thread callback.
    The navigator only READS engine state via slam_lock.
    """

    def __init__(self, engine: SLAMEngine, driver: WaveshareHTTPDriver):
        self.eng      = engine
        self.driver   = driver
        self.detector = FrontierDetector()
        self.planner  = AStarPlanner(max(1, int(INFLATION_RADIUS_M / MAP_RES_M)))
        self.tracker  = PurePursuitController()

        self.state              = NavState.WARMUP
        self.current_frontier: Optional[Frontier] = None
        self.path: list         = []

        self._last_replan    = 0.0
        self._last_progress  = 0.0
        self._last_pose      = np.zeros(3)
        self._escape_until   = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Protects reads of engine state between nav thread and main thread.
        # The main thread holds this lock while running eng.step().
        self.slam_lock = threading.Lock()

        self.stats = {"frontiers_visited": 0, "replans": 0, "stuck_events": 0}

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="navigator")
        self._thread.start()
        log.info("Navigator thread started.")

    def stop(self):
        self._running = False
        self.driver.stop()
        self.driver.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        log.info("Navigator stopped.")

    # ── Main loop ──────────────────────────────────────────────────────────────
    def _loop(self):
        while self._running:
            try:
                self._step()
            except Exception as exc:
                log.error("Navigator error: %s", exc, exc_info=True)
            time.sleep(0.05)

    # ── FSM step ───────────────────────────────────────────────────────────────
    def _step(self):
        # Snapshot SLAM state — brief lock to stay thread-safe
        with self.slam_lock:
            if not self.eng.is_connected:
                if self.state != NavState.SENSOR_LOST:
                    log.error("LiDAR disconnected — entering SENSOR_LOST state.")
                    self.driver.stop()
                    self.state = NavState.SENSOR_LOST
                return

            pose           = self.eng.pose.copy()
            frame_count    = self.eng.frame_count
            alert_level    = self.eng.proximity.alert_level
            closest_sector = self.eng.proximity.closest_sector
            grid_snapshot  = self.eng.grid   # A* reads a snapshot — fast enough

        now = time.perf_counter()

        # ── SENSOR_LOST ────────────────────────────────────────────────────────
        if self.state == NavState.SENSOR_LOST:
            return   # stay stopped; user must restart

        # ── WARMUP ────────────────────────────────────────────────────────────
        if self.state == NavState.WARMUP:
            if frame_count >= WARMUP_FRAMES:
                log.info("Warmup complete (%d frames) → DETECTING", frame_count)
                self.state = NavState.DETECTING
            else:
                log.debug("Warming up: %d/%d frames", frame_count, WARMUP_FRAMES)
            return

        # ── COMPLETE ──────────────────────────────────────────────────────────
        if self.state == NavState.COMPLETE:
            self.driver.stop()
            return

        # ── Critical obstacle (overrides NAVIGATING) ──────────────────────────
        if alert_level == 2 and self.state == NavState.NAVIGATING:
            log.warning("CRITICAL obstacle @ %s — emergency stop.", closest_sector)
            self.driver.stop()
            self.state         = NavState.OBSTACLE_STOP
            self._escape_until = now + STUCK_ESCAPE_S
            return

        # ── OBSTACLE_STOP ─────────────────────────────────────────────────────
        if self.state == NavState.OBSTACLE_STOP:
            if alert_level < 2:
                self.state         = NavState.ESCAPING
                self._escape_until = now + STUCK_ESCAPE_S
                log.info("Obstacle cleared → ESCAPING")
            else:
                self.driver.stop()
            return

        # ── ESCAPING ──────────────────────────────────────────────────────────
        if self.state == NavState.ESCAPING:
            if now < self._escape_until:
                if closest_sector == "FRONT":
                    self.driver.reverse(0.15)
                elif closest_sector == "REAR":
                    self.driver.set_velocity(0.15, 0.0)
                elif closest_sector == "RIGHT":
                    self.driver.spin_in_place(1, 0.15)    # spin left
                else:
                    self.driver.spin_in_place(-1, 0.15)   # spin right
            else:
                self.stats["stuck_events"] += 1
                self.driver.stop()
                self.state        = NavState.DETECTING
                self._last_replan = 0.0
                log.info("Escape done → DETECTING")
            return

        # ── DETECTING ─────────────────────────────────────────────────────────
        if self.state == NavState.DETECTING:
            frontiers = self.detector.detect(grid_snapshot)
            if not frontiers:
                log.info("No frontiers found — exploration complete.")
                self.state = NavState.COMPLETE
                self.driver.stop()
                return
            # Score: prefer large, close frontiers
            frontiers.sort(
                key=lambda f: f.distance_to(pose) - min(f.size, 60) * 0.25
            )
            self.current_frontier = frontiers[0]
            log.info(
                "Frontier → (%.2f, %.2f)  size=%d  dist=%.2f m",
                *self.current_frontier.centroid_world,
                self.current_frontier.size,
                self.current_frontier.distance_to(pose),
            )
            self.state = NavState.PLANNING
            return

        # ── PLANNING ──────────────────────────────────────────────────────────
        if self.state == NavState.PLANNING:
            path = self.planner.plan(
                grid_snapshot,
                pose[:2],
                self.current_frontier.centroid_world,
            )
            if path is None:
                log.warning("A* failed — re-detecting frontiers.")
                self.state = NavState.DETECTING
                return
            self.tracker.set_path(path)
            self.path           = path
            self._last_progress = now
            self._last_pose     = pose.copy()
            self._last_replan   = now
            self.stats["replans"] += 1
            log.info("Path planned: %d waypoints.", len(path))
            self.state = NavState.NAVIGATING
            return

        # ── NAVIGATING ────────────────────────────────────────────────────────
        if self.state == NavState.NAVIGATING:
            # Periodic replan
            if now - self._last_replan > REPLAN_INTERVAL_S:
                self._last_replan = now
                path = self.planner.plan(
                    grid_snapshot,
                    pose[:2],
                    self.current_frontier.centroid_world,
                )
                if path:
                    self.tracker.set_path(path)
                    self.path = path
                    log.debug("Replanned: %d waypoints.", len(path))

            # Check if frontier was reached
            if self.current_frontier.distance_to(pose) < GOAL_TOLERANCE_M * 2:
                log.info("Frontier reached.")
                self.stats["frontiers_visited"] += 1
                self.driver.stop()
                self.state = NavState.DETECTING
                return

            if self.tracker.is_done():
                log.debug("Path exhausted — re-detecting.")
                self.state = NavState.DETECTING
                return

            # Stuck detection
            moved = np.hypot(*(pose[:2] - self._last_pose[:2]))
            if moved > 0.05:
                self._last_progress = now
                self._last_pose     = pose.copy()
            elif now - self._last_progress > STUCK_TIMEOUT_S:
                log.warning("Rover appears stuck — initiating escape.")
                self.driver.stop()
                self.state         = NavState.OBSTACLE_STOP
                self._escape_until = now + STUCK_ESCAPE_S
                return

            # Drive
            v, omega = self.tracker.compute(pose)
            if alert_level == 1:
                v *= 0.4            # slow near warnings
            self.driver.set_velocity(v, omega)

# ─────────────────────────────────────────────────────────────────────────────
#  Visualiser with navigation overlay
# ─────────────────────────────────────────────────────────────────────────────

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


class NavVisualiser(SLAMVisualiser if MATPLOTLIB_OK else object):
    """Extends SLAMVisualiser with planned-path and frontier overlays."""

    def __init__(self, engine: SLAMEngine, navigator: AutonomousNavigator):
        super().__init__(engine)
        self.nav = navigator

        self.path_overlay, = self.ax_map.plot(
            [], [], "--", color="#00e5ff", lw=1.0, alpha=0.6,
            label="plan", zorder=6,
        )
        self.frontier_dot, = self.ax_map.plot(
            [], [], "*", color="#ff4081", ms=10, zorder=8, label="frontier",
        )
        self.ax_map.legend(
            fontsize=7, loc="upper right",
            facecolor=self.BG, edgecolor=self.GRID_COL, labelcolor=self.FG,
        )
        self.nav_hud = self.ax_map.text(
            0.02, 0.98, "", transform=self.ax_map.transAxes,
            fontsize=8, color=self.ACC, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=self.BG, edgecolor=self.GRID_COL, alpha=0.85),
        )

    def _update(self, frame):
        # SLAM step + nav overlay — all on the main thread
        with self.nav.slam_lock:
            artists = super()._update(frame)

        # Planned path overlay
        if self.nav.path:
            self.path_overlay.set_data(
                [p[0] for p in self.nav.path],
                [p[1] for p in self.nav.path],
            )
        else:
            self.path_overlay.set_data([], [])

        # Current frontier marker
        fr = self.nav.current_frontier
        if fr is not None:
            self.frontier_dot.set_data(
                [fr.centroid_world[0]], [fr.centroid_world[1]]
            )
        else:
            self.frontier_dot.set_data([], [])

        # HUD
        s = self.nav.stats
        self.nav_hud.set_text(
            f"STATE : {self.nav.state}\n"
            f"VISIT : {s['frontiers_visited']}\n"
            f"STUCK : {s['stuck_events']}\n"
            f"REPLNS: {s['replans']}"
        )

        return artists + (self.path_overlay, self.frontier_dot, self.nav_hud)

    def run(self):
        self.nav.start()
        self._anim = animation.FuncAnimation(
            self.fig, self._update, interval=50,
            blit=False, cache_frame_data=False,
        )
        try:
            plt.show()
        finally:
            self._anim.event_source.stop()
            self.nav.stop()
            self.eng.disconnect()
            plt.close(self.fig)

# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Connect LiDAR ─────────────────────────────────────────────────────────
    engine = SLAMEngine()
    if not engine.connect():
        print(
            "\n[FATAL] Could not connect to LiDAR.\n"
            "  • Verify LIDAR_PORT at the top of auto.py\n"
            "  • Ensure PyLidar3 and pyserial are installed\n"
            "  • Check physical USB connection\n"
        )
        sys.exit(1)

    # ── Motor driver ──────────────────────────────────────────────────────────
    driver = WaveshareHTTPDriver(
        ip           = ROVER_IP,
        port         = ROVER_PORT,
        wheel_base_m = 0.17,   # measure centre-to-centre wheel distance on your rover
    )

    # ── Navigator ─────────────────────────────────────────────────────────────
    navigator = AutonomousNavigator(engine, driver)

    # ── Run ───────────────────────────────────────────────────────────────────
    if MATPLOTLIB_OK:
        NavVisualiser(engine, navigator).run()
    else:
        # Headless / server mode
        log.info("Matplotlib not available — running headless.")
        navigator.start()
        try:
            while True:
                time.sleep(1.0)
                log.info(
                    "State=%-14s  Pose=(%.2f, %.2f, %.1f°)  FPS=%.1f  "
                    "Frames=%d  Connected=%s",
                    navigator.state,
                    engine.pose[0], engine.pose[1],
                    math.degrees(engine.pose[2]),
                    engine.fps,
                    engine.frame_count,
                    engine.is_connected,
                )
                if not engine.is_connected:
                    log.error("Sensor lost — shutting down.")
                    break
        except KeyboardInterrupt:
            log.info("Keyboard interrupt — shutting down.")
        finally:
            navigator.stop()
            engine.disconnect()
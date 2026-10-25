"""
RPLiDAR G4 / YDLIDAR G4 — Real-Hardware SLAM
=============================================
Algorithm  : ICP scan-matching + occupancy-grid mapping
Visualiser : Matplotlib real-time dashboard (dark theme)

⚠️  SIMULATION MODE HAS BEEN REMOVED.
    This code requires real LiDAR hardware on the configured serial port.
    If the sensor cannot be opened, the process exits with a clear error.

Dependencies
------------
    pip install PyLidar3 numpy scipy matplotlib pyserial

Run
---
    python slam.py
"""

import math
import sys
import time
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np
from scipy.spatial import KDTree

# ── LiDAR driver ──────────────────────────────────────────────────────────────
try:
    import PyLidar3
    LIDAR_AVAILABLE = True
except ImportError:
    LIDAR_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration  — imported from config.py (edit config.py to change)
# ─────────────────────────────────────────────────────────────────────────────

from config import (
    LIDAR_PORT as _CFG_PORT,
    MIN_RANGE_MM, MAX_RANGE_MM, MAP_RES_M, MAP_SIZE_M,
    ICP_MAX_ITER, ICP_TOLERANCE, ICP_MAX_DIST, MIN_POINTS,
    FPS_WINDOW, TRAIL_LEN, IMU_ALPHA,
    OBSTACLE_WARN_M, OBSTACLE_CRIT_M, SECTOR_HALF_DEG,
)

# PORT is overridable at runtime (e.g. by hybrid.py setting slam.PORT)
PORT = _CFG_PORT

# ─────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def scan_to_points(raw: dict) -> np.ndarray:
    """Convert raw {angle_deg: dist_mm} dict → Nx2 float array in metres."""
    pts = []
    for deg, mm in raw.items():
        if MIN_RANGE_MM < mm <= MAX_RANGE_MM:
            rad = math.radians(deg)
            pts.append([mm * math.cos(rad) / 1000.0,
                        mm * math.sin(rad) / 1000.0])
    return np.array(pts, dtype=np.float64) if pts else np.empty((0, 2))


def transform_points(pts: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply SE2 pose [x, y, theta] to Nx2 point cloud."""
    x, y, theta = pose
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    t = np.array([x, y])
    return pts @ R.T + t


def se2_compose(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Compose two SE2 poses:  result = p1 ∘ p2."""
    x1, y1, t1 = p1
    x2, y2, t2 = p2
    c, s = math.cos(t1), math.sin(t1)
    return np.array([
        x1 + c * x2 - s * y2,
        y1 + s * x2 + c * y2,
        t1 + t2
    ])


def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

# ─────────────────────────────────────────────────────────────────────────────
#  ICP scan matcher
# ─────────────────────────────────────────────────────────────────────────────

def icp(source: np.ndarray, target: np.ndarray,
        max_iter: int = ICP_MAX_ITER,
        tolerance: float = ICP_TOLERANCE,
        max_dist: float = ICP_MAX_DIST) -> tuple:
    """
    Vanilla point-to-point ICP.
    Returns (delta_pose [dx, dy, dtheta], rms_error).
    """
    if len(source) < 5 or len(target) < 5:
        return np.zeros(3), float("inf")

    src  = source.copy()
    tree = KDTree(target)
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

        # Procrustes (SVD)
        mu_s = S.mean(axis=0)
        mu_t = T.mean(axis=0)
        H = (S - mu_s).T @ (T - mu_t)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:        # reflection guard
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = mu_t - R @ mu_s

        src   = src @ R.T + t
        R_acc = R @ R_acc
        t_acc = R @ t_acc + t

        err = dists[mask].mean()
        if abs(prev_err - err) < tolerance:
            break
        prev_err = err

    dtheta = math.atan2(R_acc[1, 0], R_acc[0, 0])
    delta  = np.array([t_acc[0], t_acc[1], dtheta])
    return delta, prev_err

# ─────────────────────────────────────────────────────────────────────────────
#  Occupancy grid
# ─────────────────────────────────────────────────────────────────────────────

class OccupancyGrid:
    LOG_ODD_FREE  = -0.4
    LOG_ODD_OCC   =  0.9
    LOG_ODD_CLAMP =  5.0

    def __init__(self, size_m: float, res_m: float):
        self.res  = res_m
        self.n    = int(size_m / res_m)
        self.orig = self.n // 2          # grid origin in cell coords
        self.log  = np.zeros((self.n, self.n), dtype=np.float32)

    def _world_to_cell(self, x: float, y: float):
        cx = int(round(x / self.res)) + self.orig
        cy = int(round(y / self.res)) + self.orig
        return cx, cy

    def update(self, pose: np.ndarray, pts_world: np.ndarray):
        """Bresenham ray-cast update for all points in the scan."""
        rx, ry = self._world_to_cell(pose[0], pose[1])
        for px, py in pts_world:
            ex, ey = self._world_to_cell(px, py)
            for fx, fy in self._bresenham(rx, ry, ex, ey):
                if 0 <= fx < self.n and 0 <= fy < self.n:
                    self.log[fy, fx] = max(
                        -self.LOG_ODD_CLAMP,
                        self.log[fy, fx] + self.LOG_ODD_FREE)
            if 0 <= ex < self.n and 0 <= ey < self.n:
                self.log[ey, ex] = min(
                    self.LOG_ODD_CLAMP,
                    self.log[ey, ex] + self.LOG_ODD_OCC)

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        """Yield cells along a line (excluding endpoint)."""
        cells = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while (x0, y0) != (x1, y1):
            cells.append((x0, y0))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0  += sx
            if e2 < dx:
                err += dx
                y0  += sy
        return cells

    def to_image(self) -> np.ndarray:
        """Return uint8 image: 128=unknown(grey), 255=free(white), 0=occupied(black)."""
        img = np.full((self.n, self.n), 128, dtype=np.uint8)
        img[self.log >  0.3] = 0
        img[self.log < -0.3] = 255
        return img

    def save(self, filepath: str):
        """Save occupancy grid to a .npz file."""
        np.savez_compressed(filepath,
                            log=self.log, res=self.res,
                            n=self.n, orig=self.orig)
        print(f"[OK] Map saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'OccupancyGrid':
        """Load occupancy grid from a .npz file."""
        data = np.load(filepath)
        grid = cls.__new__(cls)
        grid.log  = data['log']
        grid.res  = float(data['res'])
        grid.n    = int(data['n'])
        grid.orig = int(data['orig'])
        print(f"[OK] Map loaded from {filepath} ({grid.n}x{grid.n}, res={grid.res}m)")
        return grid

# ─────────────────────────────────────────────────────────────────────────────
#  Obstacle proximity detector
# ─────────────────────────────────────────────────────────────────────────────

class ProximityDetector:
    """
    Computes per-sector minimum distances from local scan points.

    Sectors are defined in the robot's local frame (angle 0 = forward):
        FRONT : -HALF .. +HALF degrees
        LEFT  : 90 ± HALF
        RIGHT : 270 ± HALF
        REAR  : 180 ± HALF
    """

    SECTORS = {
        "FRONT":   0,
        "LEFT" :  90,
        "REAR" : 180,
        "RIGHT": 270,
    }

    def __init__(self):
        self.sector_dists: dict = {k: float("inf") for k in self.SECTORS}
        self.min_dist: float    = float("inf")
        self.alert_level: int   = 0      # 0=clear  1=warn  2=critical
        self.closest_sector: str = ""

    def update(self, pts_local: np.ndarray):
        """pts_local: Nx2 array in robot frame."""
        if pts_local is None or len(pts_local) == 0:
            self.sector_dists = {k: float("inf") for k in self.SECTORS}
            self.min_dist     = float("inf")
            self.alert_level  = 0
            return

        angles_deg = (np.degrees(np.arctan2(pts_local[:, 1], pts_local[:, 0])) % 360)
        dists      = np.hypot(pts_local[:, 0], pts_local[:, 1])
        half       = SECTOR_HALF_DEG

        for name, centre in self.SECTORS.items():
            diff = ((angles_deg - centre + 180) % 360) - 180
            mask = np.abs(diff) <= half
            self.sector_dists[name] = float(dists[mask].min()) if mask.any() else float("inf")

        self.min_dist       = min(self.sector_dists.values())
        self.closest_sector = min(self.sector_dists, key=self.sector_dists.get)

        if self.min_dist <= OBSTACLE_CRIT_M:
            self.alert_level = 2
        elif self.min_dist <= OBSTACLE_WARN_M:
            self.alert_level = 1
        else:
            self.alert_level = 0

# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR connection helper
# ─────────────────────────────────────────────────────────────────────────────

def _open_lidar(port: str):
    """
    Attempt to open the real LiDAR sensor.
    Returns (lidar_obj, scan_generator) on success.
    Raises RuntimeError with a descriptive message on failure — NO simulation fallback.
    """
    if not LIDAR_AVAILABLE:
        raise RuntimeError(
            "PyLidar3 is not installed.\n"
            "  Fix: pip install PyLidar3\n"
            "  Then re-run the script."
        )

    import serial  # noqa: F401  (verify pyserial is present)

    # Release stale lock if a previous run crashed
    try:
        import serial as _s
        p = _s.Serial(port)
        p.close()
    except Exception:
        pass

    lidar = PyLidar3.YdLidarG4(port)
    if not lidar.Connect():
        raise RuntimeError(
            f"Could not connect to LiDAR on port '{port}'.\n"
            "  • Check Device Manager (Windows) / ls /dev/ttyUSB* (Linux)\n"
            "  • Verify the correct port in slam.py → PORT\n"
            "  • Make sure no other process (e.g. RViz) holds the port\n"
            "  • Try unplugging and replugging the USB cable"
        )

    gen = lidar.StartScanning()
    print(f"[OK] LiDAR connected on {port}")
    return lidar, gen

# ─────────────────────────────────────────────────────────────────────────────
#  Main SLAM class
# ─────────────────────────────────────────────────────────────────────────────

class SLAMEngine:
    def __init__(self):
        self.pose        = np.zeros(3)           # [x, y, theta]  metres / rad
        self.path        = deque(maxlen=TRAIL_LEN)
        self.prev_pts    = None
        self.grid        = OccupancyGrid(MAP_SIZE_M, MAP_RES_M)
        self.frame_times = deque(maxlen=FPS_WINDOW)
        self.frame_count = 0
        self.proximity   = ProximityDetector()
        self._lidar      = None
        self._gen        = None
        self._connected  = False

        # ── IMU fusion ────────────────────────────────────────────────────────
        self.imu_yaw          = None     # latest IMU yaw in radians (relative)
        self._imu_yaw_offset  = None     # captured on first reading
        self._imu_initialized = False
        self.imu_alpha        = IMU_ALPHA  # complementary filter weight

    def set_imu_yaw(self, yaw_deg: float):
        """
        Feed an absolute yaw reading from the rover's IMU (degrees).
        On first call, captures the offset so SLAM starts at heading 0.
        Thread-safe: called from IMUReader background thread.
        """
        yaw_rad = math.radians(yaw_deg)
        if not self._imu_initialized:
            self._imu_yaw_offset = yaw_rad
            self._imu_initialized = True
        self.imu_yaw = normalize_angle(yaw_rad - self._imu_yaw_offset)

    # ── Sensor connection ──────────────────────────────────────────────────────
    def connect(self) -> bool:
        """
        Open the real LiDAR.
        Returns True on success, prints error and returns False on failure.
        Does NOT fall back to simulation.
        """
        try:
            self._lidar, self._gen = _open_lidar(PORT)
            self._connected = True
            return True
        except RuntimeError as exc:
            print(f"\n[ERROR] {exc}\n")
            return False

    def disconnect(self):
        if self._lidar and self._connected:
            try:
                self._lidar.StopScanning()
                self._lidar.Disconnect()
                print("[OK] LiDAR disconnected.")
            except Exception as exc:
                print(f"[WARN] Disconnect error: {exc}")
            self._connected = False

    # ── Raw scan ───────────────────────────────────────────────────────────────
    def _next_raw(self) -> dict | None:
        """
        Pull the next scan from the real sensor.
        Returns None if the generator is exhausted or sensor disconnected.
        """
        if self._gen is None:
            print("[ERROR] LiDAR generator not initialised — was connect() called?")
            return None
        try:
            return next(self._gen)
        except StopIteration:
            print("[WARN] LiDAR scan stream ended unexpectedly.")
            self._connected = False
            return None
        except Exception as exc:
            print(f"[ERROR] LiDAR read error: {exc}")
            self._connected = False
            return None

    # ── Single SLAM step ───────────────────────────────────────────────────────
    def step(self) -> bool:
        """
        Process one scan frame: ICP → pose update → IMU fusion → map update → proximity.
        Call this from the visualiser's animation callback (main thread only).
        Returns False if the sensor has disconnected.
        """
        if not self._connected:
            return False

        raw = self._next_raw()
        if raw is None:
            return False

        pts_local = scan_to_points(raw)
        if len(pts_local) < MIN_POINTS:
            return True   # valid read but too few points — skip ICP this frame

        # ICP scan matching
        if self.prev_pts is not None and len(self.prev_pts) >= MIN_POINTS:
            delta, err = icp(pts_local, self.prev_pts)
            if err < float("inf"):          # only apply if ICP converged
                self.pose = se2_compose(self.pose, delta)
                self.pose[2] = normalize_angle(self.pose[2])

        # IMU heading fusion (complementary filter)
        if self.imu_yaw is not None:
            icp_heading = self.pose[2]
            imu_heading = self.imu_yaw
            # Blend: handle angle wrapping correctly
            diff = normalize_angle(imu_heading - icp_heading)
            self.pose[2] = normalize_angle(
                icp_heading + (1.0 - self.imu_alpha) * diff
            )

        # Map update
        pts_world = transform_points(pts_local, self.pose)
        self.grid.update(self.pose, pts_world)

        # Proximity check
        self.proximity.update(pts_local)

        self.path.append(self.pose.copy())
        self.prev_pts    = pts_local
        self.frame_count += 1
        self.frame_times.append(time.perf_counter())
        return True

    @property
    def fps(self) -> float:
        ft = self.frame_times
        if len(ft) < 2:
            return 0.0
        return (len(ft) - 1) / (ft[-1] - ft[0])

    @property
    def is_connected(self) -> bool:
        return self._connected

# ─────────────────────────────────────────────────────────────────────────────
#  Visualiser
# ─────────────────────────────────────────────────────────────────────────────

class SLAMVisualiser:
    BG       = "#0d0f14"
    FG       = "#e8eaf0"
    ACC      = "#00e5ff"
    ACC2     = "#ff4081"
    GRID_COL = "#1e2130"

    CLR_CLEAR = "#00e676"
    CLR_WARN  = "#ffea00"
    CLR_CRIT  = "#ff1744"

    SECTOR_ARROW = {"FRONT": "↑", "LEFT": "←", "REAR": "↓", "RIGHT": "→"}

    def __init__(self, engine: SLAMEngine):
        self.eng            = engine
        self._blink_state   = True
        self._blink_counter = 0
        self.heading_arrow  = None
        self._build_figure()

    def _build_figure(self):
        plt.rcParams.update({
            "font.family"    : "monospace",
            "axes.facecolor" : self.BG,
            "figure.facecolor": self.BG,
            "text.color"     : self.FG,
            "axes.labelcolor": self.FG,
            "xtick.color"    : self.FG,
            "ytick.color"    : self.FG,
            "axes.edgecolor" : self.GRID_COL,
            "grid.color"     : self.GRID_COL,
        })

        self.fig = plt.figure(figsize=(16, 9), facecolor=self.BG)
        self.fig.canvas.manager.set_window_title("RPLiDAR G4 — ICP-SLAM")

        gs = gridspec.GridSpec(
            2, 3,
            figure=self.fig,
            width_ratios=[2.5, 1, 1],
            height_ratios=[1, 1],
            hspace=0.35, wspace=0.35,
            left=0.05, right=0.97,
            top=0.92,  bottom=0.08,
        )

        # ── Occupancy map ──────────────────────────────────────────────────────
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_map.set_title("OCCUPANCY MAP", color=self.ACC,
                              fontsize=10, fontweight="bold", pad=10)
        self.ax_map.set_aspect("equal")
        self.ax_map.tick_params(labelsize=7)
        ext = MAP_SIZE_M / 2
        self.im_map = self.ax_map.imshow(
            self.eng.grid.to_image(),
            cmap="gray", vmin=0, vmax=255,
            extent=[-ext, ext, -ext, ext],
            origin="lower", interpolation="nearest",
        )
        self.path_line, = self.ax_map.plot([], [], color=self.ACC2,
                                           lw=1.2, alpha=0.8, label="path")
        self.robot_dot, = self.ax_map.plot([], [], "o", color=self.ACC,
                                           ms=8, zorder=10, label="robot")
        self.ax_map.legend(fontsize=7, loc="upper right",
                          facecolor=self.BG, edgecolor=self.GRID_COL,
                          labelcolor=self.FG)

        # ── Live polar scan ────────────────────────────────────────────────────
        self.ax_scan = self.fig.add_subplot(gs[0, 1], projection="polar")
        self.ax_scan.set_title("LIVE SCAN", color=self.ACC,
                               fontsize=9, fontweight="bold", pad=8)
        self.ax_scan.set_theta_zero_location("N")
        self.ax_scan.set_theta_direction(-1)
        self.ax_scan.set_ylim(0, MAX_RANGE_MM / 1000)
        self.ax_scan.tick_params(labelsize=6, colors="#555e7a")
        self.ax_scan.set_facecolor(self.BG)
        self.scan_scatter = self.ax_scan.scatter(
            [], [], s=3, c=[], cmap="plasma",
            vmin=0, vmax=MAX_RANGE_MM / 1000, alpha=0.9,
        )

        # ── Obstacle proximity panel ───────────────────────────────────────────
        self.ax_prox = self.fig.add_subplot(gs[0, 2])
        self.ax_prox.set_axis_off()
        self.ax_prox.set_title("OBSTACLE PROXIMITY", color=self.ACC,
                               fontsize=9, fontweight="bold")

        self.prox_bg = mpatches.FancyBboxPatch(
            (0.05, 0.05), 0.90, 0.90,
            boxstyle="round,pad=0.02",
            transform=self.ax_prox.transAxes,
            linewidth=2, edgecolor=self.GRID_COL,
            facecolor=self.BG, zorder=0,
        )
        self.ax_prox.add_patch(self.prox_bg)

        self.prox_status = self.ax_prox.text(
            0.5, 0.82, "CLEAR", ha="center", va="center",
            fontsize=14, fontweight="bold", color=self.CLR_CLEAR,
            transform=self.ax_prox.transAxes, zorder=5,
        )
        self.prox_dist = self.ax_prox.text(
            0.5, 0.60, "-- m", ha="center", va="center",
            fontsize=18, fontweight="bold", color=self.FG,
            transform=self.ax_prox.transAxes, zorder=5,
            fontfamily="monospace",
        )

        sector_pos = {
            "FRONT": (0.5, 0.40),
            "LEFT" : (0.15, 0.22),
            "RIGHT": (0.85, 0.22),
            "REAR" : (0.5,  0.10),
        }
        self.sector_texts: dict = {}
        for name, (tx, ty) in sector_pos.items():
            arrow = self.SECTOR_ARROW[name]
            self.sector_texts[name] = self.ax_prox.text(
                tx, ty, f"{arrow} --", ha="center", va="center",
                fontsize=8, color="#7a8499",
                transform=self.ax_prox.transAxes, zorder=5,
                fontfamily="monospace",
            )

        # ── Pose readout ───────────────────────────────────────────────────────
        self.ax_pose = self.fig.add_subplot(gs[1, 1])
        self.ax_pose.set_axis_off()
        self.ax_pose.set_title("POSE", color=self.ACC,
                               fontsize=9, fontweight="bold")
        self.pose_text = self.ax_pose.text(
            0.5, 0.5, "", ha="center", va="center",
            fontsize=12, color=self.ACC,
            transform=self.ax_pose.transAxes,
            fontfamily="monospace",
        )

        # ── Stats ──────────────────────────────────────────────────────────────
        self.ax_stats = self.fig.add_subplot(gs[1, 2])
        self.ax_stats.set_axis_off()
        self.ax_stats.set_title("STATS", color=self.ACC,
                                fontsize=9, fontweight="bold")
        self.stats_text = self.ax_stats.text(
            0.5, 0.5, "", ha="center", va="center",
            fontsize=10, color=self.FG,
            transform=self.ax_stats.transAxes,
            fontfamily="monospace",
        )

        self.fig.suptitle(
            "RPLiDAR G4  ·  ICP-SLAM  ·  Real Hardware",
            fontsize=13, color=self.ACC, fontweight="bold", y=0.97,
        )

    # ── Proximity panel update ─────────────────────────────────────────────────
    def _update_proximity(self):
        prox  = self.eng.proximity
        level = prox.alert_level

        if level == 2:
            self._blink_counter += 1
            if self._blink_counter >= 3:
                self._blink_state   = not self._blink_state
                self._blink_counter = 0
            accent = self.CLR_CRIT if self._blink_state else self.BG
            bg_col = "#2a0008"
            label  = "!! CRITICAL !!"
        elif level == 1:
            accent  = self.CLR_WARN
            bg_col  = "#1f1c00"
            label   = "  WARNING  "
            self._blink_state = True
            self._blink_counter = 0
        else:
            accent  = self.CLR_CLEAR
            bg_col  = self.BG
            label   = "  CLEAR  "
            self._blink_state = True
            self._blink_counter = 0

        self.prox_bg.set_facecolor(bg_col)
        self.prox_bg.set_edgecolor(accent)
        self.prox_status.set_text(label)
        self.prox_status.set_color(accent)

        md = prox.min_dist
        dist_str = f"{md:.2f} m" if md < float("inf") else "-- m"
        self.prox_dist.set_text(dist_str)
        self.prox_dist.set_color(accent)

        for name, txt_obj in self.sector_texts.items():
            d   = prox.sector_dists[name]
            val = f"{d:.2f}" if d < float("inf") else "--"
            arrow = self.SECTOR_ARROW[name]
            txt_obj.set_text(f"{arrow} {val} m")
            if name == prox.closest_sector and level > 0:
                txt_obj.set_color(accent)
                txt_obj.set_fontsize(9)
            else:
                txt_obj.set_color("#7a8499")
                txt_obj.set_fontsize(8)

    # ── Animation callback ─────────────────────────────────────────────────────
    def _update(self, _frame):
        # SLAM step runs here — single main thread, no race conditions
        if not self.eng.step():
            # Sensor disconnected — stop animation gracefully
            print("[WARN] Sensor disconnected — stopping visualiser.")
            self._anim.event_source.stop()
            return (self.im_map, self.path_line, self.robot_dot,
                    self.scan_scatter, self.pose_text, self.stats_text,
                    self.prox_status, self.prox_dist)

        g   = self.eng
        pts = g.prev_pts
        pos = g.pose

        # Occupancy map
        self.im_map.set_data(g.grid.to_image())

        # Robot path
        if g.path:
            px = [p[0] for p in g.path]
            py = [p[1] for p in g.path]
            self.path_line.set_data(px, py)
            self.robot_dot.set_data([pos[0]], [pos[1]])

            # Heading arrow
            if self.heading_arrow:
                self.heading_arrow.remove()
            hl = 0.4
            self.heading_arrow = self.ax_map.annotate(
                "",
                xy=(pos[0] + hl * math.cos(pos[2]),
                    pos[1] + hl * math.sin(pos[2])),
                xytext=(pos[0], pos[1]),
                arrowprops=dict(arrowstyle="->", color=self.ACC, lw=1.5),
            )

        # Live polar scan
        if pts is not None and len(pts):
            angles = np.arctan2(pts[:, 1], pts[:, 0])
            dists  = np.hypot(pts[:, 0], pts[:, 1])
            self.scan_scatter.set_offsets(np.c_[angles, dists])
            self.scan_scatter.set_array(dists)

        # Proximity
        self._update_proximity()

        # Pose
        self.pose_text.set_text(
            f"X : {pos[0]:+.3f} m\n"
            f"Y : {pos[1]:+.3f} m\n"
            f"θ : {math.degrees(pos[2]):+.1f}°"
        )

        # Stats (no "SIMULATION" mode any more — always HARDWARE)
        npts = len(pts) if pts is not None else 0
        self.stats_text.set_text(
            f"FPS   : {g.fps:.1f}\n"
            f"Frame : {g.frame_count}\n"
            f"Pts   : {npts}\n"
            f"Mode  : HARDWARE"
        )

        return (self.im_map, self.path_line, self.robot_dot,
                self.scan_scatter, self.pose_text, self.stats_text,
                self.prox_status, self.prox_dist)

    def run(self):
        self._anim = animation.FuncAnimation(
            self.fig, self._update,
            interval=50,
            blit=False,
            cache_frame_data=False,
        )
        try:
            plt.show()
        finally:
            self._anim.event_source.stop()
            self.eng.disconnect()
            plt.close(self.fig)

# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = SLAMEngine()
    if not engine.connect():
        print("Exiting: real LiDAR is required — no simulation fallback.")
        sys.exit(1)
    SLAMVisualiser(engine).run()
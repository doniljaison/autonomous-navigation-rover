"""
YDLIDAR G4 — Full SLAM Implementation  (Raspberry Pi 5)
=========================================================
Algorithm  : ICP (Iterative Closest Point) scan-matching  +  occupancy-grid mapping
Visualiser : Matplotlib real-time dashboard (dark theme)

Dependencies
------------
    pip install PyLidar3 numpy scipy matplotlib pyserial

Raspberry Pi setup
------------------
    # Add your user to the dialout group so you can open serial ports without sudo:
    sudo usermod -aG dialout $USER
    # Then log out and back in (or reboot) for the change to take effect.

    # If running headless (no monitor), install a virtual display first:
    sudo apt install xvfb
    Xvfb :1 -screen 0 1280x720x24 &
    export DISPLAY=:1

    # Common serial device paths on Raspberry Pi:
    #   USB-to-serial adapter  →  /dev/ttyUSB0
    #   Hardware UART          →  /dev/ttyAMA0  or  /dev/serial0
    # Check with:  ls /dev/ttyUSB*  or  ls /dev/ttyAMA*

Run
---
    python lidar_slam_rpi5.py
"""

import math
import time
import os
from collections import deque

# ── Pick a display-safe backend BEFORE importing pyplot ──────────────────────
# TkAgg works out-of-the-box on Raspberry Pi OS (desktop).
# If you're running headless (SSH without X-forwarding), set DISPLAY first
# or switch the backend to "Agg" (no window — saves images instead).
import matplotlib
_display = os.environ.get("DISPLAY", "")
if not _display:
    # No display detected – fall back to non-interactive Agg
    matplotlib.use("Agg")
    print("[INFO] No DISPLAY found — using Agg backend (no live window).")
else:
    matplotlib.use("TkAgg")   # Most reliable on RPi OS desktop / X11

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np
from scipy.spatial import KDTree

try:
    import PyLidar3
    LIDAR_AVAILABLE = True
except ImportError:
    LIDAR_AVAILABLE = False
    print("[WARN] PyLidar3 not found — running in SIMULATION mode.")

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
# !! CHANGED: COM10  →  /dev/ttyUSB0  (typical USB-serial adapter on Linux)
# Adjust if your device shows up as ttyUSB1, ttyAMA0, etc.
PORT          = "/dev/ttyUSB0"
MIN_RANGE_MM  = 150        # ignore closer readings (noise)
MAX_RANGE_MM  = 8000       # effective sensor range
MAP_RES_M     = 0.05       # metres per grid cell  (5 cm)
MAP_SIZE_M    = 20.0       # map side length in metres
ICP_MAX_ITER  = 30         # ICP iterations per frame
ICP_TOLERANCE = 1e-5       # convergence threshold
ICP_MAX_DIST  = 0.5        # max correspondence distance (m)
MIN_POINTS    = 30         # skip frame if fewer points
FPS_WINDOW    = 30         # frames for rolling FPS average
TRAIL_LEN     = 500        # robot path history length

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
        max_dist: float = ICP_MAX_DIST) -> tuple[np.ndarray, float]:
    """
    Vanilla point-to-point ICP.
    Returns (delta_pose [dx, dy, dtheta], rms_error).
    """
    if len(source) < 5 or len(target) < 5:
        return np.zeros(3), float("inf")

    src = source.copy()
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

        src = src @ R.T + t
        R_acc = R @ R_acc
        t_acc = R @ t_acc + t

        err = dists[mask].mean()
        if abs(prev_err - err) < tolerance:
            break
        prev_err = err

    # Extract angle from accumulated rotation
    dtheta = math.atan2(R_acc[1, 0], R_acc[0, 0])
    delta  = np.array([t_acc[0], t_acc[1], dtheta])
    return delta, prev_err

# ─────────────────────────────────────────────────────────────────────────────
#  Occupancy grid
# ─────────────────────────────────────────────────────────────────────────────

class OccupancyGrid:
    LOG_ODD_FREE  = -0.4
    LOG_ODD_OCC   = 0.9
    LOG_ODD_CLAMP = 5.0

    def __init__(self, size_m: float, res_m: float):
        self.res  = res_m
        self.n    = int(size_m / res_m)
        self.orig = self.n // 2          # grid origin in cell coords
        self.log  = np.zeros((self.n, self.n), dtype=np.float32)

    def _world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        cx = int(round(x / self.res)) + self.orig
        cy = int(round(y / self.res)) + self.orig
        return cx, cy

    def update(self, pose: np.ndarray, pts_world: np.ndarray):
        """Bresenham ray-cast update for all points in the scan."""
        rx, ry = self._world_to_cell(pose[0], pose[1])
        for px, py in pts_world:
            ex, ey = self._world_to_cell(px, py)
            # Ray-trace free cells
            for fx, fy in self._bresenham(rx, ry, ex, ey):
                if 0 <= fx < self.n and 0 <= fy < self.n:
                    self.log[fy, fx] = max(-self.LOG_ODD_CLAMP,
                                           self.log[fy, fx] + self.LOG_ODD_FREE)
            # Mark endpoint occupied
            if 0 <= ex < self.n and 0 <= ey < self.n:
                self.log[ey, ex] = min(self.LOG_ODD_CLAMP,
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
        """Return uint8 image: 0=unknown(grey), 255=free(white), 1=occupied(black)."""
        img = np.full((self.n, self.n), 128, dtype=np.uint8)
        img[self.log >  0.3] = 0          # occupied → black
        img[self.log < -0.3] = 255         # free     → white
        return img

# ─────────────────────────────────────────────────────────────────────────────
#  Simulated LiDAR (for testing without hardware)
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedLidar:
    """Simple box-room simulator so the SLAM pipeline can be tested offline."""

    WALLS = [
        ((-5, -5), ( 5, -5)),
        (( 5, -5), ( 5,  5)),
        (( 5,  5), (-5,  5)),
        ((-5,  5), (-5, -5)),
        ((-1, -1), ( 1, -1)),   # inner box (obstacle)
        (( 1, -1), ( 1,  1)),
        (( 1,  1), (-1,  1)),
        ((-1,  1), (-1, -1)),
    ]

    def __init__(self):
        self._t     = 0.0
        self._noise = 0.02   # metres

    def next_scan(self) -> dict:
        self._t += 0.05
        rx = 2.0 * math.cos(self._t * 0.3)
        ry = 2.0 * math.sin(self._t * 0.3)
        out = {}
        for deg in range(0, 360, 1):
            rad = math.radians(deg)
            dx, dy = math.cos(rad), math.sin(rad)
            best = MAX_RANGE_MM / 1000.0
            for (ax, ay), (bx, by) in self.WALLS:
                t = self._ray_segment(rx, ry, dx, dy, ax, ay, bx, by)
                if t is not None and MIN_RANGE_MM / 1000 < t < best:
                    best = t
            noise = np.random.normal(0, self._noise)
            out[deg] = int((best + noise) * 1000)
        return out

    @staticmethod
    def _ray_segment(ox, oy, dx, dy, ax, ay, bx, by):
        ex, ey = bx - ax, by - ay
        denom  = dx * ey - dy * ex
        if abs(denom) < 1e-9:
            return None
        t = ((ax - ox) * ey - (ay - oy) * ex) / denom
        u = ((ax - ox) * dy - (ay - oy) * dx) / denom
        if t > 0 and 0 <= u <= 1:
            return t
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  Main SLAM class
# ─────────────────────────────────────────────────────────────────────────────

class SLAMEngine:
    def __init__(self):
        self.pose       = np.zeros(3)           # [x, y, theta] in metres/rad
        self.path       = deque(maxlen=TRAIL_LEN)
        self.prev_pts   = None
        self.grid       = OccupancyGrid(MAP_SIZE_M, MAP_RES_M)
        self.frame_times= deque(maxlen=FPS_WINDOW)
        self.icp_errors = deque(maxlen=200)
        self.frame_count= 0
        self.lidar      = None
        self.gen        = None
        self.sim        = None

    # ── Initialise sensor ──────────────────────────────────────────
    def connect(self) -> bool:
        if LIDAR_AVAILABLE:
            import serial

            # !! CHANGED: On Linux/RPi the port is released differently.
            # We use a try/except that handles both "port busy" and
            # "permission denied" gracefully.
            try:
                import serial as _s
                p = _s.Serial(PORT)
                p.close()
            except Exception as e:
                print(f"[INFO] Port pre-release skipped: {e}")

            self.lidar = PyLidar3.YdLidarG4(PORT)
            if self.lidar.Connect():
                print(f"[OK] Connected on {PORT}")
                self.gen = self.lidar.StartScanning()
                return True
            print(f"[WARN] Could not open {PORT} — falling back to simulation.")
            print(f"[HINT] Make sure you ran: sudo usermod -aG dialout $USER")

        print("[SIM] Running in simulation mode.")
        self.sim = SimulatedLidar()
        return True

    def disconnect(self):
        if self.lidar:
            try:
                self.lidar.StopScanning()
                self.lidar.Disconnect()
                print("[OK] Disconnected.")
            except Exception as e:
                print(f"[WARN] {e}")

    def _next_raw(self) -> dict | None:
        if self.gen:
            try:
                return next(self.gen)
            except StopIteration:
                return None
        return self.sim.next_scan()

    # ── Single SLAM step ───────────────────────────────────────────
    def step(self) -> bool:
        raw = self._next_raw()
        if raw is None:
            return False

        pts_local = scan_to_points(raw)
        if len(pts_local) < MIN_POINTS:
            return True

        # ── ICP scan matching ──────────────────────────────────────
        if self.prev_pts is not None and len(self.prev_pts) >= MIN_POINTS:
            delta, err = icp(pts_local, self.prev_pts)
            self.icp_errors.append(err)
            self.pose = se2_compose(self.pose, delta)
            self.pose[2] = normalize_angle(self.pose[2])

        # ── Map update ─────────────────────────────────────────────
        pts_world = transform_points(pts_local, self.pose)
        self.grid.update(self.pose, pts_world)

        self.path.append(self.pose.copy())
        self.prev_pts  = pts_local
        self.frame_count += 1
        self.frame_times.append(time.perf_counter())
        return True

    @property
    def fps(self) -> float:
        ft = self.frame_times
        if len(ft) < 2:
            return 0.0
        return (len(ft) - 1) / (ft[-1] - ft[0])

# ─────────────────────────────────────────────────────────────────────────────
#  Visualiser
# ─────────────────────────────────────────────────────────────────────────────

class SLAMVisualiser:
    BG    = "#0d0f14"
    FG    = "#e8eaf0"
    ACC   = "#00e5ff"
    ACC2  = "#ff4081"
    GRID  = "#1e2130"

    def __init__(self, engine: SLAMEngine):
        self.eng = engine
        self._build_figure()

    def _build_figure(self):
        plt.rcParams.update({
            "font.family": "monospace",
            "axes.facecolor": self.BG,
            "figure.facecolor": self.BG,
            "text.color": self.FG,
            "axes.labelcolor": self.FG,
            "xtick.color": self.FG,
            "ytick.color": self.FG,
            "axes.edgecolor": self.GRID,
            "grid.color": self.GRID,
        })

        self.fig = plt.figure(figsize=(16, 9), facecolor=self.BG)

        # !! CHANGED: set_window_title via canvas manager is wrapped in a
        # try/except — it crashes on some Linux backends (e.g. Agg/headless).
        try:
            self.fig.canvas.manager.set_window_title("YDLIDAR G4 — SLAM")
        except Exception:
            pass

        gs = gridspec.GridSpec(
            2, 3,
            figure=self.fig,
            width_ratios=[2.5, 1, 1],
            height_ratios=[1, 1],
            hspace=0.35, wspace=0.35,
            left=0.05, right=0.97,
            top=0.92,  bottom=0.08,
        )

        # ── Occupancy map (large, left) ────────────────────────────
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
        self.heading_arrow = None
        self.ax_map.legend(fontsize=7, loc="upper right",
                           facecolor=self.BG, edgecolor=self.GRID,
                           labelcolor=self.FG)

        # ── Live polar scan (top-centre) ───────────────────────────
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

        # ── ICP error history (top-right) ─────────────────────────
        self.ax_err = self.fig.add_subplot(gs[0, 2])
        self.ax_err.set_title("ICP ERROR", color=self.ACC,
                              fontsize=9, fontweight="bold")
        self.ax_err.set_xlabel("frame", fontsize=7)
        self.ax_err.set_ylabel("RMS (m)", fontsize=7)
        self.ax_err.tick_params(labelsize=7)
        self.ax_err.grid(True, lw=0.4)
        self.err_line, = self.ax_err.plot([], [], color=self.ACC2, lw=1.2)

        # ── Pose readout (bottom-centre) ───────────────────────────
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

        # ── Stats (bottom-right) ───────────────────────────────────
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
            "YDLIDAR G4  ·  ICP-SLAM  ·  Real-Time",
            fontsize=13, color=self.ACC, fontweight="bold", y=0.97,
        )

    # ── Animation callback ─────────────────────────────────────────
    def _update(self, _frame):
        self.eng.step()
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
                "", xy=(pos[0] + hl * math.cos(pos[2]),
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

        # ICP error
        errs = list(g.icp_errors)
        if errs:
            self.err_line.set_data(range(len(errs)), errs)
            self.ax_err.relim(); self.ax_err.autoscale_view()

        # Pose readout
        self.pose_text.set_text(
            f"X : {pos[0]:+.3f} m\n"
            f"Y : {pos[1]:+.3f} m\n"
            f"θ : {math.degrees(pos[2]):+.1f}°"
        )

        # Stats
        npts = len(pts) if pts is not None else 0
        mode = "HARDWARE" if g.gen else "SIMULATION"
        self.stats_text.set_text(
            f"FPS   : {g.fps:.1f}\n"
            f"Frame : {g.frame_count}\n"
            f"Pts   : {npts}\n"
            f"Mode  : {mode}"
        )

        return (self.im_map, self.path_line, self.robot_dot,
                self.scan_scatter, self.err_line,
                self.pose_text, self.stats_text)

    def run(self):
        # !! CHANGED: Agg backend has no event loop — save frames to disk instead.
        if matplotlib.get_backend() == "Agg":
            print("[INFO] Headless mode: saving map snapshots every 50 frames.")
            print("[INFO] Press Ctrl+C to stop.")
            frame = 0
            try:
                while True:
                    self._update(frame)
                    if frame % 50 == 0:
                        path = f"slam_frame_{frame:05d}.png"
                        self.fig.savefig(path, dpi=100, facecolor=self.BG)
                        print(f"[SAVE] {path}")
                    frame += 1
                    time.sleep(0.05)
            except KeyboardInterrupt:
                print("\n[STOP] Interrupted by user.")
            finally:
                self.eng.disconnect()
                plt.close(self.fig)
        else:
            ani = animation.FuncAnimation(
                self.fig, self._update,
                interval=50,
                blit=False,        # blit=True breaks imshow on some backends
                cache_frame_data=False,
            )
            try:
                plt.show()
            finally:
                ani.event_source.stop()
                self.eng.disconnect()
                plt.close(self.fig)

# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = SLAMEngine()
    if engine.connect():
        SLAMVisualiser(engine).run()
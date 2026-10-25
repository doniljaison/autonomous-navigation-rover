#!/usr/bin/env python3
"""
Hybrid Rover Control — Manual + Click-to-Navigate + SLAM + IMU
==============================================================
THE MAIN FILE — run this to operate the rover.

  Keyboard:  W/↑ Forward | S/↓ Reverse | A/← Left | D/→ Right
  Gamepad:   LB+triggers=drive  |  left stick=steer
  Space = stop/cancel auto | M = save map | Escape = quit
  Click map = set autonomous navigation goal

Usage:
  python hybrid.py                     # fresh mapping session
  python hybrid.py --map mymap.npz     # load saved map and continue
"""

import argparse
import json
import math
import socket
import sys
import time
import threading
import logging
import urllib.parse
from typing import Optional

import numpy as np

# ── Local modules ─────────────────────────────────────────────────────────────
from config import (
    LIDAR_PORT, ROVER_IP, ROVER_PORT, ROVER_ENDPOINT,
    MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED, MANUAL_SPEED, MANUAL_TURN,
    AUTO_TURN_SCALE,
    CMD_HZ, NAV_HZ, VIS_INTERVAL, WARMUP_FRAMES,
    INFLATION_RADIUS_M, GOAL_TOLERANCE_M,
    REPLAN_INTERVAL_S, STUCK_TIMEOUT_S, STUCK_ESCAPE_S,
    LOOKAHEAD_M, WHEEL_BASE_M, IMU_POLL_HZ,
    JOYSTICK_INDEX, AXIS_TURN, AXIS_FORWARD, AXIS_BACKWARD,
    ENABLE_BUTTON, REQUIRE_ENABLE, TRIGGER_MODE, DEADZONE,
)

import slam as _slam_module
from slam import SLAMEngine, SLAMVisualiser, MAP_RES_M, normalize_angle, OccupancyGrid

from nav import (
    PreconnectPool, AStarPlanner, RotateAndDriveController,
    build_http_request, clamp,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s %(asctime)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("hybrid")

# ── Override slam.py's PORT with our config ───────────────────────────────────
_slam_module.PORT = LIDAR_PORT

# ── Single shared pool for ALL motor commands ─────────────────────────────────
_pool: Optional[PreconnectPool] = None


def send_motor(left: float, right: float):
    """Send L/R to rover via shared pool. Always sends (no dedup)."""
    global _pool
    if _pool is None:
        return
    payload = json.dumps({"T": 1, "L": round(left, 3), "R": round(right, 3)})
    data = build_http_request(ROVER_IP, payload)
    _pool.send(data)


# ── IMU Reader Thread ─────────────────────────────────────────────────────────
class IMUReader:
    """
    Polls the rover's built-in IMU via HTTP at IMU_POLL_HZ.
    Sends yaw readings into SLAMEngine.set_imu_yaw() for heading fusion.
    """

    def __init__(self, engine: SLAMEngine):
        self.engine = engine
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="imu").start()
        log.info("IMU reader started @ %.0f Hz", IMU_POLL_HZ)

    def stop(self):
        self._running = False

    def _loop(self):
        period = 1.0 / IMU_POLL_HZ
        query = json.dumps({"T": 126})
        encoded = urllib.parse.quote(query, safe="")
        request = (
            f"GET {ROVER_ENDPOINT}?json={encoded} HTTP/1.0\r\n"
            f"Host: {ROVER_IP}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode("ascii")

        while self._running:
            try:
                s = socket.create_connection((ROVER_IP, ROVER_PORT), timeout=1.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.sendall(request)

                # Read full response
                chunks = []
                while True:
                    chunk = s.recv(1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                s.close()

                response = b"".join(chunks).decode("utf-8", errors="ignore")
                # Find JSON body after HTTP headers
                body_start = response.find("\r\n\r\n")
                if body_start >= 0:
                    body = response[body_start + 4:].strip()
                    if body:
                        data = json.loads(body)
                        if "y" in data:  # yaw in degrees
                            self.engine.set_imu_yaw(float(data["y"]))
            except (socket.timeout, ConnectionRefusedError, OSError):
                pass  # rover unreachable — skip this cycle
            except (json.JSONDecodeError, KeyError, ValueError):
                pass  # malformed response — skip

            time.sleep(period)


# ── SLAM Thread ───────────────────────────────────────────────────────────────
class SLAMThread:
    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="slam").start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            # Read LiDAR data OUTSIDE the lock (this blocks on serial I/O)
            ok = self.engine.step()
            if ok:
                # Brief lock only to let readers snapshot the pose safely
                with self.lock:
                    pass   # pose is already updated by step(); lock is just a fence
            time.sleep(0.001)


# ── Mode ──────────────────────────────────────────────────────────────────────
class Mode:
    MANUAL = "MANUAL"
    AUTO   = "AUTO"


class ModeManager:
    def __init__(self):
        self.mode = Mode.MANUAL
        self.goal: Optional[np.ndarray] = None

    def set_goal(self, x, y):
        self.goal = np.array([x, y])
        self.mode = Mode.AUTO
        log.info("GOAL (%.2f, %.2f) → AUTO", x, y)

    def cancel(self, reason=""):
        if self.mode == Mode.AUTO:
            log.info("→ MANUAL (%s)", reason)
        self.mode = Mode.MANUAL
        self.goal = None


# ── Gamepad Helper ────────────────────────────────────────────────────────────
_pygame_ok = False
try:
    import pygame
    _pygame_ok = True
except ImportError:
    pass


def _normalize_trigger(value: float) -> float:
    if TRIGGER_MODE == "1_to_-1":
        return (1.0 - value) / 2.0
    if TRIGGER_MODE == "0_to_1":
        return max(0.0, min(1.0, value))
    return (value + 1.0) / 2.0


class GamepadReader:
    """Optional gamepad input — runs in a thread, feeds ManualController."""

    def __init__(self, manual_ctl, mode_mgr):
        self.manual = manual_ctl
        self.mode = mode_mgr
        self._running = False
        self._js = None

    def start(self) -> bool:
        if not _pygame_ok:
            log.info("pygame not installed — gamepad disabled (keyboard still works)")
            return False
        try:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() <= JOYSTICK_INDEX:
                log.info("No gamepad found — keyboard only")
                return False
            self._js = pygame.joystick.Joystick(JOYSTICK_INDEX)
            self._js.init()
            self._running = True
            threading.Thread(target=self._loop, daemon=True, name="gamepad").start()
            log.info("Gamepad '%s' connected", self._js.get_name())
            return True
        except Exception as e:
            log.warning("Gamepad init failed: %s", e)
            return False

    def stop(self):
        self._running = False
        if _pygame_ok:
            try:
                pygame.quit()
            except Exception:
                pass

    def _loop(self):
        period = 1.0 / CMD_HZ
        while self._running:
            try:
                pygame.event.pump()
                js = self._js

                enable = js.get_button(ENABLE_BUTTON)
                if REQUIRE_ENABLE and not enable:
                    # Don't override keyboard when gamepad is idle
                    time.sleep(period)
                    continue

                # Override to manual if driving with gamepad
                if self.mode.mode == Mode.AUTO:
                    turn_raw = js.get_axis(AXIS_TURN)
                    fwd_raw = js.get_axis(AXIS_FORWARD)
                    bwd_raw = js.get_axis(AXIS_BACKWARD)
                    if abs(turn_raw) > DEADZONE or abs(fwd_raw) > 0.1 or abs(bwd_raw) > 0.1:
                        self.mode.cancel("gamepad override")
                        send_motor(0, 0)

                turn_raw = js.get_axis(AXIS_TURN)
                if abs(turn_raw) < DEADZONE:
                    turn_raw = 0.0
                turn = turn_raw * MANUAL_TURN

                forward = _normalize_trigger(js.get_axis(AXIS_FORWARD))
                backward = _normalize_trigger(js.get_axis(AXIS_BACKWARD))
                throttle = (forward - backward)

                self.manual.set_input(throttle, turn)
            except Exception:
                pass

            time.sleep(period)


# ── Manual Controller Thread ──────────────────────────────────────────────────
class ManualController:
    """Sends motor commands at CMD_HZ when in MANUAL mode."""

    def __init__(self, mode_mgr):
        self.mode = mode_mgr
        self._running = False
        self._lock = threading.Lock()
        self._fwd = 0.0
        self._turn = 0.0

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="manual").start()
        log.info("Manual ctrl started @ %d Hz", int(CMD_HZ))

    def stop(self):
        self._running = False
        send_motor(0, 0)

    def set_input(self, fwd, turn):
        with self._lock:
            self._fwd = fwd
            self._turn = turn

    def _loop(self):
        period = 1.0 / CMD_HZ
        next_time = time.perf_counter()
        while self._running:
            if self.mode.mode == Mode.MANUAL:
                with self._lock:
                    fwd, turn = self._fwd, self._turn
                throttle = fwd * MANUAL_SPEED
                left = clamp(throttle - turn, MANUAL_SPEED)
                right = clamp(throttle + turn, MANUAL_SPEED)
                send_motor(left, right)

            next_time += period
            sl = next_time - time.perf_counter()
            if sl > 0:
                time.sleep(sl)
            else:
                next_time = time.perf_counter()


# ── Auto Navigator Thread ────────────────────────────────────────────────────
FACE_THRESH = math.radians(10)   # 10° = close enough to "facing"

class AutoNavigator:
    """Direct point-to-point: face the goal, drive straight to it."""

    def __init__(self, engine, mode_mgr, slam_thread):
        self.eng = engine
        self.mode = mode_mgr
        self.slam_thread = slam_thread
        self.path: list = []
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="nav").start()
        log.info("Auto nav started @ %d Hz", int(NAV_HZ))

    def stop(self):
        self._running = False

    def _loop(self):
        period = 1.0 / NAV_HZ
        next_time = time.perf_counter()
        while self._running:
            try:
                if self.mode.mode == Mode.AUTO:
                    self._step()
                elif self.path:
                    self.path = []
            except Exception as e:
                log.error("Nav: %s", e, exc_info=True)
            next_time += period
            sl = next_time - time.perf_counter()
            if sl > 0:
                time.sleep(sl)
            else:
                next_time = time.perf_counter()

    def _step(self):
        pose = self.eng.pose.copy()
        goal = self.mode.goal
        if goal is None:
            return

        dx = goal[0] - pose[0]
        dy = goal[1] - pose[1]
        dist = math.hypot(dx, dy)

        # Goal reached
        if dist < GOAL_TOLERANCE_M:
            send_motor(0, 0)
            self.path = []
            log.info("★ GOAL REACHED (%.2f, %.2f)!", pose[0], pose[1])
            self.mode.cancel("goal reached")
            return

        if not self.path:
            self.path = [goal]
            log.info("→ (%.2f, %.2f) d=%.2f", goal[0], goal[1], dist)

        # Angle to goal
        desired = math.atan2(dy, dx)
        err = (desired - pose[2] + math.pi) % (2 * math.pi) - math.pi

        if abs(err) > math.radians(45):
            # WAY OFF — spin in place (only for big errors like target behind)
            turn = MANUAL_TURN if err < 0 else -MANUAL_TURN
            send_motor(turn, -turn)
        else:
            # DRIVE FORWARD + STEER — stay on course without stopping
            # Stronger correction = straighter line
            steer = clamp(-err * 0.6, MANUAL_TURN)
            spd = MANUAL_SPEED
            send_motor(spd + steer, spd - steer)


# ── Visualiser ────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except ImportError:
    print("pip install matplotlib"); sys.exit(1)


class HybridVisualiser(SLAMVisualiser):
    def __init__(self, engine, mode_mgr, slam_thr, auto_nav, manual_ctl, imu_reader,
                 gamepad=None):
        super().__init__(engine)
        self.mode_mgr = mode_mgr
        self.slam_thr = slam_thr
        self.auto_nav = auto_nav
        self.manual_ctl = manual_ctl
        self.imu_reader = imu_reader
        self.gamepad = gamepad
        self._keys: set = set()

        # Nav path overlay
        self.path_overlay, = self.ax_map.plot(
            [], [], "--", color="#00e5ff", lw=1.2, alpha=0.7, zorder=6)
        self.goal_marker, = self.ax_map.plot(
            [], [], "X", color="#ff4081", ms=14, mew=2, zorder=8)

        # Mode HUD
        self.mode_hud = self.ax_map.text(
            0.02, 0.98, "", transform=self.ax_map.transAxes,
            fontsize=10, va="top", fontfamily="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=self.BG, edgecolor=self.GRID_COL, alpha=0.85))

        # Help text
        self.ax_map.text(
            0.02, 0.02, "WASD:drive | Click:goal | Space:stop | M:save | Esc:quit",
            transform=self.ax_map.transAxes,
            fontsize=7, color="#7a8499", va="bottom", fontfamily="monospace")

        # IMU indicator
        self.imu_text = self.ax_map.text(
            0.98, 0.98, "", transform=self.ax_map.transAxes,
            fontsize=8, va="top", ha="right", fontfamily="monospace",
            color="#7a8499")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_key_press(self, event):
        if event.key == "escape":
            self._shutdown(); return
        if event.key == " ":
            self.mode_mgr.cancel("e-stop")
            send_motor(0, 0)
            self.auto_nav.path = []
            self._keys.clear()
            self.manual_ctl.set_input(0, 0)
            return
        if event.key == "m":
            # Save map
            filename = f"map_{int(time.time())}.npz"
            try:
                self.eng.grid.save(filename)
                log.info("Map saved as %s", filename)
            except Exception as e:
                log.error("Map save failed: %s", e)
            return
        self._keys.add(event.key)
        if event.key in ("w", "a", "s", "d", "up", "down", "left", "right"):
            if self.mode_mgr.mode == Mode.AUTO:
                self.mode_mgr.cancel("manual override")
                send_motor(0, 0)
                self.auto_nav.path = []
        self._push()

    def _on_key_release(self, event):
        self._keys.discard(event.key)
        self._push()

    def _push(self):
        fwd, turn = 0.0, 0.0
        if "w" in self._keys or "up" in self._keys: fwd += 1.0
        if "s" in self._keys or "down" in self._keys: fwd -= 1.0
        if "a" in self._keys or "left" in self._keys: turn -= MANUAL_TURN
        if "d" in self._keys or "right" in self._keys: turn += MANUAL_TURN
        self.manual_ctl.set_input(fwd, turn)

    def _on_click(self, event):
        if event.inaxes != self.ax_map or event.button != 1:
            return
        if self.eng.frame_count < WARMUP_FRAMES:
            log.warning("SLAM warming up (%d/%d)", self.eng.frame_count, WARMUP_FRAMES)
            return
        self._keys.clear()
        self.manual_ctl.set_input(0, 0)
        self.mode_mgr.set_goal(event.xdata, event.ydata)

    def _shutdown(self):
        log.info("Shutting down…")
        self.manual_ctl.stop()
        self.auto_nav.stop()
        self.imu_reader.stop()
        if self.gamepad:
            self.gamepad.stop()
        send_motor(0, 0)
        self.slam_thr.stop()
        self.eng.disconnect()
        if _pool:
            _pool.close()
        try:
            self._anim.event_source.stop()
        except Exception:
            pass
        plt.close(self.fig)

    def _update(self, _frame):
        pos = self.eng.pose.copy()
        g = self.eng
        pts = g.prev_pts

        self.im_map.set_data(g.grid.to_image())
        if g.path:
            self.path_line.set_data([p[0] for p in g.path], [p[1] for p in g.path])
            self.robot_dot.set_data([pos[0]], [pos[1]])
            if self.heading_arrow:
                self.heading_arrow.remove()
            hl = 0.4
            self.heading_arrow = self.ax_map.annotate("",
                xy=(pos[0]+hl*math.cos(pos[2]), pos[1]+hl*math.sin(pos[2])),
                xytext=(pos[0], pos[1]),
                arrowprops=dict(arrowstyle="->", color=self.ACC, lw=1.5))

        if pts is not None and len(pts):
            angles = np.arctan2(pts[:, 1], pts[:, 0])
            dists = np.hypot(pts[:, 0], pts[:, 1])
            self.scan_scatter.set_offsets(np.c_[angles, dists])
            self.scan_scatter.set_array(dists)


        self.pose_text.set_text(
            f"X:{pos[0]:+.3f}m\nY:{pos[1]:+.3f}m\nθ:{math.degrees(pos[2]):+.1f}°")
        npts = len(pts) if pts is not None else 0
        self.stats_text.set_text(
            f"FPS:{g.fps:.1f}\nFrame:{g.frame_count}\nPts:{npts}")

        # Nav path overlay
        p = self.auto_nav.path
        if p:
            self.path_overlay.set_data([x[0] for x in p], [x[1] for x in p])
        else:
            self.path_overlay.set_data([], [])

        # Goal marker
        gl = self.mode_mgr.goal
        if gl is not None:
            self.goal_marker.set_data([gl[0]], [gl[1]])
        else:
            self.goal_marker.set_data([], [])

        # Mode HUD
        mode = self.mode_mgr.mode
        extra = ""
        if mode == Mode.AUTO and gl is not None:
            d = math.hypot(gl[0]-pos[0], gl[1]-pos[1])
            extra = f" d={d:.2f}m wp={len(p)}"
        self.mode_hud.set_text(f"{mode}{extra}")
        self.mode_hud.set_color("#00e676" if mode == Mode.MANUAL else "#00e5ff")

        # IMU indicator
        imu_yaw = self.eng.imu_yaw
        if imu_yaw is not None:
            self.imu_text.set_text(f"IMU:{math.degrees(imu_yaw):+.1f}°")
            self.imu_text.set_color("#00e676")
        else:
            self.imu_text.set_text("IMU:--")
            self.imu_text.set_color("#7a8499")

        return (self.im_map, self.path_line, self.robot_dot,
                self.scan_scatter, self.pose_text, self.stats_text,
                self.path_overlay, self.goal_marker, self.mode_hud,
                self.imu_text)

    def run(self):
        self.slam_thr.start()
        self.imu_reader.start()
        self.manual_ctl.start()
        self.auto_nav.start()
        self._anim = animation.FuncAnimation(
            self.fig, self._update, interval=VIS_INTERVAL,
            blit=False, cache_frame_data=False)
        try:
            plt.show()
        finally:
            self._shutdown()


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Rover Control: Manual WASD + Click-to-Navigate + SLAM")
    parser.add_argument("--map", type=str, default=None,
                        help="Load a saved map (.npz) to continue mapping or navigate")
    args = parser.parse_args()

    # ── SLAM engine ───────────────────────────────────────────────────────────
    engine = SLAMEngine()

    # Load saved map if provided
    if args.map:
        try:
            engine.grid = OccupancyGrid.load(args.map)
            log.info("Loaded map from %s", args.map)
        except Exception as e:
            log.error("Failed to load map: %s", e)
            sys.exit(1)

    # Connect LiDAR
    if not engine.connect():
        print("\n[FATAL] LiDAR not found. Check LIDAR_PORT in config.py + USB.\n")
        sys.exit(1)

    # ── Motor pool ────────────────────────────────────────────────────────────
    global _pool
    _pool = PreconnectPool(ROVER_IP, ROVER_PORT)
    log.info("Motor pool → %s:%d", ROVER_IP, ROVER_PORT)

    # Quick connectivity test
    send_motor(0, 0)
    time.sleep(0.3)
    log.info("Rover connection test: pool ready")

    # ── Create all components ─────────────────────────────────────────────────
    mode_mgr   = ModeManager()
    slam_thr   = SLAMThread(engine)
    manual_ctl = ManualController(mode_mgr)
    auto_nav   = AutoNavigator(engine, mode_mgr, slam_thr)
    imu_reader = IMUReader(engine)

    # Optional gamepad
    gamepad = GamepadReader(manual_ctl, mode_mgr)
    gp_ok = gamepad.start()

    # ── Visualiser ────────────────────────────────────────────────────────────
    vis = HybridVisualiser(
        engine, mode_mgr, slam_thr, auto_nav, manual_ctl, imu_reader,
        gamepad=gamepad if gp_ok else None)
    vis.run()


if __name__ == "__main__":
    main()
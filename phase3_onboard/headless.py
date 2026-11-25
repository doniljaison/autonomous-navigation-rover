"""
Headless console for the on-board runtime (no web server).
==========================================================
    python -m phase3_onboard.main --headless

Type commands at the `rover>` prompt (over SSH):

    goal X Y          drive autonomously to (X, Y) in metres, map frame
    stop              stop now, cancel any goal          (an empty line also stops)
    status            pose, SLAM counters, link / LiDAR health
    save              save the map to maps/map_<time>.npz
    snap              write the current map as a PNG into the run's log folder
    resume            clear SLAM LOST after you have driven back to known area
    spin DUTY SEC     TEST: turn in place CCW (left) at DUTY (0.05..0.45) for SEC (<= 5) s
                      negative DUTY = clockwise.  Use to find the turn-in-place floor.
    fwd DUTY SEC      TEST: drive straight at DUTY (-0.45..0.45) for SEC (<= 5) s, proximity-limited
    help / quit

Safety is unchanged from the web version:
  * the link writer streams L/R at CMD_HZ and sends 0,0 when an intent is older than its TTL;
    goals are driven by the nav thread (intent TTL 0.5 s), tests re-send every 0.1 s (TTL 0.3 s);
  * Ctrl-C, `quit`, end of input (Ctrl-D) and a dropped SSH session (SIGHUP) all stop the rover,
    save the map and close the log;
  * the ESP32's own 3 s timeout remains the last line of defence.
Do NOT start this under nohup / disown: then an SSH drop would no longer stop the rover.
"""

from __future__ import annotations

import math
import os
import threading
import time

HELP = __doc__.split("Type commands", 1)[1].split("Safety is", 1)[0]


class _Test:
    """One timed open-loop motion at a fixed duty (spin / fwd), cancellable."""

    def __init__(self, rt):
        self.rt = rt
        self.cancel = threading.Event()
        self.th = None

    def running(self):
        return self.th is not None and self.th.is_alive()

    def start(self, kind, duty, sec):
        self.cancel.clear()
        self.th = threading.Thread(target=self._run, args=(kind, duty, sec), daemon=True, name="test")
        self.th.start()

    def _run(self, kind, duty, sec):
        rt = self.rt
        th0 = float(rt.eng.pose[2])
        p0 = rt.eng.pose[:2].copy()
        acc0 = rt.eng.snapshot()["accepted"]
        turned = 0.0
        last_th = th0
        t0 = time.monotonic()
        blocked_any = False
        while not self.cancel.is_set() and time.monotonic() - t0 < sec:
            if kind == "spin":
                L, R = -duty, duty                 # robot frame: +duty = CCW (left)
            else:
                L, R, blocked = rt.prox.limit(duty, duty)
                blocked_any |= blocked
            rt.link.set_intent(L, R, "test", ttl=0.3)
            time.sleep(0.1)
            th = float(rt.eng.pose[2])
            turned += math.atan2(math.sin(th - last_th), math.cos(th - last_th))
            last_th = th
        rt.link.stop_now("test_end")
        dt = time.monotonic() - t0
        time.sleep(0.4)                            # let the last scans land
        th = float(rt.eng.pose[2])
        turned += math.atan2(math.sin(th - last_th), math.cos(th - last_th))
        snap = rt.eng.snapshot()
        moved = float(math.hypot(*(rt.eng.pose[:2] - p0)))
        msg = (f"[test {kind} duty={duty:+.2f} {dt:.1f}s] SLAM heading change {math.degrees(turned):+.1f} deg "
               f"({math.degrees(turned) / max(dt, 1e-3):+.1f} deg/s), moved {moved:.2f} m, "
               f"frames accepted +{snap['accepted'] - acc0}, reject_rate {snap['reject_rate']:.2f}, "
               f"lost={snap['lost']}" + ("  [proximity limited]" if blocked_any else "")
               + ("  [cancelled]" if self.cancel.is_set() else ""))
        rt.note(msg)
        print(msg, flush=True)


def _status(rt):
    s = rt.eng.snapshot()
    x, y, th = (float(v) for v in s["pose"])
    ln = rt.link.status()
    li = rt.lidar
    with rt.lock:
        mode, goal = rt.mode, rt.goal
    lines = [
        f"mode {mode} phase {rt.nav_phase}  goal {None if goal is None else [round(float(g), 2) for g in goal]}",
        f"pose x {x:+.2f} m  y {y:+.2f} m  heading {math.degrees(th):+.1f} deg   lost={s['lost']}",
        f"slam {rt.slam_hz:.1f} Hz  frames {s['frames']}  accepted {s['accepted']}  rejected {s['rejected']}  "
        f"skipped {s['skipped']}  reject_rate {s['reject_rate']:.2f}  last {s['last']}",
        f"gyro ready={s['gyro_ready']} bias {math.degrees(s['gyro_bias']):+.3f} deg/s  scale_corr {s['gyro_scale_corr']:.3f}",
        f"link rx_age {ln['rx_age_s']} s  last_cmd {ln['last_cmd']}  watchdog_trips {ln['watchdog_trips']}  "
        f"tx_errors {ln['tx_errors']}",
        f"lidar {li.rate_hz:.2f} Hz  points {li.last_n}  dropped {li.dropped}  error {li.error}",
        f"log {rt.logger.dir if rt.logger else '(off)'}",
    ]
    print("\n".join(lines), flush=True)


def _snap_png(rt):
    from phase3_onboard.imgutil import map_rgb, png_bytes
    P = rt.P
    rgb = map_rgb(rt.eng.map_copy(), P.MAP_OCC_THRESH, P.MAP_FREE_THRESH)
    out_dir = rt.logger.dir if rt.logger else "."
    path = os.path.join(out_dir, f"map_{time.strftime('%H%M%S')}.png")
    with open(path, "wb") as f:
        f.write(png_bytes(rgb, level=1))
    x, y, th = (float(v) for v in rt.eng.pose)
    rt.note(f"map PNG {path} (pose {x:+.2f} {y:+.2f} {math.degrees(th):+.1f} deg)")
    print(path, flush=True)


def console(rt):
    test = _Test(rt)
    print("headless console. 'help' lists commands; empty line = stop; Ctrl-C / quit = stop and exit.", flush=True)
    while True:
        try:
            line = input("rover> ")
        except EOFError:          # Ctrl-D, closed pipe, or SSH gone
            print("\nend of input -> stop", flush=True)
            break
        parts = line.strip().split()
        if not parts:
            test.cancel.set()
            rt.stop()
            continue
        cmd, args = parts[0].lower(), parts[1:]
        try:
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd in ("h", "help", "?"):
                print(HELP, flush=True)
            elif cmd in ("s", "stop"):
                test.cancel.set()
                rt.stop()
            elif cmd == "goal":
                if test.running():
                    print("a test is running; 'stop' first", flush=True)
                    continue
                x, y = float(args[0]), float(args[1])
                print("goal accepted" if rt.set_goal(x, y) else "goal refused (SLAM LOST? use 'resume')", flush=True)
            elif cmd == "status":
                _status(rt)
            elif cmd == "save":
                print(rt.save_map(), flush=True)
            elif cmd == "snap":
                _snap_png(rt)
            elif cmd == "resume":
                rt.clear_lost()
            elif cmd in ("spin", "fwd"):
                duty, sec = float(args[0]), float(args[1])
                if not (0.05 <= abs(duty) <= 0.45) or not (0 < sec <= 5.0):
                    print("DUTY must be 0.05..0.45 (sign = direction), SEC 0..5", flush=True)
                    continue
                with rt.lock:
                    auto = rt.mode != "MANUAL"
                if auto:
                    print("autonomous goal active; 'stop' first", flush=True)
                    continue
                if test.running():
                    print("a test is already running", flush=True)
                    continue
                test.start(cmd, duty, sec)
            else:
                print(f"unknown command '{cmd}' (help)", flush=True)
        except (IndexError, ValueError):
            print("bad arguments (help)", flush=True)
        except Exception as e:                       # never leave the console with the rover moving
            test.cancel.set()
            rt.stop()
            print(f"error: {e} -> stopped", flush=True)
    test.cancel.set()
    rt.stop()
    try:
        print("map saved:", rt.save_map(), flush=True)
    except Exception as e:
        print("map save failed:", e, flush=True)

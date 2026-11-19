"""
G4 bring-up and soak test (Day 3 / Gate G1).
============================================
    python -m phase3_onboard.tools.lidar_check --minutes 10          # soak, logs scans
    python -m phase3_onboard.tools.lidar_check --convention           # angle convention helper

Soak reports: scan rate, points per revolution, angle direction and step,
time increment, failed reads, gaps, and `vcgencmd get_throttled` / temperature
every 5 s.  PASS needs: rate within 15 % of LIDAR_SCAN_HZ, median points > 360,
no gap > 1 s, throttled == 0x0 throughout.

--convention: put a box ~0.5 m in FRONT of the rover, then on its LEFT; the
tool prints the robot-frame bearing of the nearest return using the current
LIDAR_ANGLE_SIGN / OFFSET / mount yaw.  Front must read ~0 deg, left ~+90 deg.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from phase3_onboard.lidar import make_lidar
from phase3_onboard.logger import RunLogger
from phase3_onboard.slam.params import from_config
from phase3_onboard.sysmon import decode_throttled, read_temp_c, read_throttled


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--driver", default=None)
    ap.add_argument("--port", default=None)
    ap.add_argument("--convention", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    ov = {}
    if a.driver:
        ov["LIDAR_DRIVER"] = a.driver
    if a.port:
        ov["LIDAR_PORT"] = a.port
    P = from_config(**ov)
    src = make_lidar(P)
    src.start()
    try:
        if a.convention:
            convention(P, src)
        else:
            soak(P, src, a.minutes, not a.no_log)
    finally:
        src.stop()


def convention(P, src):
    print("Ctrl-C to stop.  Robot frame: 0 deg = forward, +90 deg = LEFT.")
    while True:
        s = src.read()
        if s is None:
            continue
        r = s.range.astype(float)
        ok = r > 0.1
        if not ok.any():
            continue
        ang = P.LIDAR_ANGLE_SIGN * s.angle.astype(float) + P.LIDAR_ANGLE_OFFSET_RAD + P.LIDAR_MOUNT_YAW_RAD
        i = np.argmin(np.where(ok, r, 1e9))
        bearing = math.degrees((ang[i] + math.pi) % (2 * math.pi) - math.pi)
        d = np.diff(np.unwrap(s.angle.astype(float)))
        direction = "increasing" if np.median(d) > 0 else "decreasing"
        print(f"nearest {r[i]:.2f} m at robot bearing {bearing:+7.1f} deg | raw angle "
              f"{math.degrees(s.angle[i]):+7.1f} deg | raw angles {direction} in scan order | n={s.n}",
              end="\r", flush=True)


def soak(P, src, minutes, log):
    lg = RunLogger(P.LOG_DIR, "lidar_soak", meta={"tool": "lidar_check"}) if log else None
    t_end = time.monotonic() + minutes * 60
    n_pts, periods, fails = [], [], 0
    steps, tincs, dirs = [], [], []
    thr_all = 0
    last = None
    last_sys = 0.0
    temps = []
    print(f"soaking {minutes:.1f} min ... (Ctrl-C to stop early)")
    try:
        while time.monotonic() < t_end:
            s = src.read()
            if s is None:
                fails += 1
                continue
            if lg:
                lg.scan(s)
            n_pts.append(s.n)
            if last is not None:
                periods.append(s.t_end - last)
            last = s.t_end
            if s.n > 10:
                d = np.diff(np.unwrap(s.angle.astype(float)))
                steps.append(float(np.median(np.abs(d))))
                dirs.append(1 if np.median(d) > 0 else -1)
            tincs.append(float(s.meta.get("tinc", 0) or 0))
            now = time.monotonic()
            if now - last_sys > 5:
                last_sys = now
                th = read_throttled()
                tc = read_temp_c()
                if th:
                    thr_all |= th
                if tc:
                    temps.append(tc)
                rate = 1 / np.mean(periods[-50:]) if periods else 0
                print(f"  {len(n_pts):6d} scans  {rate:5.2f} Hz  pts {int(np.median(n_pts[-50:]))}  "
                      f"fails {fails}  temp {tc}  throttled {hex(th) if th is not None else 'n/a'}")
    except KeyboardInterrupt:
        pass
    if lg:
        lg.close()
    per = np.array(periods)
    rate = 1 / per.mean() if len(per) else 0
    print("\n=== LiDAR soak summary ===")
    print(f"scans {len(n_pts)}, rate {rate:.2f} Hz (target {P.LIDAR_SCAN_HZ}), "
          f"max gap {per.max() if len(per) else 0:.3f} s, failed reads {fails}")
    print(f"points/rev median {np.median(n_pts) if n_pts else 0:.0f} (min {min(n_pts) if n_pts else 0}), "
          f"angle step {math.degrees(np.median(steps)) if steps else 0:.3f} deg, "
          f"raw angle direction {'increasing' if np.mean(dirs) > 0 else 'decreasing'}, "
          f"time increment {np.median(tincs) * 1e6 if tincs else 0:.1f} us")
    print(f"temperature max {max(temps) if temps else 'n/a'} C; throttled flags seen: "
          f"{hex(thr_all)} {decode_throttled(thr_all)}")
    ok = (len(per) > 0 and abs(rate - P.LIDAR_SCAN_HZ) / P.LIDAR_SCAN_HZ < 0.15 and
          np.median(n_pts) > 360 and per.max() < 1.0 and thr_all == 0)
    print("G1 LiDAR part:", "PASS" if ok else "FAIL")
    if lg:
        print("log:", lg.dir)


if __name__ == "__main__":
    main()

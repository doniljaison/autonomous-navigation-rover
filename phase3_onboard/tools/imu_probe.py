"""
Capture and summarise the raw rover stream (Day 2 / Day 4) — do this FIRST.
==========================================================================
    python -m phase3_onboard.tools.imu_probe --link uart --seconds 20
    python -m phase3_onboard.tools.imu_probe --link http --host 192.168.x.y
    python -m phase3_onboard.tools.imu_probe --link uart --handrot

Prints every message type seen, its rate, its keys and value ranges, and saves
the raw lines to a run directory (replayable).  The rover does not move
(motor stream is L=R=0).

--handrot : guided hand-rotation test (rover on the floor, wheels free):
            rotate it by hand +90 deg (CCW seen from above), wait, then -90 deg.
            Prints the integrated gyro and the firmware yaw change for each
            step, and suggests IMU_GYRO_SCALE / IMU_GYRO_SIGN / IMU_YAW_SIGN.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict

from phase3_onboard.logger import RunLogger
from phase3_onboard.rover_link import RoverLink, make_transport
from phase3_onboard.slam.params import from_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--link", choices=["uart", "http", "sim"], default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", default=None, help="UART device")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--poll-hz", type=float, default=None, help="T:126 poll rate (default config)")
    ap.add_argument("--no-feedback", action="store_true", help="do not enable T:131 streaming")
    ap.add_argument("--handrot", action="store_true")
    a = ap.parse_args()
    ov = {}
    if a.link:
        ov["LINK_TRANSPORT"] = a.link
    if a.host:
        ov["ROVER_HTTP_HOST"] = a.host
    if a.port:
        ov["UART_PORT"] = a.port
    if a.poll_hz is not None:
        ov["IMU_POLL_HZ"] = a.poll_hz
    if a.no_feedback:
        ov["FEEDBACK_STREAM"] = False
    P = from_config(**ov)
    lg = RunLogger(P.LOG_DIR, "imu_probe", meta={"tool": "imu_probe"})
    lines = []
    link = RoverLink(make_transport(P), P, lg)
    link.subscribe(lambda t, l: lines.append((t, l)))
    link.start()
    try:
        if a.handrot:
            handrot(P, lines)
        else:
            print(f"capturing {a.seconds:.0f} s from {P.LINK_TRANSPORT} ...")
            time.sleep(a.seconds)
    finally:
        link.stop()
        lg.close()
    summarise(lines)
    print(f"\nraw stream saved in {lg.dir}")


def summarise(lines):
    by = defaultdict(list)
    keys = defaultdict(lambda: defaultdict(list))
    other = 0
    for t, l in lines:
        try:
            d = json.loads(l)
        except Exception:
            other += 1
            continue
        T = d.get("T", "?")
        by[T].append(t)
        for k, v in d.items():
            if isinstance(v, (int, float)):
                keys[T][k].append(float(v))
    print("\n=== message types ===")
    for T, ts in sorted(by.items(), key=lambda kv: str(kv[0])):
        dur = (ts[-1] - ts[0]) if len(ts) > 1 else 0
        rate = (len(ts) - 1) / dur if dur > 0 else 0
        print(f"T={T}: {len(ts)} msgs, {rate:.1f} Hz")
        for k, vs in keys[T].items():
            if k == "T":
                continue
            print(f"    {k:>6}: min {min(vs):10.3f}  max {max(vs):10.3f}  last {vs[-1]:10.3f}")
    if other:
        print(f"non-JSON lines: {other}")
    if lines:
        print("\nexample lines:")
        seen = set()
        for _, l in lines:
            try:
                T = json.loads(l).get("T")
            except Exception:
                continue
            if T not in seen:
                seen.add(T)
                print("   ", l[:200])


def handrot(P, lines):
    def window(t0, t1, key):
        vals = []
        for t, l in lines:
            if t0 <= t <= t1:
                try:
                    d = json.loads(l)
                except Exception:
                    continue
                if d.get("T") in P.IMU_REPLY_TYPES and key in d:
                    vals.append((t, float(d[key])))
        return vals

    print("Keep the rover still for 3 s ...")
    time.sleep(3.0)
    t_still = time.monotonic()
    still = window(t_still - 3, t_still, P.IMU_GYRO_Z_KEY or "gz")
    bias = sum(v for _, v in still) / len(still) if still else 0.0
    results = []
    for target in (+90, -90):
        input(f"\nPress Enter, then rotate the rover by hand {target:+d} deg "
              f"({'CCW' if target > 0 else 'CW'} seen from above) within ~4 s and hold still ...")
        t0 = time.monotonic()
        time.sleep(5.0)
        t1 = time.monotonic()
        g = window(t0, t1, P.IMU_GYRO_Z_KEY or "gz")
        integ = 0.0
        for (ta, va), (tb, vb) in zip(g, g[1:]):
            integ += 0.5 * ((va - bias) + (vb - bias)) * (tb - ta)
        y = window(t0, t1, P.IMU_YAW_KEY or "y")
        dy = None
        if len(y) > 1:
            dy = ((y[-1][1] - y[0][1] + 180) % 360) - 180
        print(f"  raw gyro integral: {integ:.2f} (raw units*s)   firmware yaw change: {dy}")
        results.append((target, integ, dy))
    print("\n=== suggestion ===")
    s = [r[1] / r[0] for r in results if abs(r[1]) > 1e-6]
    if s:
        k = sum(s) / len(s)       # raw units per degree
        sign = 1 if k > 0 else -1
        print(f"raw gyro integral per degree = {k:.4f}")
        print(f"IMU_GYRO_SIGN  = {sign:+.0f}")
        print(f"IMU_GYRO_SCALE = math.radians(1.0) / {abs(k):.4f}   # = {math.radians(1.0) / abs(k):.6g} rad/s per raw unit")
    ys = [r[2] / r[0] for r in results if r[2] is not None]
    if ys:
        print(f"firmware yaw per degree = {sum(ys) / len(ys):.3f}  -> IMU_YAW_SIGN = {1 if sum(ys) > 0 else -1:+d}")
    print("Write these into phase3_onboard/config_local.py and record the result in docs/conventions.md.")


if __name__ == "__main__":
    main()

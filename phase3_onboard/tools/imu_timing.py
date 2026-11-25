#!/usr/bin/env python3
"""
Are IMU lines arriving in bursts (reader thread starved) during a live run?

    python -m phase3_onboard.tools.imu_timing runs/<run>

Reads <run>/events.jsonl, takes the received lines that carry the gyro (T:1002 by default),
and reports:
  * arrival gaps and how many lines share an identical arrival time (a "burst": they were read
    in one go after the reader thread could not run).  The ESP32 sends T:1002 about every 40 ms,
    so a healthy link shows gaps of 30-50 ms and almost no zero gaps;
  * the gyro rotation that a dt-based integrator would LOSE, if samples of a burst get dt ~ 0;
  * whether bursts line up with SLAM steps (slam events), which points at the GIL.
Read-only; works on the Pi with only numpy.
"""
import json
import os
import re
import sys

import numpy as np

KIND_KEYS = ("ev", "type", "kind", "k", "event", "e")


def kind_of(e):
    for k in KIND_KEYS:
        if k in e and isinstance(e[k], str):
            return e[k]
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run = sys.argv[1]
    tnum = sys.argv[2] if len(sys.argv) > 2 else "1002"
    tag = f'"T":{tnum}'
    pat = re.compile(r'"T"\s*:\s*' + tnum + r'\b')
    path = os.path.join(run, "events.jsonl")
    t_imu, gz, t_slam, slam_keys, first = [], [], [], set(), []
    kinds = {}
    with open(path) as f:
        for i, l in enumerate(f):
            try:
                e = json.loads(l)
            except json.JSONDecodeError:
                continue
            if i < 3:
                first.append(sorted(e.keys()))
            k = kind_of(e)
            kinds[k] = kinds.get(k, 0) + 1
            if k == "rx" and pat.search(str(e.get("line", ""))):
                t_imu.append(float(e["t"]))
                try:
                    gz.append(float(json.loads(e["line"]).get("gz", np.nan)))
                except Exception:
                    gz.append(np.nan)
            elif k == "slam":
                t_slam.append(float(e.get("t", np.nan)))
                slam_keys |= set(e.keys())
    print("event kinds:", kinds)
    if not t_imu:
        print("no rx lines containing", tag, "- first records' keys:", first)
        sys.exit(1)
    t = np.array(t_imu)
    g = np.array(gz)
    order = np.argsort(t, kind="stable")
    t, g = t[order], g[order]
    gaps = np.diff(t) * 1e3
    zero = gaps < 1.0
    print(f"\nIMU lines ({tag}): {len(t)} over {t[-1] - t[0]:.1f} s -> {len(t) / (t[-1] - t[0]):.1f} Hz")
    print("arrival gap ms: p50 %.1f  p90 %.1f  p95 %.1f  p99 %.1f  max %.0f"
          % tuple(np.percentile(gaps, [50, 90, 95, 99]).tolist() + [gaps.max()]))
    print(f"gaps < 1 ms (same read burst): {zero.mean() * 100:.1f} %   gaps > 80 ms: {(gaps > 80).mean() * 100:.1f} %")
    # burst sizes
    sizes, n = [], 1
    for z in zero:
        if z:
            n += 1
        else:
            sizes.append(n)
            n = 1
    sizes.append(n)
    sizes = np.array(sizes)
    print("lines per burst: p50 %d  p95 %d  max %d" % (np.median(sizes), np.percentile(sizes, 95), sizes.max()))

    # what a sample-to-sample dt integrator would see vs a fixed 40 ms period
    ok = np.isfinite(g)
    if ok.sum() > 10:
        gg, tt = g[ok], t[ok]
        dt = np.clip(np.diff(tt), 0, 0.2)
        rot_dt = np.sum(np.abs(gg[1:]) * dt)
        nominal = np.median(np.diff(tt)[np.diff(tt) > 0.005]) if np.any(np.diff(tt) > 0.005) else 0.04
        rot_fixed = np.sum(np.abs(gg[1:])) * nominal
        print(f"\n|gz| integrated with arrival dt: {rot_dt:.0f} deg   with a fixed {nominal * 1e3:.0f} ms period: {rot_fixed:.0f} deg"
              "  (raw units, assumed deg/s)")
        moving = np.abs(gg[1:]) > 10
        if moving.any():
            z2 = np.diff(tt) < 0.001
            print(f"while |gz| > 10 deg/s (turning): {z2[moving].mean() * 100:.1f} % of samples arrived with ~0 gap")

    if t_slam:
        ts = np.array(t_slam)
        ts = ts[np.isfinite(ts)]
        print(f"\nslam events: {len(ts)}   keys: {sorted(slam_keys)}")
        big = t[1:][gaps > 80]
        if len(big) and len(ts):
            idx = np.searchsorted(ts, big)
            near = []
            for b, i in zip(big, idx):
                d = min(abs(b - ts[j]) for j in (i - 1, i) if 0 <= j < len(ts))
                near.append(d)
            near = np.array(near) * 1e3
            print(f"long IMU gaps (>80 ms) ending within 30 ms of a slam event: {(near < 30).mean() * 100:.0f} %")
    else:
        print("\nno slam events in this log")


if __name__ == "__main__":
    main()

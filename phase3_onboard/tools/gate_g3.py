"""
Gate G3 (Day 10): real time on the Pi 4B.
=========================================
    python -m phase3_onboard.tools.gate_g3 runs/<live run recorded by main.py>

Checks (on a >= 20 min live run):
  * SLAM step p95 < 60 % of the scan period
  * no throttling (vcgencmd flags 0x0 at every sample) and temperature stable
    (max < 80 C, last-5-min mean not rising > 2 C vs the previous 5 min)
  * no memory growth (RSS last 5 min vs first 5 min after warm-up < +20 MB)
  * frame drops (scans logged but not processed by SLAM) reported
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from phase3_onboard.logger import RunReader


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--min-minutes", type=float, default=20.0)
    a = ap.parse_args()
    rd = RunReader(a.run)
    ev = rd.events()
    slam = [e for e in ev if e["k"] == "slam"]
    sysv = [e for e in ev if e["k"] == "sys"]
    scans_t = [s.t_end for s in rd.scans()]
    res = {}
    ok = True
    if len(scans_t) < 10 or not slam:
        print("not enough data"); sys.exit(1)
    dur = (scans_t[-1] - scans_t[0]) / 60
    period = float(np.median(np.diff(scans_t)))
    tot = np.array([e["ms"].get("total", np.nan) for e in slam], float)
    p95 = float(np.nanpercentile(tot, 95))
    res["duration_min"] = round(dur, 1)
    res["scan_period_ms"] = round(period * 1e3, 1)
    res["step_ms"] = {q: round(float(np.nanpercentile(tot, q)), 2) for q in (50, 95, 99)}
    res["step_p95_pct_of_period"] = round(100 * p95 / (period * 1e3), 1)
    ok &= p95 < 0.6 * period * 1e3
    lat = [e.get("lat") for e in slam if e.get("lat") is not None]
    if lat:
        res["scan_end_to_pose_ms_p95"] = round(1e3 * float(np.percentile(lat, 95)), 1)
    res["frames_logged"] = len(scans_t)
    res["frames_processed"] = len(slam)
    res["frame_drops"] = len(scans_t) - len(slam)
    if sysv:
        t = np.array([e["t"] for e in sysv]); t -= t[0]
        temp = np.array([e.get("temp") or np.nan for e in sysv], float)
        rss = np.array([e.get("rss_mb") or np.nan for e in sysv], float)
        thr = [e.get("throttled") for e in sysv if e.get("throttled") not in (None, "0x0")]
        res["temp_max_c"] = None if np.all(np.isnan(temp)) else round(float(np.nanmax(temp)), 1)
        last5 = temp[t > t[-1] - 300]; prev5 = temp[(t > t[-1] - 600) & (t <= t[-1] - 300)]
        if len(last5) and len(prev5):
            res["temp_trend_c"] = round(float(np.nanmean(last5) - np.nanmean(prev5)), 2)
            ok &= res["temp_trend_c"] < 2.0
        if res["temp_max_c"] is not None:
            ok &= res["temp_max_c"] < 80.0
        res["throttled_samples"] = len(thr)
        ok &= len(thr) == 0
        first = rss[(t > 60) & (t < 360)]; last = rss[t > t[-1] - 300]
        if len(first) and len(last):
            res["rss_growth_mb"] = round(float(np.nanmean(last) - np.nanmean(first)), 1)
            ok &= res["rss_growth_mb"] < 20
        cpu = np.array([e.get("cpu") or np.nan for e in sysv], float)
        res["cpu_pct_mean"] = round(float(np.nanmean(cpu)), 1)
    if dur < a.min_minutes:
        res["note"] = f"run shorter than {a.min_minutes} min"
        ok = False
    print(json.dumps(res, indent=1))
    print("G3:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

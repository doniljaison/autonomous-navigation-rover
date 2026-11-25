#!/usr/bin/env python3
"""
Does the LiDAR driver freeze the other Python threads?  (rover not needed, motors untouched)

    python -m phase3_onboard.tools.gil_probe [--seconds 15] [--proc]

A probe thread asks to wake up every 2 ms and records how late it actually wakes.
Phase A: nothing else running (baseline).
Phase B: the real LidarThread running in this process (G4 spinning).
Phase C (--proc): the LiDAR running in its own process via lidar_proc.LidarProcess (the fix).
If B shows stalls of ~100 ms (about one scan period), the YDLidar SDK call holds Python's
global interpreter lock while it waits for a revolution.  C should then look like A.
"""
import argparse
import time

import numpy as np


def probe(seconds, label):
    late = []
    stop = time.monotonic() + seconds
    while time.monotonic() < stop:
        t0 = time.perf_counter()
        time.sleep(0.002)
        late.append((time.perf_counter() - t0 - 0.002) * 1e3)
    a = np.array(late)
    p = np.percentile(a, [50, 99, 99.9])
    print(f"{label:28} wake-up lateness ms: p50 {p[0]:6.2f}  p99 {p[1]:6.1f}  p99.9 {p[2]:6.1f}  max {a.max():6.1f}"
          f"   stalls >20 ms: {(a > 20).sum():4d}  >50 ms: {(a > 50).sum():4d}", flush=True)
    return a


def run_phase(lt, seconds, label):
    lt.start()
    time.sleep(3.0)                        # motor spin-up / first scans
    c0 = lt.count
    t_last = None
    gaps = []
    # consume scans like the SLAM thread does, in a side thread, to check delivery timing
    import threading
    stop = [False]

    def consumer():
        nonlocal t_last
        while not stop[0]:
            s = lt.get(timeout=0.5)
            if s is None:
                continue
            now = time.monotonic()
            if t_last is not None:
                gaps.append((now - t_last) * 1e3)
            t_last = now
    th = threading.Thread(target=consumer, daemon=True)
    th.start()
    b = probe(seconds, label)
    stop[0] = True
    scans = lt.count - c0
    lt.stop()
    th.join(1.0)
    msg = f"scans during {label[0]}: {scans} ({scans / seconds:.1f} Hz)"
    if gaps:
        g = np.array(gaps)
        msg += f"   scan delivery gap ms p50 {np.median(g):.0f} p99 {np.percentile(g, 99):.0f}"
    if getattr(lt, "error", None):
        msg += f"   lidar error: {lt.error}"
    print(msg, flush=True)
    return b, scans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--proc", action="store_true", help="also test the LiDAR-in-its-own-process fix")
    ap.add_argument("--only-proc", action="store_true", help="skip phase B")
    a = ap.parse_args()
    from phase3_onboard.slam.params import from_config
    P = from_config()
    lp = None
    if a.proc or a.only_proc:
        from phase3_onboard.lidar_proc import LidarProcess
        lp = LidarProcess(P, None)         # fork now, before any other thread exists
    probe(5.0, "A: baseline (no LiDAR)")
    if not a.only_proc:
        from phase3_onboard.lidar import LidarThread, make_lidar
        b, scans = run_phase(LidarThread(make_lidar(P, None), None), a.seconds, "B: LiDAR thread running")
        big = (b > 50).sum()
        if scans and big > 0.3 * scans:
            print("=> B CONFIRMED: the in-process LiDAR read blocks all Python threads about once per scan.")
        elif big:
            print("=> B: some long stalls, but fewer than one per scan.")
        else:
            print("=> B: no long stalls from the in-process driver.")
        time.sleep(1.0)
    if lp is not None:
        c, scans = run_phase(lp, a.seconds, "C: LiDAR in own process")
        big = (c > 20).sum()
        if scans == 0:
            print("=> C FAILED: no scans arrived from the LiDAR process (see error above).")
        elif big == 0:
            print("=> C OK: no stalls, scans delivered. The fix works.")
        else:
            print(f"=> C: {big} stalls >20 ms remain; paste this output.")


if __name__ == "__main__":
    main()

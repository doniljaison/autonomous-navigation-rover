"""
Fit duty -> speed / turn rate from a run (optional prior improvement).
======================================================================
    python -m phase3_onboard.tools.calibrate_motion RUN_DIR   (run replay v3 first)

Uses accepted v3 poses (replay_v3/frames.jsonl) and the commands in the log.
Prints DUTY_TO_MPS and DUTY_TO_RADPS suggestions.  Expect these to drift with
battery voltage and floor — that is why the default prior is constant velocity.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from phase3_onboard.common import relative
from phase3_onboard.logger import RunReader


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    a = ap.parse_args()
    fr = [json.loads(l) for l in open(os.path.join(a.run, "replay_v3", "frames.jsonl"))]
    fr = [f for f in fr if f["acc"]]
    cmds = []
    for e in RunReader(a.run).events():
        if e["k"] == "tx" and '"T":1,' in e["line"]:
            d = json.loads(e["line"])
            cmds.append((e["t"], d["L"], d["R"]))
    cmds.sort()
    ct = np.array([c[0] for c in cmds])
    v_rows, w_rows = [], []
    for f0, f1 in zip(fr, fr[1:]):
        dt = f1["t"] - f0["t"]
        if not 0.05 < dt < 0.3:
            continue
        rel = relative(np.array(f0["pose"]), np.array(f1["pose"]))
        i0, i1 = np.searchsorted(ct, f0["t"] - 0.15), np.searchsorted(ct, f1["t"] - 0.15)
        if i1 <= i0:
            continue
        L = np.mean([cmds[i][1] for i in range(i0, i1)])
        R = np.mean([cmds[i][2] for i in range(i0, i1)])
        mean, half = 0.5 * (L + R), 0.5 * (R - L)
        if abs(mean) > 0.08 and abs(half) < 0.05:
            v_rows.append((mean, rel[0] / dt))
        if abs(half) > 0.08 and abs(mean) < 0.05:
            w_rows.append((half, rel[2] / dt))
    if v_rows:
        x, y = np.array(v_rows).T
        k = float(x @ y / (x @ x))
        print(f"DUTY_TO_MPS   = {k:.3f}   (n={len(x)}, resid {np.std(y - k * x):.3f} m/s)")
    if w_rows:
        x, y = np.array(w_rows).T
        k = float(x @ y / (x @ x))
        print(f"DUTY_TO_RADPS = {k:.3f}   (n={len(x)}, resid {math.degrees(np.std(y - k * x)):.1f} deg/s)")
    if not v_rows and not w_rows:
        print("not enough straight / spin segments in this run")


if __name__ == "__main__":
    main()

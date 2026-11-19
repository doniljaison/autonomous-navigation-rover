"""
Gate G2 (Day 7): SLAM offline on the square-room logs.
======================================================
    python -m phase3_onboard.tools.gate_g2 RUN_DIR [RUN_DIR ...] --room W H
           [--loop-runs NAME_SUBSTR ...] [--report docs/g2_report.md] [--set K=V ...]

For each run: replay the legacy pipeline (the "before") and v3, then check v3
against the thresholds fixed BEFORE looking at results (PHASE3_PLAN section 5):
    wall-to-wall angle within +/-2 deg of 90
    room dimensions within +/-5 % of tape
    no duplicated (rotated / doubled) walls
    loop closure <= 0.15 m and <= 3 deg  (only for runs that return to the taped
    start: square loops; pass --loop-runs to name them, default "square")
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from phase3_onboard.tools.replay import _parse_sets, replay_legacy, replay_v3

THRESH = {"corner_deg": 2.0, "dim_pct": 5.0, "dup": 0, "loop_m": 0.15, "loop_deg": 3.0}


def evaluate(summ, room, is_loop):
    m = summ["map"]
    checks = {}
    c = m.get("corner_angle_deg")
    checks["corner"] = (c is not None and abs(c - 90.0) <= THRESH["corner_deg"], c)
    if room:
        d = m.get("room_dim_err_pct")
        checks["dims"] = (d is not None and d <= THRESH["dim_pct"], d)
    checks["dup"] = (m.get("duplicated_walls", 99) <= THRESH["dup"], m.get("duplicated_walls"))
    if is_loop and summ.get("truth"):
        # simulator: the true end pose is known, so judge the estimate against it
        tr = summ["truth"]
        checks["loop"] = (tr["final_xy_err_m"] <= THRESH["loop_m"] and tr["final_yaw_err_deg"] <= THRESH["loop_deg"],
                          f'{tr["final_xy_err_m"]} m / {tr["final_yaw_err_deg"]} deg (vs sim truth)')
    elif is_loop and summ.get("loop"):
        lp = summ["loop"]
        checks["loop"] = (lp["loop_xy_m"] <= THRESH["loop_m"] and lp["loop_yaw_deg"] <= THRESH["loop_deg"],
                          f'{lp["loop_xy_m"]} m / {lp["loop_yaw_deg"]} deg')
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--room", nargs=2, type=float, default=None)
    ap.add_argument("--loop-runs", nargs="*", default=["square"])
    ap.add_argument("--report", default=None)
    ap.add_argument("--set", action="append", dest="sets")
    a = ap.parse_args()
    ov = _parse_sets(a.sets)
    rows = []
    all_ok = True
    for run in a.runs:
        name = os.path.basename(os.path.normpath(run))
        is_loop = any(s in name for s in a.loop_runs)
        before = replay_legacy(run, ov, room=a.room, quiet=True)
        after = replay_v3(run, ov, room=a.room, quiet=True)
        cb, ca = evaluate(before, a.room, is_loop), evaluate(after, a.room, is_loop)
        ok = all(v[0] for v in ca.values())
        legacy_fails = not all(v[0] for v in cb.values())
        all_ok &= ok
        rows.append((name, cb, ca, ok, legacy_fails, before, after))

    lines = ["# Gate G2 — offline SLAM on recorded logs", "",
             f"Thresholds (fixed in advance): corner 90±{THRESH['corner_deg']}°, dims ±{THRESH['dim_pct']}%, "
             f"duplicated walls = 0, loop ≤ {THRESH['loop_m']} m / {THRESH['loop_deg']}°.",
             f"Room (tape): {a.room}", "",
             "| run | check | legacy (before) | v3 (after) | v3 pass |", "|---|---|---|---|---|"]
    for name, cb, ca, ok, lf, b, af in rows:
        for k in ca:
            lines.append(f"| {name} | {k} | {cb[k][1]} | {ca[k][1]} | {'✅' if ca[k][0] else '❌'} |")
        rr = af.get("reasons", {})
        lines.append(f"| {name} | v3 gate | — | accepted {af.get('accepted')} / rejected {af.get('rejected')} "
                     f"{json.dumps(rr)} | |")
        lines.append(f"| {name} | step ms p95 | {b['step_ms'] and b['step_ms']['p95']} | "
                     f"{af['step_ms'] and af['step_ms']['p95']} | |")
        if "truth" in af:
            lines.append(f"| {name} | ATE (sim truth) | {b['truth']['ate_rmse_m']} m, yaw max "
                         f"{b['truth']['max_yaw_err_deg']}° | {af['truth']['ate_rmse_m']} m, yaw max "
                         f"{af['truth']['max_yaw_err_deg']}° | |")
        if not lf:
            lines.append(f"| {name} | note | legacy did NOT fail visibly on this log | | |")
    lines += ["", f"**G2 {'PASS' if all_ok else 'FAIL'}**", "",
              "Maps: `<run>/replay_legacy/map.png` vs `<run>/replay_v3/map.png`."]
    txt = "\n".join(lines)
    print(txt)
    if a.report:
        os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
        with open(a.report, "w") as f:
            f.write(txt + "\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

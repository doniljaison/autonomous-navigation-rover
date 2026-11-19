"""
Replay a recorded run through a SLAM pipeline at full speed.
============================================================
    python -m phase3_onboard.tools.replay runs/<run> [--pipeline v3|legacy|both]
           [--set NAME=VALUE ...] [--room W H] [--out DIR]

Writes <run>/replay_<pipeline>/ : map.npz, map.png, traj.csv, frames.jsonl,
summary.json.  `--set` overrides any config value for tuning, e.g.
    --set GATE_YAW_TOL_DEG=1.5 --set MAP_RES_M=0.075
Works on real logs and simulator logs alike (sim logs add ground-truth errors).
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import time

import numpy as np

from phase3_onboard.common import relative, wrap
from phase3_onboard.imgutil import map_rgb, png_bytes
from phase3_onboard.logger import RunReader
from phase3_onboard.slam.params import from_config
from phase3_onboard.tools.metrics import loop_error, map_metrics, percentiles


def _parse_sets(sets):
    out = {}
    for s in sets or []:
        k, v = s.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def truth_errors(truth, est):
    """truth: list of (t, pose); est: list of (t, x, y, th) in the map frame
    (= robot frame at the first estimated pose).  Returns ATE stats."""
    if not truth or not est:
        return None
    tt = np.array([t for t, _ in truth])
    tp = np.array([p for _, p in truth])
    e = np.asarray(est)
    t0 = e[0, 0]
    i0 = int(np.clip(np.searchsorted(tt, t0), 0, len(tt) - 1))
    origin = tp[i0]
    errs_xy, errs_th = [], []
    for t, x, y, th in e:
        i = int(np.clip(np.searchsorted(tt, t), 0, len(tt) - 1))
        g = relative(origin, tp[i])
        errs_xy.append(math.hypot(x - g[0], y - g[1]))
        errs_th.append(abs(math.degrees(wrap(th - g[2]))))
    gl = relative(origin, tp[-1])
    final = (math.hypot(e[-1, 1] - gl[0], e[-1, 2] - gl[1]), abs(math.degrees(wrap(e[-1, 3] - gl[2]))))
    return {"ate_rmse_m": round(float(np.sqrt(np.mean(np.square(errs_xy)))), 4),
            "max_xy_err_m": round(float(np.max(errs_xy)), 4),
            "yaw_err_rms_deg": round(float(np.sqrt(np.mean(np.square(errs_th)))), 3),
            "max_yaw_err_deg": round(float(np.max(errs_th)), 3),
            "final_xy_err_m": round(final[0], 4), "final_yaw_err_deg": round(final[1], 3),
            "true_loop_xy_m": round(math.hypot(gl[0], gl[1]), 3)}


def replay_v3(run_dir, overrides=None, out=None, room=None, quiet=False):
    from phase3_onboard.slam.engine import SlamEngine, SlamInputs, warmup_jit
    P = from_config(**(overrides or {}))
    warmup_jit(P)
    rd = RunReader(run_dir)
    eng = SlamEngine(P)
    inp = SlamInputs(P, eng)
    truth, frames, est = [], [], []
    t_wall = time.perf_counter()
    for kind, x in rd.merged():
        if kind == "ev":
            k = x["k"]
            if k == "rx":
                inp.on_rx(x["t"], x["line"])
            elif k == "tx":
                inp.on_tx(x["t"], x["line"])
            elif k == "truth":
                truth.append((x["t"], np.array(x["pose"])))
        else:
            r = eng.step(x, cmd_idle=inp.cmd_idle(x.t_end), cmd=inp.last_cmd)
            if r is None:
                continue
            d = r.as_log()
            d["t"] = round(r.t, 4)
            frames.append(d)
            est.append((r.t, *r.pose))
    wall = time.perf_counter() - t_wall
    return _finish(run_dir, "v3", eng.grid, est, frames, truth, wall, out, room, quiet,
                   extra={"accepted": eng.accepted, "rejected": eng.rejected, "skipped": eng.skipped,
                          "lost_at_end": eng.lost, "imu_samples": inp.n_imu,
                          "gyro_bias_dps": round(math.degrees(eng.gyro.bias), 3)})


def replay_legacy(run_dir, overrides=None, out=None, room=None, quiet=False):
    from phase3_onboard.imu import ImuParser
    from phase3_onboard.slam.legacy import LegacyEngine
    P = from_config(**(overrides or {}))
    rd = RunReader(run_dir)
    eng = LegacyEngine(P)
    parser = ImuParser(P)
    truth, frames, est = [], [], []
    t_wall = time.perf_counter()
    for kind, x in rd.merged():
        if kind == "ev":
            if x["k"] == "rx":
                msg = parser.parse_line(x["line"])
                s = parser.sample(x["t"], msg) if msg else None
                if s is not None:
                    eng.feed_yaw(x["t"], s.yaw)
            elif x["k"] == "truth":
                truth.append((x["t"], np.array(x["pose"])))
        else:
            t1 = time.perf_counter()
            p = eng.step(x)
            dt = (time.perf_counter() - t1) * 1e3
            if p is None:
                continue
            est.append((x.t_end, *p))
            _, dth, err, n = eng.icp_log[-1]
            frames.append({"t": round(x.t_end, 4), "pose": [round(float(v), 5) for v in p],
                           "mdth": dth, "err": None if not np.isfinite(err) else round(float(err), 4),
                           "npts": n, "ms": {"total": round(dt, 2)}})
    wall = time.perf_counter() - t_wall
    return _finish(run_dir, "legacy", eng.grid, est, frames, truth, wall, out, room, quiet)


def _finish(run_dir, name, grid, est, frames, truth, wall, out, room, quiet, extra=None):
    out = out or os.path.join(run_dir, f"replay_{name}")
    os.makedirs(out, exist_ok=True)
    grid.save(os.path.join(out, "map.npz"))
    e = np.asarray(est) if est else np.zeros((0, 4))
    cells = grid.w2c(e[:, 1], e[:, 2]) if len(e) else None
    with open(os.path.join(out, "map.png"), "wb") as f:
        f.write(png_bytes(map_rgb(grid.log, path_cells=cells)))
    np.savetxt(os.path.join(out, "traj.csv"), e, delimiter=",", header="t,x,y,theta", comments="", fmt="%.5f")
    with open(os.path.join(out, "frames.jsonl"), "w") as f:
        for d in frames:
            f.write(json.dumps(d) + "\n")
    summ = {"pipeline": name, "run": os.path.basename(os.path.normpath(run_dir)), "frames": len(frames),
            "replay_wall_s": round(wall, 2)}
    if extra:
        summ.update(extra)
    summ["step_ms"] = percentiles([f["ms"].get("total") for f in frames if f.get("ms")])
    if name == "v3":
        for st in ("pre", "field", "match", "map"):
            summ[f"{st}_ms"] = percentiles([f["ms"].get(st) for f in frames if f["ms"].get(st) is not None])
        from collections import Counter
        summ["reasons"] = dict(Counter(f["why"] for f in frames))
    summ["loop"] = loop_error(e)
    mm = map_metrics(grid.log, grid.res, grid.orig, room)
    mm.pop("walls", None)
    summ["map"] = mm
    te = truth_errors(truth, est)
    if te:
        summ["truth"] = te
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    if not quiet:
        print(json.dumps(summ, indent=1))
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--pipeline", default="both", choices=["v3", "legacy", "both"])
    ap.add_argument("--set", action="append", dest="sets", help="NAME=VALUE config override")
    ap.add_argument("--room", nargs=2, type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ov = _parse_sets(a.sets)
    if a.pipeline in ("legacy", "both"):
        replay_legacy(a.run, ov, None if a.pipeline == "both" else a.out, a.room)
    if a.pipeline in ("v3", "both"):
        replay_v3(a.run, ov, None if a.pipeline == "both" else a.out, a.room)


if __name__ == "__main__":
    main()

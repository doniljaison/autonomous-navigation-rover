"""
Phase 0 diagnosis (Day 3): confirm or reject RC1-RC7 on a recorded log.
=======================================================================
    python -m phase3_onboard.tools.diagnose RUN_DIR [--plots]

Writes RUN_DIR/diag/diagnosis.md (+ PNG plots with --plots, needs matplotlib).

RC1  legacy ICP dtheta vs gyro dtheta per frame (large disagreements while turning)
RC3  rotation during one revolution -> smear at 2 m / 5 m
RC4  IMU: rate, gaps, gyro units/sign (vs scan-matching-only heading), firmware
     yaw vs gyro integral, lag between gyro and scan-matched yaw rate
RC5  how many legacy map updates came from frames that disagreed with the gyro
RC6  invalid / spike points per scan; legacy step time
RC7  raw points per revolution vs distinct integer degrees kept by PyLidar3-style dicts
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

from phase3_onboard.imu import GyroYaw, ImuParser
from phase3_onboard.logger import RunReader
from phase3_onboard.slam.legacy import LegacyEngine, legacy_points
from phase3_onboard.slam.params import from_config
from phase3_onboard.slam.preprocess import prepare


def _stats(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return "n/a"
    return f"median {np.median(x):.3g}, p95 {np.percentile(x, 95):.3g}, max {np.max(x):.3g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--yaw-tol-deg", type=float, default=2.0)
    a = ap.parse_args()
    P = from_config()
    rd = RunReader(a.run)
    parser = ImuParser(P)
    gyro = GyroYaw("gyro", history_s=1e9, bias_window_s=P.GYRO_BIAS_WINDOW_S)
    leg = LegacyEngine(P)

    imu_t, imu_rate, imu_yaw = [], [], []
    last_nonzero = -1e9
    n_raw, n_deg, n_zero, n_spike, n_valid = [], [], [], [], []
    scan_rot = []
    frames = []   # t, legacy dth, gyro dth, icp err, legacy ms
    prev_t = None
    import time as _t
    for kind, x in rd.merged():
        if kind == "ev":
            if x["k"] == "tx":
                msg = parser.parse_line(x["line"])
                if msg and msg.get("T") == 1 and (abs(msg.get("L", 0)) > 1e-3 or abs(msg.get("R", 0)) > 1e-3):
                    last_nonzero = x["t"]
            elif x["k"] == "rx":
                msg = parser.parse_line(x["line"])
                s = parser.sample(x["t"], msg) if msg else None
                if s is None:
                    continue
                still = x["t"] - last_nonzero > 0.5 and (s.rate is None or abs(s.rate - gyro.bias) < math.radians(3))
                gyro.add(s, stationary=still)
                leg.feed_yaw(x["t"], s.yaw)
                imu_t.append(x["t"])
                imu_rate.append(s.rate if s.rate is not None else np.nan)
                imu_yaw.append(s.yaw if s.yaw is not None else np.nan)
            continue
        sc = x
        prep = prepare(sc, P)
        n_raw.append(sc.n)
        n_deg.append(len(legacy_points(sc, P.LIDAR_ANGLE_SIGN, P.LIDAR_ANGLE_OFFSET_RAD)))
        n_zero.append(prep.n_zero)
        n_spike.append(prep.n_spikes)
        n_valid.append(prep.n_valid)
        gd = gyro.delta(prev_t, sc.t_end) if (prev_t is not None and gyro.available) else None
        if gyro.available:
            scan_rot.append(abs(gyro.delta(sc.t_start, sc.t_end)))
        t1 = _t.perf_counter()
        leg.step(sc)
        ms = (_t.perf_counter() - t1) * 1e3
        ld = leg.icp_log[-1][1] if leg.icp_log else None
        err = leg.icp_log[-1][2] if leg.icp_log else None
        frames.append((sc.t_end, ld, gd, err, ms))
        prev_t = sc.t_end

    out = os.path.join(a.run, "diag")
    os.makedirs(out, exist_ok=True)
    L = [f"# Phase 0 diagnosis — `{os.path.basename(os.path.normpath(a.run))}`", ""]
    meta = rd.meta
    L.append(f"Host `{meta.get('host')}` ({meta.get('machine')}), created {meta.get('created')}, "
             f"sim={meta.get('sim', False)}, pattern={meta.get('pattern', '-')}.")
    L.append("")

    # RC7
    L += ["## RC7 — angular resolution", "",
          f"* raw points / revolution: {_stats(n_raw)}",
          f"* distinct integer-degree keys (PyLidar3-style dict): {_stats(n_deg)}",
          f"* => legacy keeps ~{100 * np.median(n_deg) / max(1, np.median(n_raw)):.0f}% of the samples "
          f"(1° = {math.radians(1) * 2 * 100:.1f} cm at 2 m, {math.radians(1) * 5 * 100:.1f} cm at 5 m)", ""]
    # RC6
    lm = [f[4] for f in frames]
    L += ["## RC6 — invalid returns and spikes", "",
          f"* zero / no-return points per scan: {_stats(n_zero)}",
          f"* isolated spikes removed by v3 per scan: {_stats(n_spike)}",
          f"* legacy step time (ms, this machine, jitted raycast): {_stats(lm)}",
          "  (the original pure-Python Bresenham is ~10-50x slower than this)", ""]
    # IMU
    it = np.array(imu_t)
    L += ["## RC4 — IMU stream", ""]
    if len(it) > 2:
        dt = np.diff(it)
        L.append(f"* IMU samples: {len(it)} over {it[-1] - it[0]:.1f} s -> {len(it) / (it[-1] - it[0]):.1f} Hz; "
                 f"gaps: {_stats(dt * 1e3)} ms")
        rr = np.array(imu_rate)
        if np.isfinite(rr).any():
            L.append(f"* gyro z (after config scale) |rate|: {_stats(np.degrees(np.abs(rr[np.isfinite(rr)])))} deg/s; "
                     f"bias estimate {math.degrees(gyro.bias):.3f} deg/s (ready={gyro.bias_ready})")
        yy = np.array(imu_yaw)
        if np.isfinite(yy).any() and gyro.available:
            fin = np.isfinite(yy)
            yu = np.unwrap(yy[fin])
            gi = gyro.yaw_at_many(it[fin])
            drift = (yu - yu[0]) - (gi - gi[0])
            L.append(f"* firmware yaw `y` minus gyro integral over the run: final {math.degrees(drift[-1]):.2f} deg, "
                     f"max |.| {math.degrees(np.abs(drift).max()):.2f} deg  (large => do not trust `y`)")
    else:
        L.append("* **no IMU samples parsed** — check IMU_* keys in config (tools/imu_probe.py)")
    L.append("")
    # RC1 / RC5
    ok = [(t, ld, gd) for t, ld, gd, _, _ in frames if ld is not None and gd is not None]
    L += ["## RC1 / RC5 — legacy ICP vs gyro per frame", ""]
    if ok:
        d = np.array([math.degrees((ld - gd + math.pi) % (2 * math.pi) - math.pi) for _, ld, gd in ok])
        g = np.array([abs(math.degrees(gd)) for _, _, gd in ok])
        bad = np.abs(d) > a.yaw_tol_deg
        turning = g > 2.0
        L.append(f"* frames compared: {len(d)}; |ICP dθ − gyro dθ|: {_stats(np.abs(d))} deg")
        L.append(f"* frames with |diff| > {a.yaw_tol_deg}°: {bad.sum()} ({100 * bad.mean():.1f}%) — "
                 f"while turning (>2°/frame): {(bad & turning).sum()} of {turning.sum()}, "
                 f"straight/still: {(bad & ~turning).sum()} of {(~turning).sum()}")
        L.append(f"* RC5: every one of those {bad.sum()} frames was still written into the legacy map "
                 "(no gate), and the 30% pull toward firmware yaw cannot undo a discrete snap.")
        L.append(f"* verdict RC1: {'CONFIRMED' if bad.sum() > 0.02 * len(d) and (bad & turning).sum() > (bad & ~turning).sum() else 'not confirmed on this log'}")
    else:
        L.append("* not available (no gyro samples or too few frames)")
    L.append("")
    # RC3
    if scan_rot:
        r = np.degrees(np.array(scan_rot))
        L += ["## RC3 — scan skew", "",
              f"* rotation during one revolution: {_stats(r)} deg",
              f"* => worst smear at 2 m {math.radians(r.max()) * 2 * 100:.1f} cm, at 5 m "
              f"{math.radians(r.max()) * 5 * 100:.1f} cm (v3 de-skews with the gyro)", ""]
    with open(os.path.join(out, "diagnosis.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

    if a.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed: skipping plots")
            return
        if ok:
            t0 = frames[0][0]
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax[0].plot([t - t0 for t, _, _ in ok], [math.degrees(gd) for _, _, gd in ok], label="gyro dθ")
            ax[0].plot([t - t0 for t, _, _ in ok], [math.degrees(ld) for _, ld, _ in ok], ".", ms=3, label="legacy ICP dθ")
            ax[0].set_ylabel("deg / frame"); ax[0].legend()
            ax[1].plot([t - t0 for t, _, _ in ok], d, ".", ms=3, color="crimson")
            ax[1].axhline(a.yaw_tol_deg, ls="--", c="k", lw=.7); ax[1].axhline(-a.yaw_tol_deg, ls="--", c="k", lw=.7)
            ax[1].set_ylabel("ICP − gyro (deg)"); ax[1].set_xlabel("s")
            fig.tight_layout(); fig.savefig(os.path.join(out, "rc1_icp_vs_gyro.png"), dpi=110)
        print(f"plots in {out}")


if __name__ == "__main__":
    main()

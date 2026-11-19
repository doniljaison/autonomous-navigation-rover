"""
Generate the synthetic log set (same patterns as the Day-2 real recordings).
===========================================================================
    python -m phase3_onboard.tools.sim_logs --out runs_sim [--stress] [--seed N]

Standard set : handrot, spin, slow_square, fast_square, wander
--stress     : harder variants — empty (perfectly symmetric) room, 2 rad/s
               spins, 3 % gyro scale error, larger gyro bias + noise, 2x spikes,
               LiDAR mounted 6 cm forward (replay with --set LIDAR_MOUNT_X_M=0.06),
               IMU at 15 Hz.
"""

from __future__ import annotations

import argparse

from phase3_onboard.sim import SimParams, generate_run


def stress_params(seed):
    return SimParams(seed=seed, clutter=False, k_w=5.6, gyro_scale_err=0.03, gyro_bias_dps=1.0,
                     gyro_noise_dps=0.6, spike_rate=0.012, dropout=0.08, imu_stream_hz=15.0,
                     lidar_mount=(0.06, 0.0, 0.0), slip_noise=0.15)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs_sim")
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--patterns", nargs="*", default=["handrot", "spin", "slow_square", "fast_square", "wander"])
    a = ap.parse_args()
    for pat in a.patterns:
        p = stress_params(a.seed) if a.stress else SimParams(seed=a.seed)
        tag = f"sim{'_stress' if a.stress else ''}_{pat}"
        d = generate_run(pat, a.out, p, tag=tag)
        print(d)


if __name__ == "__main__":
    main()

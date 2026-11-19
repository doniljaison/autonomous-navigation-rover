"""
Gate G4 rehearsal in the simulator (no hardware).
=================================================
    python -m phase3_onboard.tools.sim_e2e

1. map the room with a short scripted manual drive (drive intents, like the browser)
2. 5 click-to-goal targets; success = TRUE position within 0.30 m of the goal
3. "browser dies" while driving: intents stop -> rover must be commanded to 0
   within INTENT_TTL_S + 2 command periods, and actually stop
4. save the map, restart on it with a start pose that is 0.15 m / 6 deg off,
   and check re-localisation on the first scans
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time

import numpy as np

from phase3_onboard.common import relative
from phase3_onboard.main import Runtime
from phase3_onboard.sim import SimParams
import phase3_onboard.sim as simmod
from phase3_onboard.slam.params import from_config


def truth_in_map(rt):
    return relative(np.array(rt.live_sim.core.p.start_pose), rt.live_sim.truth_pose)


def wait(cond, timeout):
    t = time.monotonic()
    while time.monotonic() - t < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


def main():
    tmp = tempfile.mkdtemp(prefix="sim_e2e_")
    P = from_config(LINK_TRANSPORT="sim", LIDAR_DRIVER="sim", LOG_DIR=os.path.join(tmp, "runs"),
                    MAPS_DIR=os.path.join(tmp, "maps"))
    simmod._LIVE = None
    rt = Runtime(P, tag="e2e", sim_params=SimParams(seed=7))
    rt.start(web=False)
    ok_all = True
    try:
        time.sleep(3.0)   # standstill: gyro bias + bootstrap
        print("1) mapping drive")
        for fwd, turn, dur in [(0.6, 0, 2.0), (0, 0.8, 1.6), (0.6, 0, 2.0), (0, -0.8, 1.2), (0.5, 0, 1.5)]:
            t = time.monotonic()
            while time.monotonic() - t < dur:
                rt.drive(fwd, turn)
                time.sleep(0.1)
        time.sleep(1.0)
        goals = [(1.4, 0.4), (0.3, 1.8), (2.4, 2.2), (0.2, 0.2), (2.6, 0.3)]
        print("2) click-to-goal")
        reached = 0
        for gx, gy in goals:
            rt.set_goal(gx, gy)
            done = wait(lambda: rt.mode == "MANUAL", 40)
            tp = truth_in_map(rt)
            err = math.hypot(tp[0] - gx, tp[1] - gy)
            good = done and err <= 0.30
            reached += good
            est = rt.eng.pose
            print(f"   goal ({gx:.1f},{gy:.1f}) -> true ({tp[0]:.2f},{tp[1]:.2f}) err {err:.2f} m "
                  f"[est err vs truth {math.hypot(est[0]-tp[0], est[1]-tp[1]):.3f} m] {'OK' if good else 'FAIL'}")
            time.sleep(0.5)
        print(f"   {reached}/5 goals within 0.30 m (true position)")
        ok_all &= reached == 5

        print("3) browser dies while driving")
        # spin in place (proximity limiting never blocks turning, so this always drives)
        for _ in range(10):
            rt.drive(0.0, 1.0)
            time.sleep(0.1)
        was_driving = rt.link.last_sent != (0.0, 0.0)
        t_last = rt.manual[2]                   # time of the last browser message
        stopped = wait(lambda: rt.link.last_sent == (0.0, 0.0), 2.0)
        dt_cmd = time.monotonic() - t_last
        lim = P.INTENT_TTL_S + 2.0 / P.CMD_HZ
        halted = wait(lambda: abs(rt.live_sim.core.w) < 0.02 and abs(rt.live_sim.core.v) < 0.01, 2.0)
        print(f"   driving before drop: {was_driving}; commanded 0 {dt_cmd:.2f} s after the last "
              f"message (limit {lim:.2f} s); rover stopped: {halted}")
        ok_all &= was_driving and stopped and dt_cmd <= lim and halted

        print("4) save map, reload with an offset start pose")
        path = rt.save_map()
        true_now = truth_in_map(rt)
    finally:
        rt.shutdown()
        simmod._LIVE = None

    # restart the simulator at the rover's current true pose; give the map a wrong guess
    sp = SimParams(seed=8)
    start_world = rt.live_sim.core.pose.copy()
    sp.start_pose = tuple(start_world)
    guess = (true_now[0] + 0.15, true_now[1] - 0.05, true_now[2] + math.radians(6))
    rt2 = Runtime(P, map_path=path, init_pose=guess, tag="e2e_reload", sim_params=sp)
    rt2.start(web=False)
    try:
        ok = wait(lambda: rt2.eng.accepted >= 3, 10)
        est = rt2.eng.pose
        err = math.hypot(est[0] - true_now[0], est[1] - true_now[1])
        yerr = abs(math.degrees((est[2] - true_now[2] + math.pi) % (2 * math.pi) - math.pi))
        print(f"   re-localised: {ok}, error vs truth {err:.3f} m / {yerr:.2f} deg "
              f"(start guess was 0.16 m / 6 deg off)")
        ok_all &= ok and err < 0.05 and yerr < 1.0
    finally:
        rt2.shutdown()
    print("\nG4 rehearsal (sim):", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

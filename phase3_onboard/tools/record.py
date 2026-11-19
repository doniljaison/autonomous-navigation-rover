"""
Headless recorder (Day 2 on the PC rig, Day 8 on the Pi).
=========================================================
    python -m phase3_onboard.tools.record --tag handrot --seconds 25
    python -m phase3_onboard.tools.record --tag spin --pattern spin --duty 0.35
    python -m phase3_onboard.tools.record --link http --set ROVER_HTTP_HOST='"10.0.0.5"' \
        --set LIDAR_PORT='"COM5"' --set IMU_POLL_HZ=15 --tag pc_spin --pattern spin

Logs raw scans + every rover line + every command (same format as main.py).
Square loops are driven by hand through the web UI instead:
    python -m phase3_onboard.main --tag slow_square          (on the Pi)
Patterns (open loop, no encoders): none | spin
    spin: still 3 s, +turn for --spin-s, still 2 s, -turn for --spin-s, still 3 s
"""

from __future__ import annotations

import argparse
import time

from phase3_onboard.lidar import LidarThread, make_lidar
from phase3_onboard.logger import RunLogger
from phase3_onboard.rover_link import RoverLink, make_transport
from phase3_onboard.slam.params import from_config
from phase3_onboard.tools.replay import _parse_sets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pattern", choices=["none", "spin"], default="none")
    ap.add_argument("--seconds", type=float, default=60.0, help="duration for pattern=none")
    ap.add_argument("--duty", type=float, default=0.35)
    ap.add_argument("--spin-s", type=float, default=8.0)
    ap.add_argument("--link", choices=["uart", "http", "sim"], default=None)
    ap.add_argument("--set", action="append", dest="sets")
    a = ap.parse_args()
    ov = _parse_sets(a.sets)
    if a.link:
        ov["LINK_TRANSPORT"] = a.link
    P = from_config(**ov)
    lg = RunLogger(P.LOG_DIR, a.tag, meta={"tool": "record", "pattern": a.pattern, "duty": a.duty})
    link = RoverLink(make_transport(P), P, lg)
    lid = LidarThread(make_lidar(P), lg)
    link.start()
    lid.start()
    print(f"recording to {lg.dir}  (Ctrl-C stops and leaves the rover stopped)")

    def hold(L, R, dur):
        t = time.monotonic()
        while time.monotonic() - t < dur:
            link.set_intent(L, R, "record", ttl=0.3)
            time.sleep(0.05)
            print(f"\r  scans {lid.count}  {lid.rate_hz:.1f} Hz  rx {link.rx_count}  cmd {link.last_sent}   ",
                  end="", flush=True)
    try:
        if a.pattern == "spin":
            hold(0, 0, 3)
            lg.event("mode", msg="spin +")
            hold(-a.duty, a.duty, a.spin_s)
            hold(0, 0, 2)
            lg.event("mode", msg="spin -")
            hold(a.duty, -a.duty, a.spin_s)
            hold(0, 0, 3)
        else:
            hold(0, 0, a.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        link.stop()
        lid.stop()
        lg.close()
        print(f"\nsaved {lg.dir}: {lid.count} scans, dropped-for-SLAM n/a (all logged)")


if __name__ == "__main__":
    main()

"""
Request -> reply latency of the rover link (paper item, deferred but cheap).
===========================================================================
    python -m phase3_onboard.tools.link_latency --link uart -n 500
    python -m phase3_onboard.tools.link_latency --link http --host 192.168.x.y -n 200

Sends {"T":126} one at a time and times the matching IMU reply.  The motor
stream keeps running at L=R=0 so the measurement includes normal link load.
"""

from __future__ import annotations

import argparse
import json
import threading
import time

import numpy as np

from phase3_onboard.rover_link import RoverLink, make_transport
from phase3_onboard.slam.params import from_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--link", choices=["uart", "http", "sim"], required=True)
    ap.add_argument("--host", default=None)
    ap.add_argument("-n", type=int, default=300)
    a = ap.parse_args()
    ov = {"LINK_TRANSPORT": a.link, "IMU_POLL_HZ": 0, "FEEDBACK_STREAM": False}
    if a.host:
        ov["ROVER_HTTP_HOST"] = a.host
    P = from_config(**ov)
    ev = threading.Event()
    got = {}

    def on(t, line):
        try:
            if json.loads(line).get("T") in P.IMU_REPLY_TYPES:
                got["t"] = t
                ev.set()
        except Exception:
            pass
    link = RoverLink(make_transport(P), P, None)
    link.subscribe(on)
    link.start()
    lat, lost = [], 0
    try:
        for _ in range(a.n):
            ev.clear()
            t0 = time.monotonic()
            link.send_raw({"T": 126})
            if ev.wait(1.0):
                lat.append((got["t"] - t0) * 1e3)
            else:
                lost += 1
            time.sleep(0.02)
    finally:
        link.stop()
    if lat:
        x = np.array(lat)
        print(f"{a.link}: n={len(x)} lost={lost}  p50 {np.percentile(x, 50):.1f} ms  "
              f"p95 {np.percentile(x, 95):.1f} ms  p99 {np.percentile(x, 99):.1f} ms  max {x.max():.1f} ms")
    else:
        print("no replies")


if __name__ == "__main__":
    main()

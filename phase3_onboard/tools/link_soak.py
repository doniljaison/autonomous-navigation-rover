"""
Rover link soak + watchdog + direction tests (Day 4 / Gate G1).
===============================================================
    python -m phase3_onboard.tools.link_soak --minutes 10          # L=R=0 stream + IMU poll
    python -m phase3_onboard.tools.link_soak --motor-test          # WHEELS OFF THE GROUND
    python -m phase3_onboard.tools.link_soak --watchdog-test       # WHEELS OFF THE GROUND

Soak PASS: IMU replies arrive at >= 80 % of IMU_POLL_HZ with no gap > 0.5 s,
no tx errors, for the whole period.

--motor-test  : forward 1 s, then +turn (should spin CCW / LEFT seen from
                above) 1 s.  Answer the prompts; the tool tells you whether to
                set SWAP_LR / INVERT_FWD in config_local.py.
--watchdog-test: sends a drive intent once and stops refreshing it; checks the
                stream falls back to L=R=0 within INTENT_TTL_S + one period.
                Then: kill -9 test — run `python -m phase3_onboard.main`, drive,
                `kill -9` the process; the firmware's 3 s timeout must stop it.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

from phase3_onboard.logger import RunLogger
from phase3_onboard.rover_link import RoverLink, make_transport
from phase3_onboard.slam.params import from_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--link", choices=["uart", "http", "sim"], default=None)
    ap.add_argument("--motor-test", action="store_true")
    ap.add_argument("--watchdog-test", action="store_true")
    ap.add_argument("--duty", type=float, default=0.18)
    a = ap.parse_args()
    ov = {"LINK_TRANSPORT": a.link} if a.link else {}
    P = from_config(**ov)
    lg = RunLogger(P.LOG_DIR, "link_soak", meta={"tool": "link_soak"})
    rx = defaultdict(list)

    def on(t, line):
        try:
            rx[json.loads(line).get("T")].append(t)
        except Exception:
            rx["bad"].append(t)
    link = RoverLink(make_transport(P), P, lg)
    link.subscribe(on)
    link.start()
    try:
        if a.motor_test:
            motor_test(link, a.duty)
        elif a.watchdog_test:
            watchdog_test(P, link, a.duty)
        else:
            soak(P, link, rx, a.minutes)
    finally:
        link.stop()
        lg.close()
        print("log:", lg.dir)


def soak(P, link, rx, minutes):
    t0 = time.monotonic()
    t_end = t0 + minutes * 60
    print(f"soaking {minutes:.1f} min on {P.LINK_TRANSPORT} (motors stay at 0) ...")
    try:
        while time.monotonic() < t_end:
            time.sleep(5)
            el = time.monotonic() - t0
            print(f"  {el:6.0f} s  tx {link.tx_count}  rx {link.rx_count}  types "
                  f"{ {k: len(v) for k, v in rx.items()} }  tx_err {link.tx_errors}")
    except KeyboardInterrupt:
        pass
    dur = time.monotonic() - t0
    print("\n=== link soak summary ===")
    ok = link.tx_errors == 0
    for T, ts in rx.items():
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        mg = max(gaps) if gaps else float("inf")
        print(f"T={T}: {len(ts)} msgs, {len(ts) / dur:.1f} Hz, max gap {mg:.3f} s")
    imu = [ts for T, ts in rx.items() if T in P.IMU_REPLY_TYPES]
    if P.IMU_POLL_HZ > 0:
        n = max((len(ts) for ts in imu), default=0)
        gaps = max((max([b - a for a, b in zip(ts, ts[1:])] or [9]) for ts in imu), default=9)
        ok &= n / dur >= 0.8 * P.IMU_POLL_HZ and gaps < 0.5
    print(f"motor commands sent: {link.tx_count} ({link.tx_count / dur:.1f}/s incl. polls)")
    print("G1 link part:", "PASS" if ok else "FAIL")


def motor_test(link, duty):
    input("WHEELS OFF THE GROUND (rover on a box). Press Enter to drive FORWARD for 1 s ...")
    t = time.monotonic()
    while time.monotonic() - t < 1.0:
        link.set_intent(duty, duty, "test")
        time.sleep(0.05)
    link.stop_now()
    fwd = input("Did all wheels turn FORWARD? [y/n] ").strip().lower() == "y"
    input("Press Enter to command +turn (left side back, right side forward) for 1 s ...")
    t = time.monotonic()
    while time.monotonic() - t < 1.0:
        link.set_intent(-duty, duty, "test")
        time.sleep(0.05)
    link.stop_now()
    ccw = input("Did the LEFT wheels turn backward and the RIGHT wheels forward? [y/n] ").strip().lower() == "y"
    print("\n=== result ===")
    if not fwd:
        print("set INVERT_FWD = True in config_local.py (and re-run this test)")
    if fwd and not ccw:
        print("set SWAP_LR = True in config_local.py (and re-run this test)")
    if fwd and ccw:
        print("directions OK: +L/+R forward, R>L turns CCW (left). Record in docs/conventions.md")


def watchdog_test(P, link, duty):
    input("WHEELS OFF THE GROUND. Enter = send ONE intent and stop refreshing it ...")
    link.set_intent(duty, duty, "watchdog-test")
    t0 = time.monotonic()
    stopped_at = None
    seen_drive = False
    while time.monotonic() - t0 < 3.0:
        nonzero = link.last_sent != (0.0, 0.0)
        seen_drive |= nonzero
        if seen_drive and not nonzero:
            stopped_at = time.monotonic() - t0
            break
        time.sleep(0.002)
    lim = P.INTENT_TTL_S + 2.0 / P.CMD_HZ
    print(f"stream fell back to L=R=0 after {stopped_at:.3f} s (limit {lim:.3f} s)" if stopped_at
          else "stream did NOT fall back to zero!")
    print("watchdog:", "PASS" if stopped_at is not None and stopped_at <= lim else "FAIL")


if __name__ == "__main__":
    main()

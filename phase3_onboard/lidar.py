"""
LiDAR sources -> timestamped full-resolution `Scan`s.
=====================================================
YDLidarSource  : official YDLidar-SDK Python binding (build from source on the
                 Pi, see pi/setup_pi4.sh).  G4: 230400 baud, 9 kHz, 10 Hz,
                 not single-channel, triangle type, no intensity.
PyLidar3Source : Phase-2 driver, kept only as a fallback (1-degree keys, no
                 timing; flagged in scan.meta so nobody mistakes it for v3 input)
SimLidarSource : simulator
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from typing import Optional

import numpy as np

from phase3_onboard.common import Scan

log = logging.getLogger("lidar")


class LidarSource:
    name = "base"

    def start(self):
        raise NotImplementedError

    def read(self, timeout: float = 1.0) -> Optional[Scan]:
        raise NotImplementedError

    def stop(self):
        pass


class YDLidarSource(LidarSource):
    name = "ydlidar"

    def __init__(self, P):
        self.P = P
        self.laser = None
        self._scan = None
        self.seq = 0
        self.errors = 0
        self.clock_mode = "stamp"  # or "recv" if the SDK stamp looks wrong

    def start(self):
        try:
            import ydlidar
        except ImportError as e:
            raise RuntimeError(
                "YDLidar-SDK Python binding not found. Build it on the Pi with "
                "pi/setup_pi4.sh (or `pip install` from a YDLidar-SDK checkout).") from e
        P = self.P
        self._yd = ydlidar
        ydlidar.os_init()
        L = ydlidar.CYdLidar()
        L.setlidaropt(ydlidar.LidarPropSerialPort, P.LIDAR_PORT)
        L.setlidaropt(ydlidar.LidarPropSerialBaudrate, int(P.LIDAR_BAUD))
        L.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
        L.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
        L.setlidaropt(ydlidar.LidarPropScanFrequency, float(P.LIDAR_SCAN_HZ))
        L.setlidaropt(ydlidar.LidarPropSampleRate, int(P.LIDAR_SAMPLE_KHZ))
        L.setlidaropt(ydlidar.LidarPropSingleChannel, bool(P.LIDAR_SINGLE_CHANNEL))
        L.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)
        L.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
        L.setlidaropt(ydlidar.LidarPropMaxRange, float(P.LIDAR_MAX_RANGE_M))
        L.setlidaropt(ydlidar.LidarPropMinRange, float(P.LIDAR_MIN_RANGE_M))
        L.setlidaropt(ydlidar.LidarPropIntenstiy, bool(P.LIDAR_INTENSITY))
        L.setlidaropt(ydlidar.LidarPropReversion, bool(P.LIDAR_SDK_REVERSION))
        L.setlidaropt(ydlidar.LidarPropInverted, bool(P.LIDAR_SDK_INVERTED))
        L.setlidaropt(ydlidar.LidarPropAutoReconnect, True)
        L.setlidaropt(ydlidar.LidarPropFixedResolution, False)   # raw samples, full resolution
        if not L.initialize():
            raise RuntimeError(f"YDLidar initialize() failed on {P.LIDAR_PORT} "
                               "(check dmesg, power, brltty, udev symlink)")
        if not L.turnOn():
            raise RuntimeError("YDLidar turnOn() failed (check 5 V supply / `vcgencmd get_throttled`)")
        self.laser = L
        self._scan = ydlidar.LaserScan()
        log.info("YDLidar G4 running on %s", P.LIDAR_PORT)

    def read(self, timeout: float = 1.0) -> Optional[Scan]:
        if self.laser is None:
            return None
        if not self.laser.doProcessSimple(self._scan):
            self.errors += 1
            return None
        t_recv = time.monotonic()
        sc = self._scan
        pts = sc.points
        n = pts.size() if hasattr(pts, "size") else len(pts)
        ang = np.empty(n, np.float32)
        rng = np.empty(n, np.float32)
        for i in range(n):
            p = pts[i]
            ang[i] = p.angle
            rng[i] = p.range
        cfg = sc.config
        scan_time = float(getattr(cfg, "scan_time", 0.0)) or (1.0 / self.P.LIDAR_SCAN_HZ)
        tinc = float(getattr(cfg, "time_increment", 0.0))
        tof = (np.arange(n, dtype=np.float32) * tinc) if tinc > 0 else \
            np.linspace(0, scan_time, n, dtype=np.float32)
        span = float(tof[-1]) if n else 0.0
        # SDK stamp is wall-clock ns at the first point; convert to monotonic
        t_start = t_recv - span - self.P.LIDAR_TIMESTAMP_LATENCY_S
        if self.clock_mode == "stamp":
            st = float(getattr(sc, "stamp", 0)) * 1e-9
            if st > 0:
                cand = st - (time.time() - time.monotonic())
                if abs(cand - t_start) < 0.3:
                    t_start = cand
                else:
                    self.clock_mode = "recv"
                    log.warning("YDLidar stamp offset %.3f s from receive time; using receive time",
                                cand - t_start)
        s = Scan(self.seq, t_start, t_start + span, t_recv, ang, rng, tof,
                 meta={"scan_time": scan_time, "tinc": tinc, "clock": self.clock_mode})
        self.seq += 1
        return s

    def stop(self):
        if self.laser is not None:
            try:
                self.laser.turnOff()
                self.laser.disconnecting()
            except Exception:
                pass
            self.laser = None


class PyLidar3Source(LidarSource):
    """Phase-2 fallback.  Output is 1-degree quantised and has no timing."""
    name = "pylidar3"

    def __init__(self, P):
        self.P = P
        self.dev = None
        self.gen = None
        self.seq = 0

    def start(self):
        import PyLidar3
        self.dev = PyLidar3.YdLidarG4(self.P.LIDAR_PORT)
        if not self.dev.Connect():
            raise RuntimeError(f"PyLidar3 could not connect on {self.P.LIDAR_PORT}")
        self.gen = self.dev.StartScanning()
        log.warning("Using PyLidar3 fallback: 1-degree quantised scans, no per-point timing")

    def read(self, timeout: float = 1.0) -> Optional[Scan]:
        try:
            d = next(self.gen)
        except Exception:
            return None
        t = time.monotonic()
        keys = sorted(d.keys())
        ang = np.radians(np.array(keys, dtype=np.float32))
        ang = ((ang + math.pi) % (2 * math.pi) - math.pi).astype(np.float32)
        rng = (np.array([d[k] for k in keys], dtype=np.float32) / 1000.0)
        period = 1.0 / self.P.LIDAR_SCAN_HZ
        tof = np.linspace(0, period, len(keys), dtype=np.float32)
        s = Scan(self.seq, t - period, t, t, ang, rng, tof, meta={"quantised": True})
        self.seq += 1
        return s

    def stop(self):
        try:
            self.dev.StopScanning()
            self.dev.Disconnect()
        except Exception:
            pass


class SimLidarSource(LidarSource):
    name = "sim"

    def __init__(self, live_sim):
        self.sim = live_sim

    def start(self):
        self.sim.start()

    def read(self, timeout: float = 1.0) -> Optional[Scan]:
        try:
            return self.sim.scan_q.get(timeout=timeout)
        except queue.Empty:
            return None


def make_lidar(P, live_sim=None) -> LidarSource:
    if P.LIDAR_DRIVER == "ydlidar":
        return YDLidarSource(P)
    if P.LIDAR_DRIVER == "pylidar3":
        return PyLidar3Source(P)
    if P.LIDAR_DRIVER == "sim":
        from phase3_onboard.sim import live_instance
        return SimLidarSource(live_sim or live_instance())
    raise ValueError(P.LIDAR_DRIVER)


class LidarThread:
    """Reads scans, logs them, and offers only the newest to SLAM (a slow SLAM
    step drops scans instead of building latency; drops are counted)."""

    def __init__(self, source: LidarSource, logger=None):
        self.src = source
        self.logger = logger
        self._latest: Optional[Scan] = None
        self._cv = threading.Condition()
        self._running = False
        self.count = 0
        self.dropped = 0
        self.none_reads = 0
        self.last_t: Optional[float] = None
        self.rate_hz = 0.0
        self.last_n = 0
        self.error: Optional[str] = None

    def start(self):
        self.src.start()
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="lidar").start()

    def stop(self):
        self._running = False
        self.src.stop()

    def _loop(self):
        while self._running:
            try:
                s = self.src.read(timeout=1.0)
            except Exception as e:
                self.error = str(e)
                log.error("LiDAR read error: %s", e)
                time.sleep(0.2)
                continue
            if s is None:
                self.none_reads += 1
                continue
            if self.logger:
                self.logger.scan(s)
            if self.last_t is not None and s.t_end > self.last_t:
                inst = 1.0 / (s.t_end - self.last_t)
                self.rate_hz = inst if self.rate_hz == 0 else 0.9 * self.rate_hz + 0.1 * inst
            self.last_t = s.t_end
            self.last_n = s.n
            self.count += 1
            with self._cv:
                if self._latest is not None:
                    self.dropped += 1
                self._latest = s
                self._cv.notify()

    def get(self, timeout: float = 1.0) -> Optional[Scan]:
        with self._cv:
            if self._latest is None:
                self._cv.wait(timeout)
            s, self._latest = self._latest, None
            return s

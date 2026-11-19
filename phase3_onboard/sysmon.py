"""CPU %, SoC temperature, throttling flags and RSS — for G1/G3 and the paper."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

# vcgencmd get_throttled bits (Pi 4B and 5)
THROTTLE_BITS = {
    0: "under-voltage now", 1: "arm freq capped now", 2: "throttled now", 3: "soft temp limit now",
    16: "under-voltage occurred", 17: "arm freq capped occurred", 18: "throttling occurred",
    19: "soft temp limit occurred",
}


def read_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def read_throttled():
    exe = shutil.which("vcgencmd")
    if not exe:
        return None
    try:
        out = subprocess.check_output([exe, "get_throttled"], timeout=1).decode()
        return int(out.strip().split("=")[1], 16)
    except Exception:
        return None


def decode_throttled(v):
    if v is None:
        return []
    return [name for bit, name in THROTTLE_BITS.items() if v & (1 << bit)]


def _cpu_times():
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = list(map(int, parts))
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals), idle


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return None


class SysMon:
    def __init__(self, logger=None, hz: float = 1.0):
        self.logger = logger
        self.period = 1.0 / hz
        self.state = {}
        self._running = False
        self.max_temp = None
        self.throttled_ever = 0

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="sysmon").start()

    def stop(self):
        self._running = False

    def _loop(self):
        try:
            prev = _cpu_times()
        except Exception:
            prev = None
        k = 0
        thr = None
        while self._running:
            time.sleep(self.period)
            cpu = None
            if prev is not None:
                try:
                    cur = _cpu_times()
                    dt, di = cur[0] - prev[0], cur[1] - prev[1]
                    cpu = round(100.0 * (1 - di / dt), 1) if dt > 0 else None
                    prev = cur
                except Exception:
                    pass
            if k % 5 == 0:
                thr = read_throttled()
                if thr:
                    self.throttled_ever |= thr
            k += 1
            temp = read_temp_c()
            if temp is not None:
                self.max_temp = temp if self.max_temp is None else max(self.max_temp, temp)
            self.state = {"cpu": cpu, "temp": temp, "max_temp": self.max_temp,
                          "throttled": None if thr is None else hex(thr),
                          "throttled_flags": decode_throttled(thr), "rss_mb": rss_mb(),
                          "load1": os.getloadavg()[0] if hasattr(os, "getloadavg") else None}
            if self.logger:
                self.logger.event("sys", **self.state)

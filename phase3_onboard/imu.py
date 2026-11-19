"""
IMU parsing and gyro yaw integration.
=====================================
The WAVE ROVER has no wheel encoders, so the motion prior is:
    rotation    : integrated gyro z rate (bias removed)
    translation : constant velocity from the last accepted SLAM motion

`ImuParser` turns raw rover JSON lines into `ImuSample`s using the field map
in config (fix it after running tools/imu_probe.py on the real stream).
`GyroYaw` keeps a short history and answers "yaw change between t_a and t_b".
"""

from __future__ import annotations

import bisect
import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ImuSample:
    t: float
    rate: Optional[float]      # rad/s CCW, bias NOT removed
    yaw: Optional[float]       # rad, absolute (firmware fusion; may include magnetometer)
    raw: dict


class ImuParser:
    def __init__(self, cfg):
        self.types = set(cfg.IMU_REPLY_TYPES)
        self.gz_key = cfg.IMU_GYRO_Z_KEY
        self.gz_scale = cfg.IMU_GYRO_SCALE * cfg.IMU_GYRO_SIGN
        self.yaw_key = cfg.IMU_YAW_KEY
        self.yaw_sign = cfg.IMU_YAW_SIGN
        self.volt_key = cfg.FEEDBACK_VOLTAGE_KEY
        self.last_voltage: Optional[float] = None

    @staticmethod
    def parse_line(line: str) -> Optional[dict]:
        line = line.strip()
        if not line.startswith("{"):
            return None
        try:
            d = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None
        return d if isinstance(d, dict) else None

    def sample(self, t: float, msg: dict) -> Optional[ImuSample]:
        if msg.get("T") not in self.types:
            return None
        if self.volt_key in msg:
            try:
                self.last_voltage = float(msg[self.volt_key])
            except (TypeError, ValueError):
                pass
        rate = yaw = None
        if self.gz_key and self.gz_key in msg:
            try:
                rate = float(msg[self.gz_key]) * self.gz_scale
            except (TypeError, ValueError):
                rate = None
        if self.yaw_key and self.yaw_key in msg:
            try:
                yaw = math.radians(float(msg[self.yaw_key])) * self.yaw_sign
            except (TypeError, ValueError):
                yaw = None
        if rate is None and yaw is None:
            return None
        return ImuSample(t, rate, yaw, msg)


class GyroYaw:
    """Integrated relative yaw from gyro (or differenced absolute yaw).

    source = "gyro"     : integrate rate (trapezoid), bias estimated at standstill
    source = "yaw_diff" : unwrap the firmware's yaw angle
    source = "none"     : always 0 (prior = constant velocity only)
    """

    def __init__(self, source: str = "gyro", history_s: float = 5.0, bias_window_s: float = 2.0):
        self.source = source
        self.hist_t: deque = deque()
        self.hist_yaw: deque = deque()
        self.history_s = history_s
        self.bias = 0.0
        self.bias_window_s = bias_window_s
        self._bias_acc: list = []
        self._bias_t0: Optional[float] = None
        self.bias_ready = False
        self._last_t: Optional[float] = None
        self._last_rate: Optional[float] = None
        self._yaw = 0.0
        self._last_abs: Optional[float] = None
        self.n = 0
        self.last_rate_corrected = 0.0
        self.scale_corr = 1.0          # online correction (SlamEngine, bounded)

    # -- feeding -------------------------------------------------------------------
    def add(self, s: ImuSample, stationary: bool = False):
        if self.source == "gyro":
            if s.rate is None:
                return
            if stationary:
                self._bias_update(s.t, s.rate)
            r = (s.rate - self.bias) * self.scale_corr
            self.last_rate_corrected = r
            if self._last_t is not None:
                dt = s.t - self._last_t
                if 0 < dt < 0.5:
                    self._yaw += 0.5 * (r + self._last_rate) * dt
                elif dt >= 0.5:
                    # gap: assume the rate held; better than losing the turn
                    self._yaw += r * min(dt, 1.0)
            self._last_t = s.t
            self._last_rate = r
        elif self.source == "yaw_diff":
            if s.yaw is None:
                return
            if self._last_abs is not None:
                d = (s.yaw - self._last_abs + math.pi) % (2 * math.pi) - math.pi
                if self._last_t is not None and s.t > self._last_t:
                    self.last_rate_corrected = d / (s.t - self._last_t)
                self._yaw += d
            self._last_abs = s.yaw
            self._last_t = s.t
        else:
            return
        if self.hist_t and s.t <= self.hist_t[-1]:
            return
        self.hist_t.append(s.t)
        self.hist_yaw.append(self._yaw)
        self.n += 1
        while self.hist_t and self.hist_t[0] < s.t - self.history_s:
            self.hist_t.popleft()
            self.hist_yaw.popleft()

    def _bias_update(self, t: float, rate: float):
        if self._bias_t0 is None:
            self._bias_t0 = t
        self._bias_acc.append(rate)
        if len(self._bias_acc) > 400:
            self._bias_acc.pop(0)
        if t - self._bias_t0 >= self.bias_window_s and len(self._bias_acc) >= 10:
            new_bias = float(np.median(self._bias_acc))
            if not self.bias_ready:
                self.bias = new_bias
                self.bias_ready = True
            else:  # slow tracking while stationary
                self.bias = 0.98 * self.bias + 0.02 * new_bias

    def reset_bias_window(self):
        self._bias_acc.clear()
        self._bias_t0 = None

    # -- queries -------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.source != "none" and len(self.hist_t) >= 2

    @property
    def latest_t(self) -> Optional[float]:
        return self.hist_t[-1] if self.hist_t else None

    def yaw_at(self, t: float) -> float:
        """Relative integrated yaw at time t (linear interpolation, linear extrapolation
        from the last rate for a short time past the newest sample)."""
        if not self.hist_t:
            return 0.0
        ht = self.hist_t
        if t >= ht[-1]:
            dt = min(t - ht[-1], 0.15)
            return self.hist_yaw[-1] + self.last_rate_corrected * dt
        if t <= ht[0]:
            return self.hist_yaw[0]
        i = bisect.bisect_left(ht, t)
        t0, t1 = ht[i - 1], ht[i]
        y0, y1 = self.hist_yaw[i - 1], self.hist_yaw[i]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return y0 + a * (y1 - y0)

    def yaw_at_many(self, ts: np.ndarray) -> np.ndarray:
        if not self.hist_t:
            return np.zeros_like(ts)
        ht = np.fromiter(self.hist_t, dtype=np.float64)
        hy = np.fromiter(self.hist_yaw, dtype=np.float64)
        out = np.interp(ts, ht, hy)
        late = ts > ht[-1]
        if np.any(late):
            out[late] = hy[-1] + self.last_rate_corrected * np.minimum(ts[late] - ht[-1], 0.15)
        return out

    def delta(self, t_a: float, t_b: float) -> float:
        return self.yaw_at(t_b) - self.yaw_at(t_a)

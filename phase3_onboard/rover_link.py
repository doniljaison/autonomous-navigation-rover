"""
Rover link — the ONLY code that talks to the ESP32.
====================================================
* one writer thread streams {"T":1,"L":..,"R":..} at CMD_HZ (duty, not m/s)
* drive *intents* carry a TTL; a stale intent becomes L=R=0 (software watchdog)
* zeros are sent on stop(), on exceptions, and by main.py's SIGTERM/SIGINT/atexit
* the firmware's own 3 s timeout is the last line of defence
* every received line is timestamped on arrival (time.monotonic) and handed to
  subscribers raw; every sent line is logged raw
* IMU poll ({"T":126}) is interleaved by the same writer, so there is exactly
  one writer on the port

Transports: UartTransport (Pi 4B header UART), HttpTransport (old WiFi path,
kept for the latency comparison), SimTransport (simulator).
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
import urllib.parse
from collections import Counter, deque
from typing import Callable, Optional

log = logging.getLogger("link")

LineCallback = Callable[[float, str], None]


# ─────────────────────────────────────────────────────────────────────────────
#  Transports
# ─────────────────────────────────────────────────────────────────────────────

class Transport:
    name = "base"

    def __init__(self):
        self.on_line: Optional[LineCallback] = None

    def open(self):
        raise NotImplementedError

    def write_line(self, line: str):
        raise NotImplementedError

    def close(self):
        pass


class UartTransport(Transport):
    name = "uart"

    def __init__(self, port: str, baud: int = 115200):
        super().__init__()
        self.port, self.baud = port, baud
        self._ser = None
        self._running = False
        self._wlock = threading.Lock()
        self.rx_errors = 0

    def open(self):
        import serial
        self._ser = serial.Serial()
        self._ser.port = self.port
        self._ser.baudrate = self.baud
        self._ser.timeout = 0.05
        self._ser.write_timeout = 0.2
        # keep RTS/DTR low: on the ESP32's USB port these lines reset the chip
        self._ser.rts = False
        self._ser.dtr = False
        self._ser.open()
        self._ser.reset_input_buffer()
        self._running = True
        threading.Thread(target=self._reader, daemon=True, name="uart-rx").start()
        log.info("UART open %s @ %d", self.port, self.baud)

    def _reader(self):
        buf = b""
        while self._running:
            try:
                chunk = self._ser.read(self._ser.in_waiting or 1)
            except Exception as e:  # unplugged / port error
                self.rx_errors += 1
                log.error("UART read error: %s", e)
                time.sleep(0.2)
                continue
            if not chunk:
                continue
            t = time.monotonic()
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace").strip()
                if line and self.on_line:
                    self.on_line(t, line)
            if len(buf) > 8192:      # garbage without newlines
                buf = b""
                self.rx_errors += 1

    def write_line(self, line: str):
        with self._wlock:
            self._ser.write((line + "\n").encode("ascii"))

    def close(self):
        self._running = False
        try:
            if self._ser:
                self._ser.close()
        except Exception:
            pass


class HttpTransport(Transport):
    """GET http://host/js?json=... ; the JSON reply (if any) is delivered as an rx
    line.  Requests run on a worker so a slow WiFi request never blocks the
    caller; if the worker backs up, the oldest pending motor command is dropped."""
    name = "http"

    def __init__(self, host: str, port: int = 80, endpoint: str = "/js", timeout: float = 0.5):
        super().__init__()
        self.host, self.port, self.endpoint, self.timeout = host, port, endpoint, timeout
        self._q: deque = deque(maxlen=4)
        self._cv = threading.Condition()
        self._running = False
        self.errors = 0

    def open(self):
        self._running = True
        threading.Thread(target=self._worker, daemon=True, name="http-tx").start()

    def write_line(self, line: str):
        with self._cv:
            self._q.append(line)
            self._cv.notify()

    def _worker(self):
        while self._running:
            with self._cv:
                while not self._q and self._running:
                    self._cv.wait(0.5)
                if not self._running:
                    return
                line = self._q.popleft()
            q = urllib.parse.quote(line, safe="")
            req = (f"GET {self.endpoint}?json={q} HTTP/1.0\r\nHost: {self.host}\r\n"
                   f"Connection: close\r\n\r\n").encode("ascii")
            try:
                s = socket.create_connection((self.host, self.port), timeout=self.timeout)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.sendall(req)
                chunks = []
                while True:
                    c = s.recv(2048)
                    if not c:
                        break
                    chunks.append(c)
                s.close()
                t = time.monotonic()
                body = b"".join(chunks).split(b"\r\n\r\n", 1)
                if len(body) == 2 and self.on_line:
                    for ln in body[1].decode("utf-8", "replace").splitlines():
                        ln = ln.strip()
                        if ln.startswith("{"):
                            self.on_line(t, ln)
            except OSError:
                self.errors += 1

    def close(self):
        self._running = False
        with self._cv:
            self._cv.notify_all()


class SimTransport(Transport):
    name = "sim"

    def __init__(self, live_sim):
        super().__init__()
        self.sim = live_sim
        self._running = False

    def open(self):
        self.sim.start()
        self._running = True
        threading.Thread(target=self._reader, daemon=True, name="sim-rx").start()

    def _reader(self):
        import queue
        while self._running:
            try:
                t, line = self.sim.rx_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.on_line:
                self.on_line(t, line)

    def write_line(self, line: str):
        self.sim.tx(line)

    def close(self):
        self._running = False


def make_transport(P, live_sim=None) -> Transport:
    if P.LINK_TRANSPORT == "uart":
        return UartTransport(P.UART_PORT, P.UART_BAUD)
    if P.LINK_TRANSPORT == "http":
        return HttpTransport(P.ROVER_HTTP_HOST, P.ROVER_HTTP_PORT, P.ROVER_HTTP_ENDPOINT)
    if P.LINK_TRANSPORT == "sim":
        from phase3_onboard.sim import live_instance
        return SimTransport(live_sim or live_instance())
    raise ValueError(P.LINK_TRANSPORT)


# ─────────────────────────────────────────────────────────────────────────────
#  RoverLink
# ─────────────────────────────────────────────────────────────────────────────

class RoverLink:
    def __init__(self, transport: Transport, P, logger=None):
        self.tr = transport
        self.P = P
        self.logger = logger
        self._subs: list[LineCallback] = []
        self._lock = threading.Lock()
        self._intent = (0.0, 0.0)
        self._intent_src = "none"
        self._intent_expiry = 0.0
        self._extra: deque = deque(maxlen=32)
        self._wake = threading.Event()
        self._running = False
        self._th: Optional[threading.Thread] = None
        self.last_sent = (0.0, 0.0)
        self.last_nonzero_t = -1e9
        self.tx_count = 0
        self.rx_count = 0
        self.rx_types: Counter = Counter()
        self.last_rx_t: Optional[float] = None
        self.tx_errors = 0
        self.watchdog_trips = 0
        self._was_fresh = False
        self.tr.on_line = self._on_line

    # -- subscribers ---------------------------------------------------------------
    def subscribe(self, cb: LineCallback):
        self._subs.append(cb)

    def _on_line(self, t: float, line: str):
        self.rx_count += 1
        self.last_rx_t = t
        try:
            if line.startswith("{"):
                i = line.find('"T":')
                if i >= 0:
                    j = i + 4
                    k = j
                    while k < len(line) and (line[k].isdigit() or line[k] == "-"):
                        k += 1
                    self.rx_types[line[j:k]] += 1
        except Exception:
            pass
        if self.logger:
            self.logger.event("rx", t=t, line=line)
        for cb in self._subs:
            try:
                cb(t, line)
            except Exception as e:  # a subscriber must never kill the reader
                log.error("rx subscriber error: %s", e)

    # -- lifecycle -------------------------------------------------------------------
    def start(self):
        self.tr.open()
        if self.P.ECHO_OFF:
            self.send_raw({"T": 143, "cmd": 0})
        if self.P.FEEDBACK_STREAM:
            self.send_raw({"T": 131, "cmd": 1})
        self._running = True
        self._th = threading.Thread(target=self._writer, daemon=True, name="link-tx")
        self._th.start()

    def stop(self):
        """Stop the stream and leave the rover stopped (best effort, idempotent)."""
        self._running = False
        if self._th and self._th.is_alive() and self._th is not threading.current_thread():
            self._th.join(timeout=1.0)
        for _ in range(3):
            self._write(0.0, 0.0)
            time.sleep(0.02)
        self.tr.close()

    # -- commands --------------------------------------------------------------------
    def set_intent(self, left: float, right: float, source: str = "?", ttl: Optional[float] = None):
        ttl = self.P.INTENT_TTL_S if ttl is None else ttl
        with self._lock:
            self._intent = (float(left), float(right))
            self._intent_src = source
            self._intent_expiry = time.monotonic() + ttl

    def stop_now(self, source: str = "stop"):
        with self._lock:
            self._intent = (0.0, 0.0)
            self._intent_src = source
            self._intent_expiry = 0.0
        self._write(0.0, 0.0)

    def send_raw(self, msg: dict):
        """Queue an arbitrary command (sent by the writer thread, never concurrently)."""
        self._extra.append(json.dumps(msg, separators=(",", ":")))
        self._wake.set()
        if not self._running:  # before start(): write directly
            try:
                self._send_line(self._extra.pop())
            except Exception:
                pass

    # -- writer ----------------------------------------------------------------------
    def _send_line(self, line: str):
        self.tr.write_line(line)
        self.tx_count += 1
        if self.logger:
            self.logger.event("tx", line=line)

    def _write(self, L: float, R: float):
        P = self.P
        lim = min(P.MAX_DUTY, 0.5)
        L = max(-lim, min(lim, L))
        R = max(-lim, min(lim, R))
        if P.INVERT_FWD:
            L, R = -L, -R
        if P.SWAP_LR:
            L, R = R, L
        try:
            self._send_line(json.dumps({"T": 1, "L": round(L, 3), "R": round(R, 3)}, separators=(",", ":")))
            self.last_sent = (L, R)
            if abs(L) > 1e-3 or abs(R) > 1e-3:
                self.last_nonzero_t = time.monotonic()
        except Exception as e:
            self.tx_errors += 1
            if self.tx_errors in (1, 10, 100) or self.tx_errors % 1000 == 0:
                log.error("link write failed (%d): %s", self.tx_errors, e)

    def _writer(self):
        period = 1.0 / self.P.CMD_HZ
        imu_period = 1.0 / self.P.IMU_POLL_HZ if self.P.IMU_POLL_HZ > 0 else None
        next_cmd = time.monotonic()
        next_imu = next_cmd
        while self._running:
            now = time.monotonic()
            try:
                if now >= next_cmd:
                    with self._lock:
                        fresh = now < self._intent_expiry
                        L, R = self._intent if fresh else (0.0, 0.0)
                    if self._was_fresh and not fresh and (abs(self._intent[0]) + abs(self._intent[1]) > 0):
                        self.watchdog_trips += 1
                        log.warning("drive intent from '%s' expired -> stop", self._intent_src)
                    self._was_fresh = fresh
                    self._write(L, R)
                    next_cmd += period
                    if next_cmd < now:
                        next_cmd = now + period
                while self._extra:
                    self._send_line(self._extra.popleft())
                if imu_period and now >= next_imu:
                    self._send_line('{"T":126}')
                    next_imu += imu_period
                    if next_imu < now:
                        next_imu = now + imu_period
            except Exception as e:
                self.tx_errors += 1
                log.error("writer error: %s", e)
                try:
                    self._write(0.0, 0.0)
                except Exception:
                    pass
                time.sleep(0.05)
            nxt = min(next_cmd, next_imu) if imu_period else next_cmd
            sl = nxt - time.monotonic()
            if sl > 0 and not self._extra:
                self._wake.wait(min(sl, 0.02))
            self._wake.clear()
        # leaving the loop: make sure the last thing sent is a stop
        self._write(0.0, 0.0)

    # -- status ----------------------------------------------------------------------
    def status(self) -> dict:
        now = time.monotonic()
        return {
            "transport": self.tr.name,
            "tx": self.tx_count, "rx": self.rx_count,
            "rx_types": dict(self.rx_types),
            "rx_age_s": None if self.last_rx_t is None else round(now - self.last_rx_t, 3),
            "last_cmd": [round(self.last_sent[0], 3), round(self.last_sent[1], 3)],
            "intent_src": self._intent_src,
            "tx_errors": self.tx_errors,
            "watchdog_trips": self.watchdog_trips,
        }

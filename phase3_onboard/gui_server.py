"""
GUI link server (runs inside `main --headless`).
================================================
A small TCP server that streams what the desktop GUI (phase3_onboard/gui.py) needs and
accepts its commands.  It replaces the HTTP UI's polling with a push stream:

  Pi -> GUI   frames  [4-byte big-endian length][1 type byte][payload]
      'J'  JSON  state ~10 Hz (pose, thinned scan, path, goal, health), trail ~1 Hz, notes, pong
      'M'  map   [2-byte header length][JSON header][zlib(uint8 cells)]   ~1 Hz, only when changed
           cells: 0 unknown, 1 free, 2 occupied, cropped to the known area (header r0, c0, h, w)
  GUI -> Pi   one JSON object per line: {"cmd": "drive"|"goal"|"stop"|"save"|"snap"|"resume"|
              "spin"|"fwd"|"ping", ...}

Safety: a client that has sent any motion command (drive / goal / test) is a *controller*.
If a controller disconnects or is silent for GUI_HEARTBEAT_S (the GUI pings at 5 Hz), the
rover is stopped and any goal cancelled -- the same guarantee as the web UI's watchdog.
Manual drive intents still expire after INTENT_TTL_S in the link writer as before.
Work done here is small (JSON of ~250 points, one zlib of a cropped uint8 map per second),
so it does not compete with SLAM the way PNG + HTTP polling did.
"""

from __future__ import annotations

import json
import logging
import math
import re
import socket
import struct
import threading
import time
import zlib
from collections import deque

import numpy as np

log = logging.getLogger("gui")

STATE_HZ = 10.0
MAP_HZ = 1.0
TRAIL_HZ = 1.0
GUI_HEARTBEAT_S = 1.0
_IMU_RE = re.compile(r'"T"\s*:\s*1002\b')


def _frame(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 1) + kind + payload


def _jframe(obj) -> bytes:
    return _frame(b"J", json.dumps(obj, separators=(",", ":"), default=str).encode())


class ImuGapMonitor:
    """Arrival-time health of the gyro stream (T:1002), fed from the link reader thread.
    Cheap: one regex search and one deque append per received line."""

    def __init__(self, window_s=5.0):
        self.t = deque(maxlen=1000)
        self.window_s = window_s

    def on_line(self, t, line):
        if _IMU_RE.search(line):
            self.t.append(t)

    def stats(self):
        now = time.monotonic()
        ts = [x for x in list(self.t) if x > now - self.window_s]
        if len(ts) < 3:
            return {"hz": 0.0, "max_gap_ms": None, "bunched_pct": None, "age_s": None if not self.t else round(now - self.t[-1], 2)}
        g = np.diff(np.array(ts)) * 1e3
        return {"hz": round(len(ts) / self.window_s, 1), "max_gap_ms": round(float(g.max()), 0),
                "p95_gap_ms": round(float(np.percentile(g, 95)), 0),
                "bunched_pct": round(float((g < 1.0).mean() * 100), 0), "age_s": round(now - ts[-1], 2)}


class _Client:
    def __init__(self, srv, sock, addr):
        self.srv, self.sock, self.addr = srv, sock, addr
        self.alive = True
        self.controller = False
        self.last_rx = time.monotonic()
        self.map_version = -1
        self.wlock = threading.Lock()

    def send(self, data: bytes):
        with self.wlock:
            self.sock.sendall(data)


class GuiServer:
    def __init__(self, rt, host="0.0.0.0", port=8765):
        self.rt = rt
        self.host, self.port = host, port
        self.clients: list[_Client] = []
        self.clock = threading.Lock()
        self._running = False
        self.imu = ImuGapMonitor()
        rt.link.subscribe(self.imu.on_line)
        from phase3_onboard.headless import _Test
        self.test = _Test(rt)
        self._map_cache = (-1, None)
        self._trail_t = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self):
        self.lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.lsock.bind((self.host, self.port))
        self.lsock.listen(4)
        self._running = True
        threading.Thread(target=self._accept, daemon=True, name="gui-accept").start()
        threading.Thread(target=self._pump, daemon=True, name="gui-pump").start()
        self.rt.note(f"GUI server on port {self.port} (python phase3_onboard/gui.py --host <this-pi>)")

    def stop(self):
        self._running = False
        try:
            self.lsock.close()
        except Exception:
            pass
        with self.clock:
            for c in self.clients:
                self._close(c, quiet=True)

    def _accept(self):
        while self._running:
            try:
                s, addr = self.lsock.accept()
            except OSError:
                return
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(2.0)          # a stalled client can block the pump for at most 2 s
            c = _Client(self, s, addr)
            with self.clock:
                self.clients.append(c)
            self.rt.note(f"GUI connected from {addr[0]}")
            threading.Thread(target=self._reader, args=(c,), daemon=True, name="gui-rx").start()

    def _close(self, c: _Client, why="disconnected", quiet=False):
        if not c.alive:
            return
        c.alive = False
        try:
            c.sock.close()
        except Exception:
            pass
        if c.controller and not quiet:
            self.test.cancel.set()
            self.rt.stop()
            self.rt.note(f"GUI {c.addr[0]} {why} -> rover stopped")
        elif not quiet:
            self.rt.note(f"GUI {c.addr[0]} {why}")

    # ── client -> Pi ─────────────────────────────────────────────────────────
    def _reader(self, c: _Client):
        buf = b""
        f = c.sock
        while c.alive and self._running:
            try:
                chunk = f.recv(4096)
            except socket.timeout:
                continue                 # the heartbeat check in _pump decides about silence
            except OSError:
                chunk = b""
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                c.last_rx = time.monotonic()
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                try:
                    self._handle(c, msg)
                except Exception as e:
                    log.exception("GUI command %s", msg)
                    self._reply(c, f"error: {e}")
            if len(buf) > 65536:
                buf = b""
        with self.clock:
            if c in self.clients:
                self.clients.remove(c)
        self._close(c)

    def _reply(self, c, text):
        try:
            c.send(_jframe({"type": "note", "msg": text}))
        except OSError:
            pass

    def _handle(self, c: _Client, m: dict):
        rt = self.rt
        cmd = m.get("cmd")
        if cmd == "ping":
            c.send(_jframe({"type": "pong", "t": m.get("t")}))
        elif cmd == "drive":
            fwd, turn = float(m.get("fwd", 0)), float(m.get("turn", 0))
            if fwd or turn:
                c.controller = True
            rt.drive(fwd, turn)
        elif cmd == "goal":
            if self.test.running():
                return self._reply(c, "a test is running; stop first")
            c.controller = True
            ok = rt.set_goal(float(m["x"]), float(m["y"]))
            self._reply(c, "goal accepted" if ok else "goal refused (SLAM LOST? press Resume)")
        elif cmd == "stop":
            self.test.cancel.set()
            rt.stop()
        elif cmd == "save":
            self._reply(c, "map saved: " + rt.save_map())
        elif cmd == "snap":
            from phase3_onboard.headless import _snap_png
            _snap_png(rt)
        elif cmd == "resume":
            rt.clear_lost()
        elif cmd in ("spin", "fwd"):
            duty, sec = float(m["duty"]), float(m["sec"])
            if not (0.05 <= abs(duty) <= 0.45 and 0 < sec <= 5.0):
                return self._reply(c, "test refused: duty 0.05..0.45, seconds <= 5")
            with rt.lock:
                auto = rt.mode != "MANUAL"
            if auto or self.test.running():
                return self._reply(c, "test refused: goal or test active (Stop first)")
            c.controller = True
            self.test.start(cmd, duty, sec)
        else:
            self._reply(c, f"unknown command {cmd!r}")

    # ── Pi -> client ─────────────────────────────────────────────────────────
    def _state(self) -> dict:
        rt = self.rt
        snap = rt.eng.snapshot()
        with rt.lock:
            mode, goal, path = rt.mode, rt.goal, list(rt.path)
        sw = snap["scan_world"]
        if len(sw) > 300:
            sw = sw[np.linspace(0, len(sw) - 1, 300).astype(np.int64)]
        P = rt.P
        d = {
            "type": "state", "t": time.monotonic(),
            "mode": mode, "phase": rt.nav_phase,
            "pose": [round(float(v), 4) for v in snap["pose"]],
            "lost": bool(snap["lost"]),
            "goal": None if goal is None else [round(float(goal[0]), 3), round(float(goal[1]), 3)],
            "path": [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in path[::2]],
            "scan": np.round(sw, 3).tolist(),
            "slam": {"frames": snap["frames"], "accepted": snap["accepted"], "rejected": snap["rejected"],
                     "skipped": snap["skipped"], "reject_rate": round(snap["reject_rate"], 3),
                     "hz": round(rt.slam_hz, 1), "last": snap["last"], "gyro_ready": snap["gyro_ready"],
                     "gyro_bias_dps": round(math.degrees(snap["gyro_bias"]), 3),
                     "gyro_scale_corr": round(snap["gyro_scale_corr"], 3)},
            "imu": self.imu.stats(),
            "link": rt.link.status(),
            "lidar": {"rate_hz": round(rt.lidar.rate_hz, 2), "points": rt.lidar.last_n,
                      "dropped": rt.lidar.dropped, "error": rt.lidar.error},
            "prox": rt.prox.as_dict(),
            "sys": rt.sysmon.state,
            "test": self.test.running(),
            "notes": list(rt.notes)[-10:],
            "map_meta": {"res": rt.eng.grid.res, "orig": rt.eng.grid.orig, "n": rt.eng.grid.n},
            "limits": {"manual_max": P.MANUAL_MAX_DUTY, "manual_turn": P.MANUAL_TURN_DUTY},
        }
        if getattr(rt, "live_sim", None) is not None:
            try:
                from phase3_onboard.common import relative
                rel = relative(np.array(rt.live_sim.core.p.start_pose), rt.live_sim.truth_pose)
                d["sim_truth"] = [round(float(v), 4) for v in rel]
            except Exception:
                pass
        return d

    def _map_frame(self):
        rt = self.rt
        v = rt.eng.grid.version
        if v == self._map_cache[0]:
            return self._map_cache
        P = rt.P
        lo = rt.eng.map_copy()
        cells = np.zeros(lo.shape, np.uint8)
        cells[lo < P.MAP_FREE_THRESH] = 1
        cells[lo > P.MAP_OCC_THRESH] = 2
        rows = np.flatnonzero(cells.any(axis=1))
        cols = np.flatnonzero(cells.any(axis=0))
        if len(rows):
            r0, r1 = int(rows[0]), int(rows[-1]) + 1
            c0, c1 = int(cols[0]), int(cols[-1]) + 1
        else:
            r0 = r1 = c0 = c1 = 0
        crop = np.ascontiguousarray(cells[r0:r1, c0:c1])
        g = rt.eng.grid
        hdr = json.dumps({"version": v, "n": g.n, "res": g.res, "orig": g.orig,
                          "r0": r0, "c0": c0, "h": r1 - r0, "w": c1 - c0}).encode()
        payload = struct.pack(">H", len(hdr)) + hdr + zlib.compress(crop.tobytes(), 1)
        self._map_cache = (v, _frame(b"M", payload))
        return self._map_cache

    def _pump(self):
        period = 1.0 / STATE_HZ
        next_map = 0.0
        nxt = time.monotonic()
        while self._running:
            now = time.monotonic()
            with self.clock:
                clients = [c for c in self.clients if c.alive]
            # heartbeat watchdog
            for c in clients:
                if now - c.last_rx > GUI_HEARTBEAT_S:
                    with self.clock:
                        if c in self.clients:
                            self.clients.remove(c)
                    self._close(c, why=f"silent for {now - c.last_rx:.1f} s")
            clients = [c for c in clients if c.alive]
            if clients:
                try:
                    st = _jframe(self._state())
                    mf = None
                    if now >= next_map:
                        next_map = now + 1.0 / MAP_HZ
                        mv, mf = self._map_frame()
                    tf = None
                    if now - self._trail_t >= 1.0 / TRAIL_HZ:
                        self._trail_t = now
                        with self.rt.eng.lock:
                            trail = list(self.rt.eng.path)[-1500::3]
                        tf = _jframe({"type": "trail", "pts": [[round(p[1], 3), round(p[2], 3)] for p in trail]})
                    for c in clients:
                        try:
                            c.send(st)
                            if mf is not None and c.map_version != mv:
                                c.send(mf)
                                c.map_version = mv
                            if tf is not None:
                                c.send(tf)
                        except OSError:
                            with self.clock:
                                if c in self.clients:
                                    self.clients.remove(c)
                            self._close(c, why="send failed")
                except Exception:
                    log.exception("GUI pump")
            nxt += period
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.monotonic()

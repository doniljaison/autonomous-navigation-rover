"""
Run logger and reader.
======================
Each run is a directory ``runs/<YYYYmmdd_HHMMSS>_<tag>/`` with

    meta.json     platform, config snapshot, git revision, notes
    scans.bin     framed binary LiDAR revolutions (full resolution, raw angles)
    events.jsonl  one JSON object per line, all with "t" (time.monotonic seconds)
                  and "k" (kind):
                    rx    raw line received from the rover   {"line": "..."}
                    tx    raw line sent to the rover         {"line": "..."}
                    slam  SLAM output per scan               {pose, accepted, ...}
                    sys   CPU %, temperature, throttled flags
                    mode  mode changes / goals / notes
                    truth simulator ground-truth pose        {"pose": [x,y,th]}

Raw rover lines are stored unparsed so that a corrected IMU parser can be
re-applied to old logs.  Writing happens on a background thread so the SLAM
loop never blocks on the SD card.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import struct
import subprocess
import threading
import time
from typing import Iterator

import numpy as np

from phase3_onboard.common import Scan

_MAGIC = b"SCN1"
_HDR = struct.Struct("<4sIdddI")   # magic, seq, t_start, t_end, t_recv, n


def _git_rev() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, timeout=2).decode().strip()
    except Exception:
        return "unknown"


class RunLogger:
    def __init__(self, root: str, tag: str = "run", meta: dict | None = None,
                 run_dir: str | None = None):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = run_dir or os.path.join(root, f"{stamp}_{tag}")
        os.makedirs(self.dir, exist_ok=True)
        from phase3_onboard import config
        m = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tag": tag,
            "host": platform.node(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "git": _git_rev(),
            "mono_to_wall": time.time() - time.monotonic(),
            "config": config.snapshot(),
        }
        if meta:
            m.update(meta)
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(m, f, indent=1, default=str)
        self._scans = open(os.path.join(self.dir, "scans.bin"), "wb")
        self._events = open(os.path.join(self.dir, "events.jsonl"), "w", buffering=1 << 16)
        self._q: queue.Queue = queue.Queue(maxsize=20000)
        self.dropped = 0
        self._closed = False
        self._th = threading.Thread(target=self._run, daemon=True, name="logger")
        self._th.start()

    # -- producers (any thread) ------------------------------------------------
    def scan(self, s: Scan):
        self._put(("scan", s))

    def event(self, kind: str, t: float | None = None, **fields):
        fields["t"] = time.monotonic() if t is None else t
        fields["k"] = kind
        self._put(("ev", fields))

    def _put(self, item):
        if self._closed:
            return
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    # -- writer thread -----------------------------------------------------------
    def _run(self):
        last_flush = time.monotonic()
        while True:
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                item = None
            if item is not None:
                if item[0] == "stop":
                    break
                if item[0] == "scan":
                    write_scan(self._scans, item[1])
                else:
                    self._events.write(json.dumps(item[1], separators=(",", ":"), default=_jdefault) + "\n")
            if time.monotonic() - last_flush > 1.0:
                self._scans.flush()
                self._events.flush()
                last_flush = time.monotonic()
        self._scans.flush()
        self._events.flush()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._q.put(("stop", None))
        self._th.join(timeout=5)
        self._scans.close()
        self._events.close()


def _jdefault(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def write_scan(f, s: Scan):
    n = s.n
    f.write(_HDR.pack(_MAGIC, s.seq, s.t_start, s.t_end, s.t_recv, n))
    f.write(np.asarray(s.angle, dtype=np.float32).tobytes())
    f.write(np.asarray(s.range, dtype=np.float32).tobytes())
    f.write(np.asarray(s.tof, dtype=np.float32).tobytes())


def read_scans(path: str) -> Iterator[Scan]:
    with open(path, "rb") as f:
        while True:
            hdr = f.read(_HDR.size)
            if len(hdr) < _HDR.size:
                return
            magic, seq, t0, t1, tr, n = _HDR.unpack(hdr)
            if magic != _MAGIC:
                raise ValueError(f"corrupt scans.bin at offset {f.tell() - _HDR.size}")
            body = f.read(12 * n)
            if len(body) < 12 * n:
                return  # truncated last record (power loss) — ignore
            arr = np.frombuffer(body, dtype=np.float32)
            yield Scan(seq, t0, t1, tr, arr[:n].copy(), arr[n:2 * n].copy(), arr[2 * n:].copy())


class RunReader:
    """Iterate a run's scans and events merged in time order.

    Scans are ordered by their *receive* time (when the live system would have
    seen them).  Yields ("scan", Scan) or ("ev", dict).
    """

    def __init__(self, run_dir: str):
        self.dir = run_dir
        with open(os.path.join(run_dir, "meta.json")) as f:
            self.meta = json.load(f)

    def events(self) -> list[dict]:
        out = []
        with open(os.path.join(self.dir, "events.jsonl")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # truncated last line
        return out

    def scans(self) -> Iterator[Scan]:
        return read_scans(os.path.join(self.dir, "scans.bin"))

    def merged(self) -> Iterator[tuple]:
        evs = self.events()
        evs.sort(key=lambda e: e["t"])
        i = 0
        for s in self.scans():
            while i < len(evs) and evs[i]["t"] <= s.t_recv:
                yield ("ev", evs[i])
                i += 1
            yield ("scan", s)
        while i < len(evs):
            yield ("ev", evs[i])
            i += 1

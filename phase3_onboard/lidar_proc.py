"""
LiDAR reader in its own process.

Why: gil_probe showed that the YDLidar SDK call holds Python's GIL while it waits for a
revolution, freezing every other thread for ~95 ms once per scan.  That bunches the IMU
lines (arrival-stamped late -> wrong heading prior / de-skew -> rotated walls in the map),
starves the IMU poll and motor writer (IMU arrived at 7.7 Hz instead of 25 Hz) and makes
the UI laggy.  In a separate process the SDK can only block its own interpreter.

LidarProcess is a drop-in replacement for lidar.LidarThread as main.Runtime uses it:
    start(), stop(), get(timeout) -> newest scan or None,
    rate_hz, last_n, count, dropped, error
The child runs the ORIGINAL LidarThread unchanged.  Every call it makes on the logger is
forwarded through the pipe and replayed on the real RunLogger in this process, so the
run log (and therefore tools.replay) gets exactly what it got before.

Timestamps stay valid: time.monotonic() is CLOCK_MONOTONIC, shared by all processes.

The child is forked in the constructor (before the link/SLAM threads exist) and waits
for start().  It exits on stop(), or by itself if this process disappears.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import threading
import time

log = logging.getLogger("lidar_proc")


class _LogProxy:
    """Stands in for RunLogger inside the child: forwards method calls through the pipe."""

    def __init__(self, send):
        self._send = send

    def __bool__(self):
        return True

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args, **kw):
            self._send(("log", (name, args, kw)))
        return call


def _child(P, conn, go, stop_ev, use_logger):
    # Ctrl-C / SSH hang-up reach the whole process group: let the parent decide.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    ppid = os.getppid()
    lock = threading.Lock()

    def send(msg):
        with lock:
            conn.send(msg)

    def parent_alive():
        return os.getppid() == ppid

    while not go.wait(0.5):
        if stop_ev.is_set() or not parent_alive():
            return
    if stop_ev.is_set():
        return

    lt = None
    try:
        from phase3_onboard.lidar import LidarThread, make_lidar
        lt = LidarThread(make_lidar(P, None), _LogProxy(send) if use_logger else None)
        lt.start()
        last_stats = 0.0
        while not stop_ev.is_set() and parent_alive():
            s = lt.get(timeout=0.2)
            if s is not None:
                send(("scan", s))
            now = time.monotonic()
            if now - last_stats > 0.5:
                last_stats = now
                send(("stats", {"rate_hz": getattr(lt, "rate_hz", 0.0),
                                "last_n": getattr(lt, "last_n", 0),
                                "dropped": getattr(lt, "dropped", 0),
                                "error": getattr(lt, "error", None)}))
    except (BrokenPipeError, EOFError, OSError):
        pass                                   # parent gone
    except BaseException as e:                 # report driver errors to the parent
        try:
            send(("stats", {"error": f"lidar process: {type(e).__name__}: {e}"}))
        except Exception:
            pass
    finally:
        if lt is not None:
            try:
                lt.stop()
            except Exception:
                pass
        try:
            send(("bye", None))
        except Exception:
            pass


class LidarProcess:
    def __init__(self, P, logger=None):
        ctx = mp.get_context("fork")           # P need not be picklable; Linux only
        self._conn, child_conn = ctx.Pipe(duplex=False)
        self._go = ctx.Event()
        self._stop_ev = ctx.Event()
        self.logger = logger
        self._proc = ctx.Process(target=_child, name="lidar-proc", daemon=True,
                                 args=(P, child_conn, self._go, self._stop_ev, logger is not None))
        self._proc.start()
        child_conn.close()
        self._cv = threading.Condition()
        self._latest = None
        self._stopping = False
        self._rx_thread = None
        self._log_err = False
        self.rate_hz = 0.0
        self.last_n = 0
        self.count = 0
        self.error = None
        self._child_dropped = 0
        self._local_dropped = 0

    @property
    def dropped(self):
        return self._child_dropped + self._local_dropped

    def start(self):
        self._go.set()
        self._rx_thread = threading.Thread(target=self._rx, daemon=True, name="lidar-rx")
        self._rx_thread.start()

    def _rx(self):
        while True:
            try:
                kind, a = self._conn.recv()     # blocks with the GIL released
            except (EOFError, OSError):
                if not self._stopping:
                    self.error = self.error or "lidar process exited"
                    log.error("lidar process exited unexpectedly")
                break
            if kind == "scan":
                with self._cv:
                    if self._latest is not None:
                        self._local_dropped += 1
                    self._latest = a
                    self.count += 1
                    n = getattr(a, "n", None)
                    if n is None:
                        for f in ("ranges", "r", "pts", "xy"):
                            v = getattr(a, f, None)
                            if v is not None:
                                n = len(v)
                                break
                    if n is not None:
                        self.last_n = int(n)
                    self._cv.notify_all()
            elif kind == "log":
                name, args, kw = a
                try:
                    getattr(self.logger, name)(*args, **kw)
                except Exception as e:
                    if not self._log_err:
                        self._log_err = True
                        log.error("forwarded logger call %s failed: %s", name, e)
            elif kind == "stats":
                if "rate_hz" in a:
                    self.rate_hz = a["rate_hz"]
                if a.get("last_n"):
                    self.last_n = a["last_n"]
                if "dropped" in a:
                    self._child_dropped = a["dropped"] or 0
                if "error" in a:
                    self.error = a["error"]
            elif kind == "bye":
                break

    def get(self, timeout=None):
        with self._cv:
            if self._latest is None:
                self._cv.wait(timeout)
            s, self._latest = self._latest, None
            return s

    def stop(self):
        self._stopping = True
        self._stop_ev.set()
        self._go.set()
        self._proc.join(3.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(1.0)
        if self._rx_thread is not None:
            self._rx_thread.join(1.0)

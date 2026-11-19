#!/usr/bin/env python3
"""
Rover desktop GUI (Tkinter) for `python -m phase3_onboard.main --headless`.
==========================================================================
Live (on your PC or on the Pi's own desktop):
    python phase3_onboard/gui.py --host <pi-address>            # port 8765 by default
Offline (no rover, e.g. after a replay):
    python phase3_onboard/gui.py --map runs/<run>/replay_v3/map.npz --traj runs/<run>/replay_v3/traj.csv
    python phase3_onboard/gui.py --map maps/map_<time>.npz

Needs only Python 3 with tkinter and numpy.  It runs as its own process, so drawing the map
never competes with the rover's SLAM / UART threads.

Controls
    W A S D / arrows   drive while held (re-sent at 10 Hz; released = stop)
    Space / Esc        STOP (also cancels a goal)          M  save map
    click on the map   go to that point (A* + pure pursuit)
    drag               pan              wheel  zoom             F  follow the rover on/off
Closing the window, losing WiFi, or this program freezing for more than 1 s stops the rover.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import socket
import struct
import threading
import time
import tkinter as tk
import zlib
from tkinter import ttk

import numpy as np

# ── colours (same palette as the old web UI) ───────────────────────────────────
BG, PANEL, LINE, FG, MUTE = "#0e1116", "#151a21", "#252c36", "#dfe5ee", "#7f8a99"
ACC, PATH, OK, WARN, BAD = "#35c4e8", "#ff5c8a", "#3ddc84", "#ffc53d", "#ff4d4f"
SCAN = "#ffb000"
LUT = np.array([[128, 128, 128], [250, 250, 250], [20, 20, 20]], np.uint8)   # unknown, free, occupied
SETTINGS = os.path.join(os.path.expanduser("~"), ".rover_gui.json")


# ═══════════════════════════════════════════════════════════════════════════════
#  Network link (background thread)
# ═══════════════════════════════════════════════════════════════════════════════
class Link:
    def __init__(self, inbox: queue.Queue):
        self.inbox = inbox
        self.sock = None
        self.wlock = threading.Lock()
        self.target = None
        self.connected = False
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="link").start()
        threading.Thread(target=self._pinger, daemon=True, name="ping").start()

    def connect(self, host, port):
        self.target = (host, int(port))
        self._drop()

    def _drop(self):
        s, self.sock = self.sock, None
        self.connected = False
        if s:
            try:
                s.close()
            except OSError:
                pass

    def close(self):
        self._stop.set()
        self._drop()

    def send(self, obj) -> bool:
        s = self.sock
        if not s:
            return False
        try:
            with self.wlock:
                s.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode())
            return True
        except OSError:
            self._drop()
            return False

    def _pinger(self):
        while not self._stop.is_set():
            if self.connected:
                self.send({"cmd": "ping", "t": time.monotonic()})
            time.sleep(0.2)

    @staticmethod
    def _recv_exact(s, n):
        b = bytearray()
        while len(b) < n:
            c = s.recv(n - len(b))
            if not c:
                raise ConnectionError("closed")
            b += c
        return bytes(b)

    def _run(self):
        while not self._stop.is_set():
            if not self.target:
                time.sleep(0.2)
                continue
            target = self.target
            try:
                s = socket.create_connection(target, timeout=3.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(3.0)
                self.sock = s
                self.connected = True
                self.inbox.put(("status", f"connected to {target[0]}:{target[1]}"))
                while not self._stop.is_set() and self.sock is s and self.target == target:
                    n = struct.unpack(">I", self._recv_exact(s, 4))[0]
                    body = self._recv_exact(s, n)
                    kind, payload = body[:1], body[1:]
                    if kind == b"J":
                        self.inbox.put(("json", json.loads(payload)))
                    elif kind == b"M":
                        hl = struct.unpack(">H", payload[:2])[0]
                        hdr = json.loads(payload[2:2 + hl])
                        cells = np.frombuffer(zlib.decompress(payload[2 + hl:]), np.uint8)
                        self.inbox.put(("map", (hdr, cells.reshape(hdr["h"], hdr["w"]) if hdr["h"] else cells)))
            except (OSError, ConnectionError, ValueError, struct.error) as e:
                if self.connected or self.sock is not None:
                    self.inbox.put(("status", f"link lost: {e}"))
                self._drop()
                time.sleep(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Application
# ═══════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self, args):
        super().__init__()
        self.title("Rover")
        self.configure(bg=BG)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1280, sw - 20)}x{min(820, sh - 70)}+0+0")
        self.minsize(900, 600)
        self.offline = bool(args.map)
        self.inbox: queue.Queue = queue.Queue()
        self.link = None if self.offline else Link(self.inbox)

        # model
        self.S = None                  # latest state message
        self.cells = None              # full n x n uint8 (row = cy, y up)
        self.meta = None               # {"n","res","orig"}
        self.trail = []
        self.rtt_ms = None
        self.status_msg = "offline viewer" if self.offline else "not connected"
        self.notes_local = []
        # view
        self.view = {"cx": 0.0, "cy": 0.0, "scale": 60.0}
        self.follow = not self.offline
        self.dirty_map = True
        self._img = None
        self._drag = None
        # input
        self.keys = set()
        self._release_jobs = {}
        self._was_driving = False

        self._style()
        self._build()
        self._bind_events()
        if self.offline:
            self._load_offline(args.map, args.traj)
        else:
            host = args.host or self._settings().get("host", "")
            self.host_var.set(host)
            self.port_var.set(str(args.port))
            if host:
                self._connect()
        self.after(30, self._tick)
        if self.offline:
            self.after(200, self._offline_notes)
        if not self.offline:
            self.after(100, self._drive_loop)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── settings ──────────────────────────────────────────────────────────────
    @staticmethod
    def _settings():
        try:
            with open(SETTINGS) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_settings(self, **kw):
        d = self._settings()
        d.update(kw)
        try:
            with open(SETTINGS, "w") as f:
                json.dump(d, f)
        except Exception:
            pass

    # ── widgets ───────────────────────────────────────────────────────────────
    def _style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=PANEL, foreground=FG, fieldbackground="#1d2430", bordercolor=LINE)
        st.configure("TFrame", background=PANEL)
        st.configure("TLabel", background=PANEL, foreground=FG)
        st.configure("Mute.TLabel", foreground=MUTE, font=("TkDefaultFont", 9))
        st.configure("H.TLabel", foreground=MUTE, font=("TkDefaultFont", 9, "bold"))
        st.configure("TButton", background="#1d2430", foreground=FG, padding=5)
        st.map("TButton", background=[("active", "#26303f")])
        st.configure("Stop.TButton", background=BAD, foreground="#ffffff", font=("TkDefaultFont", 14, "bold"), padding=10)
        st.map("Stop.TButton", background=[("active", "#ff7072")])
        st.configure("TEntry", foreground=FG, insertcolor=FG)
        st.configure("TSpinbox", foreground=FG, arrowcolor=FG, insertcolor=FG)

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.cv = tk.Canvas(self, bg="#808080", highlightthickness=0, cursor="crosshair")
        self.cv.grid(row=0, column=0, sticky="nsew")
        side = ttk.Frame(self, padding=(12, 8))
        side.grid(row=0, column=1, sticky="ns")
        self.side = side
        r = 0
        if not self.offline:
            cf = ttk.Frame(side)
            cf.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
            self.host_var, self.port_var = tk.StringVar(), tk.StringVar()
            ttk.Label(cf, text="Pi", style="Mute.TLabel").pack(side="left")
            self.host_entry = ttk.Entry(cf, textvariable=self.host_var, width=16)
            self.host_entry.pack(side="left", padx=4)
            ttk.Entry(cf, textvariable=self.port_var, width=6).pack(side="left")
            ttk.Button(cf, text="Connect", command=self._connect).pack(side="left", padx=(4, 0))
            self.stop_btn = ttk.Button(side, text="STOP  (Space)", style="Stop.TButton", command=self._stop)
            self.stop_btn.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(8, 4)); r += 1
            bf = ttk.Frame(side)
            bf.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
            for i, (t, c) in enumerate((("Save map", lambda: self._cmd("save")), ("Snapshot", lambda: self._cmd("snap")),
                                         ("Resume", lambda: self._cmd("resume")))):
                ttk.Button(bf, text=t, command=c).grid(row=0, column=i, sticky="ew", padx=1)
                bf.columnconfigure(i, weight=1)
        vf = ttk.Frame(side)
        vf.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(4, 0)); r += 1
        self.follow_btn = ttk.Button(vf, text="", command=self._toggle_follow)
        self.follow_btn.grid(row=0, column=0, sticky="ew", padx=1)
        ttk.Button(vf, text="Fit map", command=self._fit).grid(row=0, column=1, sticky="ew", padx=1)
        vf.columnconfigure(0, weight=1); vf.columnconfigure(1, weight=1)
        self._update_follow_btn()

        self.rows = {}

        def head(text):
            nonlocal r
            if self.offline:          # offline: no live panel
                return
            ttk.Label(side, text=text.upper(), style="H.TLabel").grid(row=r, column=0, columnspan=2, sticky="w", pady=(7, 1))
            r += 1

        def row(key, label):
            nonlocal r
            if self.offline:
                return
            ttk.Label(side, text=label, style="Mute.TLabel").grid(row=r, column=0, sticky="w")
            v = tk.Label(side, text="—", bg=PANEL, fg=FG, font=("TkFixedFont", 9), anchor="e", width=30)
            v.grid(row=r, column=1, sticky="e")
            self.rows[key] = v
            r += 1

        if not self.offline:
            head("Test motion (on the floor, full payload)")
            tf = ttk.Frame(side)
            tf.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
            self.duty_var, self.sec_var = tk.StringVar(value="0.20"), tk.StringVar(value="2.0")
            ttk.Label(tf, text="duty", style="Mute.TLabel").grid(row=0, column=0)
            ttk.Spinbox(tf, from_=0.05, to=0.45, increment=0.01, textvariable=self.duty_var, width=5).grid(row=0, column=1, padx=2)
            ttk.Label(tf, text="sec", style="Mute.TLabel").grid(row=0, column=2)
            ttk.Spinbox(tf, from_=0.5, to=5.0, increment=0.5, textvariable=self.sec_var, width=4).grid(row=0, column=3, padx=2)
            tb = ttk.Frame(side)
            tb.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(3, 0)); r += 1
            for i, (t, kind, sign) in enumerate((("⟲ spin L", "spin", 1), ("spin R ⟳", "spin", -1),
                                                  ("▲ fwd", "fwd", 1), ("▼ rev", "fwd", -1))):
                ttk.Button(tb, text=t, command=lambda k=kind, s=sign: self._test(k, s)).grid(row=0, column=i, sticky="ew", padx=1)
                tb.columnconfigure(i, weight=1)
            row("test_result", "last test")

        head("Rover")
        row("mode", "mode / phase")
        row("pose", "pose x, y, θ")
        row("goal", "goal")
        head("SLAM")
        row("rate", "rate / accepted")
        row("rej", "reject rate (30)")
        row("last", "last frame")
        row("inl", "inliers / rms")
        row("dth", "Δθ gyro / match")
        row("ms", "step ms (match)")
        row("gyro", "gyro bias / scale")
        head("Sensors & link")
        row("imu", "IMU gaps (5 s)")
        row("lidar", "LiDAR")
        row("link", "UART link")
        row("cmd", "cmd L / R")
        row("prox", "front / rear")
        row("net", "GUI link")
        row("sys", "Pi CPU / temp")
        # events: a strip under the map (keeps the side panel short enough for 768-px screens)
        self.notes = tk.Text(self, height=5, bg=PANEL, fg=MUTE, bd=0, font=("TkFixedFont", 9), wrap="none",
                             highlightthickness=0, padx=8, pady=4)
        self.notes.grid(row=1, column=0, sticky="ew")
        self.notes.configure(state="disabled")
        side.grid_configure(rowspan=2)

    def _bind_events(self):
        self.cv.bind("<Configure>", lambda e: self._mark_dirty())
        self.cv.bind("<ButtonPress-1>", self._press)
        self.cv.bind("<B1-Motion>", self._motion)
        self.cv.bind("<ButtonRelease-1>", self._release)
        self.cv.bind("<MouseWheel>", lambda e: self._zoom(1.15 if e.delta > 0 else 1 / 1.15, e.x, e.y))
        self.cv.bind("<Button-4>", lambda e: self._zoom(1.15, e.x, e.y))
        self.cv.bind("<Button-5>", lambda e: self._zoom(1 / 1.15, e.x, e.y))
        self.bind_all("<KeyPress>", self._key_down)
        self.bind_all("<KeyRelease>", self._key_up)
        self.bind("<FocusOut>", lambda e: self.keys.clear())

    # ── commands ──────────────────────────────────────────────────────────────
    def _connect(self):
        host = self.host_var.get().strip()
        try:
            port = int(self.port_var.get())
        except ValueError:
            port = 8765
        if not host:
            return
        self._save_settings(host=host)
        self.status_msg = f"connecting to {host}:{port} ..."
        self.link.connect(host, port)
        self.cv.focus_set()

    def _cmd(self, cmd, **kw):
        if self.link and not self.link.send({"cmd": cmd, **kw}):
            self._note(f"not connected: '{cmd}' not sent")

    def _stop(self):
        self.keys.clear()
        self._cmd("stop")

    def _test(self, kind, sign):
        try:
            duty, sec = float(self.duty_var.get()), float(self.sec_var.get())
        except ValueError:
            return self._note("test: bad duty / seconds")
        self._cmd(kind, duty=sign * duty, sec=sec)
        self.cv.focus_set()

    def _note(self, msg):
        self.notes_local.append(time.strftime("%H:%M:%S ") + msg)
        self.notes_local = self.notes_local[-5:]

    # ── keyboard (with X11 auto-repeat filtering) ─────────────────────────────
    KEYMAP = {"w": "w", "up": "w", "s": "s", "down": "s", "a": "a", "left": "a", "d": "d", "right": "d"}

    def _typing(self):
        w = self.focus_get()
        return isinstance(w, (tk.Entry, ttk.Entry, ttk.Spinbox, tk.Spinbox))

    def _key_down(self, e):
        k = e.keysym.lower()
        if k in ("space", "escape"):
            if k == "space" and self._typing():
                return
            return self._stop()
        if self._typing():
            return
        if k == "m" and not self.offline:
            return self._cmd("save")
        if k == "f":
            return self._toggle_follow()
        if k in self.KEYMAP and not self.offline:
            km = self.KEYMAP[k]
            job = self._release_jobs.pop(km, None)
            if job:
                self.after_cancel(job)
            self.keys.add(km)

    def _key_up(self, e):
        k = e.keysym.lower()
        if k in self.KEYMAP:
            km = self.KEYMAP[k]
            # X11 sends release+press pairs while a key is held: release only if no press follows
            self._release_jobs[km] = self.after(40, lambda: (self.keys.discard(km), self._release_jobs.pop(km, None)))

    def _drive_loop(self):
        fwd = (1 if "w" in self.keys else 0) - (1 if "s" in self.keys else 0)
        turn = (1 if "a" in self.keys else 0) - (1 if "d" in self.keys else 0)     # +turn = CCW (left)
        active = bool(fwd or turn)
        if active or self._was_driving:
            self._cmd("drive", fwd=fwd, turn=turn)
        self._was_driving = active
        self.after(100, self._drive_loop)

    # ── mouse / view ──────────────────────────────────────────────────────────
    def _w2s(self, x, y):
        W, H = self.cv.winfo_width(), self.cv.winfo_height()
        v = self.view
        return W / 2 + (x - v["cx"]) * v["scale"], H / 2 - (y - v["cy"]) * v["scale"]

    def _s2w(self, px, py):
        W, H = self.cv.winfo_width(), self.cv.winfo_height()
        v = self.view
        return v["cx"] + (px - W / 2) / v["scale"], v["cy"] - (py - H / 2) / v["scale"]

    def _press(self, e):
        self.cv.focus_set()
        self._drag = {"x": e.x, "y": e.y, "cx": self.view["cx"], "cy": self.view["cy"], "moved": False}

    def _motion(self, e):
        d = self._drag
        if not d:
            return
        dx, dy = e.x - d["x"], e.y - d["y"]
        if abs(dx) + abs(dy) > 5:
            d["moved"] = True
            if self.follow:
                self.follow = False
                self._update_follow_btn()
        if d["moved"]:
            self.view["cx"] = d["cx"] - dx / self.view["scale"]
            self.view["cy"] = d["cy"] + dy / self.view["scale"]
            self._mark_dirty()

    def _release(self, e):
        d, self._drag = self._drag, None
        if d and not d["moved"]:
            x, y = self._s2w(e.x, e.y)
            if self.offline:
                self._note(f"clicked ({x:+.2f}, {y:+.2f}) m")
            else:
                self._cmd("goal", x=round(x, 3), y=round(y, 3))

    def _zoom(self, f, px, py):
        x, y = self._s2w(px, py)
        v = self.view
        v["scale"] = min(400.0, max(8.0, v["scale"] * f))
        # keep the point under the cursor fixed
        W, H = self.cv.winfo_width(), self.cv.winfo_height()
        v["cx"] = x - (px - W / 2) / v["scale"]
        v["cy"] = y + (py - H / 2) / v["scale"]
        self._mark_dirty()

    def _toggle_follow(self):
        self.follow = not self.follow
        self._update_follow_btn()
        self._mark_dirty()

    def _update_follow_btn(self):
        self.follow_btn.configure(text="Follow: on" if self.follow else "Follow: off")

    def _fit(self):
        if self.cells is None or self.meta is None:
            return
        rows = np.flatnonzero(self.cells.any(axis=1))
        cols = np.flatnonzero(self.cells.any(axis=0))
        if not len(rows):
            return
        res, o = self.meta["res"], self.meta["orig"]
        x0, x1 = (cols[0] - o) * res, (cols[-1] - o) * res
        y0, y1 = (rows[0] - o) * res, (rows[-1] - o) * res
        W, H = max(self.cv.winfo_width(), 100), max(self.cv.winfo_height(), 100)
        self.view.update(cx=(x0 + x1) / 2, cy=(y0 + y1) / 2,
                         scale=max(8.0, min(400.0, 0.9 * min(W / max(x1 - x0, 0.5), H / max(y1 - y0, 0.5)))))
        self.follow = False
        self._update_follow_btn()
        self._mark_dirty()

    def _mark_dirty(self):
        self.dirty_map = True

    # ── data ──────────────────────────────────────────────────────────────────
    def _apply_map(self, hdr, crop):
        n = hdr["n"]
        if self.cells is None or self.cells.shape != (n, n):
            self.cells = np.zeros((n, n), np.uint8)
        self.cells[:] = 0
        if hdr["h"]:
            self.cells[hdr["r0"]:hdr["r0"] + hdr["h"], hdr["c0"]:hdr["c0"] + hdr["w"]] = crop
        self.meta = {"n": n, "res": hdr["res"], "orig": hdr["orig"]}
        self._mark_dirty()

    def _load_offline(self, map_path, traj_path):
        d = np.load(map_path, allow_pickle=False)
        lo = d["log"]
        cells = np.zeros(lo.shape, np.uint8)
        cells[lo < -0.4] = 1
        cells[lo > 0.6] = 2
        self.cells = cells
        self.meta = {"n": int(d["n"]), "res": float(d["res"]), "orig": int(d["orig"])}
        self.status_msg = f"offline: {map_path}"
        pose = d["pose"] if "pose" in d else None
        if traj_path:
            self.trail = _read_traj(traj_path)
        self.S = {"pose": [float(v) for v in pose] if pose is not None else (self.trail[-1] + [0.0] if self.trail else None)}
        self.after(50, self._fit)

    def _tick(self):
        try:
            while True:
                kind, obj = self.inbox.get_nowait()
                if kind == "json":
                    t = obj.get("type")
                    if t == "state":
                        self.S = obj
                        mm = obj.get("map_meta")
                        if mm and self.meta is None:
                            self.meta = mm
                    elif t == "trail":
                        self.trail = obj["pts"]
                    elif t == "pong" and obj.get("t") is not None:
                        rtt = (time.monotonic() - float(obj["t"])) * 1e3
                        self.rtt_ms = rtt if self.rtt_ms is None else 0.7 * self.rtt_ms + 0.3 * rtt
                    elif t == "note":
                        self._note(obj["msg"])
                elif kind == "map":
                    self._apply_map(*obj)
                elif kind == "status":
                    self.status_msg = obj
                    self._note(obj)
        except queue.Empty:
            pass
        self._follow_step()
        self._draw()
        if not self.offline:
            self._panel()
        self.after(40, self._tick)

    def _follow_step(self):
        if not (self.follow and self.S and self.S.get("pose")):
            return
        x, y = self.S["pose"][:2]
        W, H = self.cv.winfo_width(), self.cv.winfo_height()
        v = self.view
        # re-centre only when the rover leaves the middle half of the view: cheap on the Pi desktop
        if abs(x - v["cx"]) * v["scale"] > W / 4 or abs(y - v["cy"]) * v["scale"] > H / 4:
            v["cx"], v["cy"] = x, y
            self._mark_dirty()

    # ── drawing ───────────────────────────────────────────────────────────────
    def _render_map(self):
        W, H = self.cv.winfo_width(), self.cv.winfo_height()
        if W < 10 or H < 10:
            return
        self.cv.delete("map")
        if self.cells is not None and self.meta is not None:
            res, o, n = self.meta["res"], self.meta["orig"], self.meta["n"]
            v = self.view
            xs = v["cx"] + (np.arange(W) + 0.5 - W / 2) / v["scale"]
            ys = v["cy"] - (np.arange(H) + 0.5 - H / 2) / v["scale"]
            ci = np.rint(xs / res).astype(np.int64) + o
            ri = np.rint(ys / res).astype(np.int64) + o
            cok, rok = (ci >= 0) & (ci < n), (ri >= 0) & (ri < n)
            sub = np.zeros((H, W), np.uint8)
            sub[np.ix_(rok, cok)] = self.cells[np.ix_(ri[rok], ci[cok])]
            rgb = LUT[sub]
            ppm = b"P6 %d %d 255\n" % (W, H) + rgb.tobytes()
            self._img = tk.PhotoImage(data=ppm, format="PPM")
            self.cv.create_image(0, 0, image=self._img, anchor="nw", tags="map")
        # 1 m grid
        x0, y1 = self._s2w(0, 0)
        x1, y0 = self._s2w(W, H)
        if self.view["scale"] >= 15:
            for gx in range(math.ceil(x0), math.floor(x1) + 1):
                sx, _ = self._w2s(gx, 0)
                self.cv.create_line(sx, 0, sx, H, fill="#6f8796", stipple="gray25", tags="map")
            for gy in range(math.ceil(y0), math.floor(y1) + 1):
                _, sy = self._w2s(0, gy)
                self.cv.create_line(0, sy, W, sy, fill="#6f8796", stipple="gray25", tags="map")
        self.cv.tag_lower("map")
        self.dirty_map = False

    def _draw(self):
        if self.dirty_map:
            self._render_map()
        cv = self.cv
        cv.delete("ov")
        W, H = cv.winfo_width(), cv.winfo_height()
        S = self.S or {}
        # trail
        if len(self.trail) > 1:
            pts = []
            for p in self.trail:
                pts.extend(self._w2s(p[0], p[1]))
            cv.create_line(*pts, fill=PATH, width=2, tags="ov")
        # planned path
        if S.get("path") and S.get("pose"):
            pts = list(self._w2s(*S["pose"][:2]))
            for p in S["path"]:
                pts.extend(self._w2s(p[0], p[1]))
            if len(pts) >= 4:
                cv.create_line(*pts, fill=ACC, width=2, dash=(6, 4), tags="ov")
        # scan (yellow = live scan placed at the ESTIMATED pose; should sit on the black walls)
        for p in S.get("scan", ()):
            sx, sy = self._w2s(p[0], p[1])
            cv.create_rectangle(sx - 1.5, sy - 1.5, sx + 1.5, sy + 1.5, fill=SCAN, outline="", tags="ov")
        if S.get("goal"):
            gx, gy = self._w2s(*S["goal"])
            cv.create_line(gx - 8, gy - 8, gx + 8, gy + 8, fill=PATH, width=3, tags="ov")
            cv.create_line(gx + 8, gy - 8, gx - 8, gy + 8, fill=PATH, width=3, tags="ov")
        if S.get("sim_truth"):
            tx, ty = self._w2s(*S["sim_truth"][:2])
            cv.create_oval(tx - 9, ty - 9, tx + 9, ty + 9, outline="white", tags="ov")
        if S.get("pose"):
            x, y, th = S["pose"]
            sc = self.view["scale"]
            L, Wd = max(12, 0.26 * sc), max(9, 0.19 * sc)
            c, s = math.cos(th), math.sin(th)
            corners = [(L / 2, Wd / 2), (L / 2, -Wd / 2), (-L / 2, -Wd / 2), (-L / 2, Wd / 2)]
            rx, ry = self._w2s(x, y)
            poly = []
            for a, b in corners:     # robot frame (m in px) -> screen (y down)
                poly += [rx + a * c - b * s, ry - (a * s + b * c)]
            cv.create_polygon(*poly, fill=BAD if S.get("lost") else ACC, outline=BG, tags="ov")
            nx, ny = rx + (L / 2) * c, ry - (L / 2) * s
            cv.create_line(rx, ry, nx, ny, fill=BG, width=3, arrow="last", tags="ov")
        # scale bar + banners
        sc = self.view["scale"]
        cv.create_rectangle(12, H - 22, 12 + sc, H - 16, fill=FG, outline="", tags="ov")
        cv.create_text(16 + sc, H - 19, text="1 m", anchor="w", fill=FG, tags="ov")
        cv.create_text(W - 10, H - 10, anchor="se", fill="#e8e8e8", font=("TkDefaultFont", 9), tags="ov",
                       text=("drag = pan · wheel = zoom · F follow" if self.offline else
                             "WASD/arrows drive · click = goal · drag = pan · wheel = zoom · Space/Esc STOP · M save · F follow"))
        hud = self.status_msg
        if self.S and not self.offline and S.get("pose"):
            p = S["pose"]
            sl = S.get("slam", {})
            hud = (f"{S.get('mode', '')}  x {p[0]:+.2f}  y {p[1]:+.2f}  θ {math.degrees(p[2]):+.1f}°   "
                   f"SLAM {sl.get('hz', 0)} Hz   reject {sl.get('reject_rate', 0) * 100:.0f}%")
        cv.create_text(12, 12, text=hud, anchor="nw", fill=FG, font=("TkFixedFont", 10), tags="ov")
        banner = None
        if not self.offline and not (self.link and self.link.connected):
            banner = "NOT CONNECTED"
        elif S.get("lost"):
            banner = "SLAM LOST — rover stopped. Drive back slowly, then Resume"
        if banner:
            cv.create_rectangle(W / 2 - 230, 8, W / 2 + 230, 38, fill=BAD, outline="", tags="ov")
            cv.create_text(W / 2, 23, text=banner, fill="white", font=("TkDefaultFont", 11, "bold"), tags="ov")

    def _offline_notes(self):
        self.notes.configure(state="normal")
        self.notes.delete("1.0", "end")
        self.notes.insert("end", "\n".join(reversed(self.notes_local[-5:] + [self.status_msg])))
        self.notes.configure(state="disabled")
        self.after(300, self._offline_notes)

    def _set(self, key, text, color=FG):
        w = self.rows.get(key)
        if w is not None:
            w.configure(text=text, fg=color)

    def _panel(self):
        S = self.S
        conn = self.link and self.link.connected
        self._set("net", (f"connected · rtt {self.rtt_ms:.0f} ms" if self.rtt_ms is not None else "connected")
                  if conn else "disconnected", OK if conn and (self.rtt_ms or 0) < 150 else (WARN if conn else BAD))
        notes = (S.get("notes", []) if S else []) + self.notes_local
        self.notes.configure(state="normal")
        self.notes.delete("1.0", "end")
        self.notes.insert("end", "\n".join(reversed(notes[-12:])))
        self.notes.configure(state="disabled")
        if not S:
            return
        f2 = lambda v: "—" if v is None else f"{v:.2f}"
        p = S["pose"]
        self._set("mode", f"{S['mode']} · {S['phase']}" + (" · TEST" if S.get("test") else ""),
                  ACC if S["mode"] == "AUTO" else OK)
        self._set("pose", f"{p[0]:+.2f}, {p[1]:+.2f}, {math.degrees(p[2]):+.1f}°")
        self._set("goal", "—" if not S.get("goal") else f"{S['goal'][0]:+.2f}, {S['goal'][1]:+.2f}")
        sl = S["slam"]
        last = sl.get("last") or {}
        self._set("rate", f"{sl['hz']} Hz · {sl['accepted']}/{sl['frames']}")
        rr = sl["reject_rate"]
        self._set("rej", f"{rr * 100:.0f} %", BAD if rr > 0.3 else WARN if rr > 0.1 else OK)
        self._set("last", last.get("why", "—"), OK if last.get("acc") else WARN)
        if last.get("inl") is not None:
            self._set("inl", f"{last['inl']} ({last['ir'] * 100:.0f}%) / {last['rms'] * 100:.1f} cm")
        d = lambda v: "—" if v is None else f"{math.degrees(v):+.2f}°"
        self._set("dth", f"{d(last.get('gdth'))} / {d(last.get('mdth'))}")
        ms = last.get("ms") or {}
        self._set("ms", f"{f2(ms.get('total'))} ({f2(ms.get('match'))})", WARN if (ms.get("total") or 0) > 60 else FG)
        self._set("gyro", ("" if sl["gyro_ready"] else "estimating… ") + f"{sl['gyro_bias_dps']:+.3f}°/s ×{sl['gyro_scale_corr']}",
                  FG if sl["gyro_ready"] and abs(sl["gyro_scale_corr"] - 1) < 0.1 else WARN)
        im = S.get("imu") or {}
        if im.get("max_gap_ms") is not None:
            bad = im["max_gap_ms"] > 80 or (im.get("bunched_pct") or 0) > 10
            self._set("imu", f"{im['hz']} Hz · max {im['max_gap_ms']:.0f} ms · bunched {im['bunched_pct']:.0f}%",
                      BAD if bad else OK)
        else:
            self._set("imu", "no T:1002 lines", BAD)
        li = S["lidar"]
        self._set("lidar", f"{li['rate_hz']} Hz · {li['points']} pts · drop {li['dropped']}" + (" · ERR" if li.get("error") else ""),
                  OK if li["rate_hz"] > 4 else BAD)
        lk = S["link"]
        age = lk.get("rx_age_s")
        self._set("link", f"rx age {age if age is not None else '—'} s · wd {lk['watchdog_trips']} · err {lk['tx_errors']}",
                  OK if age is not None and age < 0.5 else BAD)
        self._set("cmd", f"{lk['last_cmd'][0]:+.2f} / {lk['last_cmd'][1]:+.2f} ({lk.get('intent_src', '')})")
        pr = S.get("prox") or {}
        self._set("prox", f"{f2(pr.get('front'))} / {f2(pr.get('rear'))} m",
                  WARN if pr.get("front") is not None and pr["front"] < 0.3 else FG)
        sy = S.get("sys") or {}
        thr = sy.get("throttled")
        self._set("sys", f"{sy.get('cpu', '—')} % · {sy.get('temp', '—')} °C · thr {thr if thr is not None else 'n/a'}",
                  BAD if (thr and thr != "0x0") or (sy.get("temp") or 0) > 75 else FG)
        tests = [n for n in S.get("notes", []) if "[test " in n]
        if tests:
            self._set("test_result", tests[-1].split("] ", 1)[-1][:60] if "] " in tests[-1] else tests[-1][-60:])

    def _on_close(self):
        if self.link:
            self.link.send({"cmd": "stop"})
            time.sleep(0.05)
            self.link.close()
        self.destroy()


def _read_traj(path):
    """traj.csv from tools/replay: find the x / y columns by header name, else use columns 1, 2."""
    out = []
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return out
    head = [h.strip().lower() for h in rows[0]]
    try:
        float(rows[0][0])
        ix, iy, start = 1, 2, 0
    except ValueError:
        ix = head.index("x") if "x" in head else 1
        iy = head.index("y") if "y" in head else 2
        start = 1
    for r in rows[start:]:
        try:
            out.append([float(r[ix]), float(r[iy])])
        except (ValueError, IndexError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", help="Pi address (live mode)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--map", help="offline: a saved map .npz (maps/... or runs/<run>/replay_v3/map.npz)")
    ap.add_argument("--traj", help="offline: replay trajectory CSV to overlay")
    a = ap.parse_args()
    App(a).mainloop()


if __name__ == "__main__":
    main()

"""
Minimal web UI server (stdlib only — nothing extra to install on the Pi).
=========================================================================
GET  /                 UI page
GET  /api/state        pose, scan, trail, path, gate stats, link/lidar/sys status (JSON)
GET  /api/map.png      occupancy map PNG (re-encoded at most WEB_MAP_HZ, cached by version)
POST /api/drive        {"fwd": -1..1, "turn": -1..1}   drive intent, expires after INTENT_TTL_S
POST /api/goal         {"x": m, "y": m}
POST /api/stop | /api/save | /api/resume

Transport choice: short polling (state ~8 Hz, map ~3 Hz) instead of WebSocket.
It needs no extra dependency, and safety does not depend on it: the browser
re-sends drive intent at 10 Hz and the Pi treats silence as STOP.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from phase3_onboard.imgutil import map_rgb, png_bytes

log = logging.getLogger("web")
STATIC = os.path.join(os.path.dirname(__file__), "static")


class _MapCache:
    def __init__(self, rt):
        self.rt = rt
        self.version = -1
        self.png = b""
        self.t = 0.0
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            v = self.rt.eng.grid.version
            now = time.monotonic()
            if v != self.version and (now - self.t) >= 1.0 / self.rt.P.WEB_MAP_HZ:
                log_odds = self.rt.eng.map_copy()
                P = self.rt.P
                self.png = png_bytes(map_rgb(log_odds, P.MAP_OCC_THRESH, P.MAP_FREE_THRESH), level=1)
                self.version, self.t = v, now
            return self.png, self.version


def make_handler(rt, cache):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, separators=(",", ":"), default=str).encode())

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                with open(os.path.join(STATIC, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(rt.state())
            elif path == "/api/map.png":
                png, v = cache.get()
                self._send(200, png, "image/png", {"X-Map-Version": str(v)})
            else:
                self._send(404, b"{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except json.JSONDecodeError:
                return self._json({"ok": False, "err": "bad json"}, 400)
            path = self.path.split("?", 1)[0]
            try:
                if path == "/api/drive":
                    rt.drive(body.get("fwd", 0.0), body.get("turn", 0.0))
                    return self._json({"ok": True})
                if path == "/api/goal":
                    return self._json({"ok": rt.set_goal(float(body["x"]), float(body["y"]))})
                if path == "/api/stop":
                    rt.stop()
                    return self._json({"ok": True})
                if path == "/api/save":
                    return self._json({"ok": True, "path": rt.save_map()})
                if path == "/api/resume":
                    rt.clear_lost()
                    return self._json({"ok": True})
            except Exception as e:
                log.exception("POST %s", path)
                return self._json({"ok": False, "err": str(e)}, 500)
            self._json({"ok": False, "err": "unknown"}, 404)

    return H


class WebServer:
    def __init__(self, rt, host="0.0.0.0", port=8080):
        self.rt = rt
        self.httpd = ThreadingHTTPServer((host, port), make_handler(rt, _MapCache(rt)))
        self.httpd.daemon_threads = True

    def start(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True, name="web").start()

    def stop(self):
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:
            pass

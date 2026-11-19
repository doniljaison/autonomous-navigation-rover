"""
Occupancy grid (float32 log-odds) with numba raycasting and a local distance field.

Cell convention is the same as Phase 2 so old ``.npz`` maps still load:
    cx = round(x / res) + orig,  cy = round(y / res) + orig,  log[cy, cx]
"""

from __future__ import annotations


import numpy as np

from phase3_onboard.slam.jit import njit


@njit
def _raycast(log, n, rx, ry, ex, ey, hit, miss, clamp, do_hit):
    """Bresenham from (rx,ry) to each (ex,ey) in cell coords.  Free updates first
    for all rays, then hits, so a ray never erases this scan's own endpoints."""
    m = ex.shape[0]
    for k in range(m):
        x0 = rx
        y0 = ry
        x1 = ex[k]
        y1 = ey[k]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while x0 != x1 or y0 != y1:
            if 0 <= x0 < n and 0 <= y0 < n:
                v = log[y0, x0] + miss
                log[y0, x0] = v if v > -clamp else -clamp
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
    for k in range(m):
        if do_hit[k]:
            x1 = ex[k]
            y1 = ey[k]
            if 0 <= x1 < n and 0 <= y1 < n:
                v = log[y1, x1] + hit
                log[y1, x1] = v if v < clamp else clamp


@njit
def _chamfer(occ, cap):
    """Two-pass 8-neighbour chamfer distance (in cells) from occupied cells, capped."""
    h, w = occ.shape
    d = np.empty((h, w), dtype=np.float32)
    big = np.float32(cap)
    for y in range(h):
        for x in range(w):
            d[y, x] = 0.0 if occ[y, x] else big
    r2 = np.float32(1.41421356)
    for y in range(h):
        for x in range(w):
            v = d[y, x]
            if v == 0.0:
                continue
            if x > 0 and d[y, x - 1] + 1.0 < v:
                v = d[y, x - 1] + 1.0
            if y > 0:
                if d[y - 1, x] + 1.0 < v:
                    v = d[y - 1, x] + 1.0
                if x > 0 and d[y - 1, x - 1] + r2 < v:
                    v = d[y - 1, x - 1] + r2
                if x < w - 1 and d[y - 1, x + 1] + r2 < v:
                    v = d[y - 1, x + 1] + r2
            d[y, x] = v
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            v = d[y, x]
            if v == 0.0:
                continue
            if x < w - 1 and d[y, x + 1] + 1.0 < v:
                v = d[y, x + 1] + 1.0
            if y < h - 1:
                if d[y + 1, x] + 1.0 < v:
                    v = d[y + 1, x] + 1.0
                if x < w - 1 and d[y + 1, x + 1] + r2 < v:
                    v = d[y + 1, x + 1] + r2
                if x > 0 and d[y + 1, x - 1] + r2 < v:
                    v = d[y + 1, x - 1] + r2
            d[y, x] = v
    return d


class OccupancyGrid:
    def __init__(self, size_m: float = 20.0, res_m: float = 0.05,
                 hit: float = 0.9, miss: float = -0.4, clamp: float = 3.5):
        self.res = float(res_m)
        self.n = int(round(size_m / res_m))
        self.orig = self.n // 2
        self.log = np.zeros((self.n, self.n), dtype=np.float32)
        self.hit, self.miss, self.clamp = float(hit), float(miss), float(clamp)
        self.version = 0
        self.updates = 0

    # -- coordinates ---------------------------------------------------------------
    def w2c(self, x, y):
        return (np.rint(np.asarray(x) / self.res).astype(np.int64) + self.orig,
                np.rint(np.asarray(y) / self.res).astype(np.int64) + self.orig)

    def c2w(self, cx, cy):
        return (np.asarray(cx) - self.orig) * self.res, (np.asarray(cy) - self.orig) * self.res

    # -- update --------------------------------------------------------------------
    def update(self, origin_xy, hits_xy: np.ndarray, free_xy: np.ndarray | None = None):
        """origin_xy: sensor position in world.  hits_xy: endpoints that are
        obstacles.  free_xy: endpoints of free-only rays (no hit)."""
        rx, ry = self.w2c(origin_xy[0], origin_xy[1])
        parts_x, parts_y, parts_h = [], [], []
        if hits_xy is not None and len(hits_xy):
            cx, cy = self.w2c(hits_xy[:, 0], hits_xy[:, 1])
            parts_x.append(cx); parts_y.append(cy); parts_h.append(np.ones(len(cx), np.bool_))
        if free_xy is not None and len(free_xy):
            cx, cy = self.w2c(free_xy[:, 0], free_xy[:, 1])
            parts_x.append(cx); parts_y.append(cy); parts_h.append(np.zeros(len(cx), np.bool_))
        if not parts_x:
            return
        ex = np.concatenate(parts_x).astype(np.int64)
        ey = np.concatenate(parts_y).astype(np.int64)
        dh = np.concatenate(parts_h)
        _raycast(self.log, self.n, int(rx), int(ry), ex, ey,
                 np.float32(self.hit), np.float32(self.miss), np.float32(self.clamp), dh)
        self.version += 1
        self.updates += 1

    # -- distance field ------------------------------------------------------------
    def distance_field(self, cx_center: int, cy_center: int, half_cells: int,
                       occ_thresh: float, cap_m: float):
        """Distance (metres) to the nearest occupied cell for a square window.
        Returns (D, x0, y0): D[j, i] is cell (x0+i, y0+j)."""
        x0 = max(0, cx_center - half_cells)
        y0 = max(0, cy_center - half_cells)
        x1 = min(self.n, cx_center + half_cells + 1)
        y1 = min(self.n, cy_center + half_cells + 1)
        occ = self.log[y0:y1, x0:x1] > occ_thresh
        cap_cells = cap_m / self.res
        n_occ = int(occ.sum())
        d = np.full(occ.shape, np.float32(cap_m), dtype=np.float32)
        if n_occ == 0:
            return d, x0, y0, 0
        # Only the occupied bounding box (+ the cap distance) can be below the cap,
        # so run the transform there: ~10x less work in a room-sized map.
        rows = np.flatnonzero(occ.any(axis=1))
        cols = np.flatnonzero(occ.any(axis=0))
        m = int(np.ceil(cap_cells)) + 1
        r0, r1 = max(0, rows[0] - m), min(occ.shape[0], rows[-1] + m + 1)
        c0, c1 = max(0, cols[0] - m), min(occ.shape[1], cols[-1] + m + 1)
        sub = _chamfer(np.ascontiguousarray(occ[r0:r1, c0:c1]), np.float32(cap_cells))
        sub *= np.float32(self.res)
        np.minimum(sub, np.float32(cap_m), out=sub)
        d[r0:r1, c0:c1] = sub
        return d, x0, y0, n_occ

    def occupied_count(self, thresh: float) -> int:
        return int((self.log > thresh).sum())

    # -- export ----------------------------------------------------------------------
    def to_image(self, occ: float = 0.6, free: float = -0.4) -> np.ndarray:
        img = np.full((self.n, self.n), 128, dtype=np.uint8)
        img[self.log > occ] = 0
        img[self.log < free] = 255
        return img

    def save(self, path: str, **extra):
        np.savez_compressed(path, log=self.log, res=self.res, n=self.n, orig=self.orig,
                            hit=self.hit, miss=self.miss, clamp=self.clamp, **extra)

    @classmethod
    def load(cls, path: str) -> "OccupancyGrid":
        d = np.load(path, allow_pickle=False)
        g = cls.__new__(cls)
        g.log = d["log"].astype(np.float32)
        g.res = float(d["res"])
        g.n = int(d["n"])
        g.orig = int(d["orig"])
        g.hit = float(d["hit"]) if "hit" in d else 0.9
        g.miss = float(d["miss"]) if "miss" in d else -0.4
        g.clamp = float(d["clamp"]) if "clamp" in d else 3.5
        # Phase-2 maps were clamped at +/-5; bring them into this range
        np.clip(g.log, -g.clamp, g.clamp, out=g.log)
        g.version = 1
        g.updates = 0
        return g


def bresenham_cells(x0, y0, x1, y1):
    """Pure-Python reference used by tests."""
    out = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while (x0, y0) != (x1, y1):
        out.append((x0, y0))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return out


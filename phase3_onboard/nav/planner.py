"""
A* on a down-sampled, inflated occupancy grid (8-connected), plus line-of-sight
path shortening.  Unknown cells are allowed but cost more than known-free ones.
"""

from __future__ import annotations

import heapq
import logging
import math
from typing import Optional

import numpy as np

log = logging.getLogger("plan")


def _disk(r: int):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def build_cost_grid(log_odds: np.ndarray, res: float, plan_res: float, occ_th: float,
                    free_th: float, inflate_m: float):
    """Returns (blocked[bool], unknown[bool], factor).  Plan cell (i,j) covers map
    cells [j*f:(j+1)*f, i*f:(i+1)*f]."""
    from scipy.ndimage import binary_dilation
    f = max(1, int(round(plan_res / res)))
    n = log_odds.shape[0] // f
    L = log_odds[:n * f, :n * f].reshape(n, f, n, f)
    occ = (L > occ_th).any(axis=(1, 3))
    known_free = (L < free_th).all(axis=(1, 3))
    r = int(math.ceil(inflate_m / (res * f)))
    blocked = binary_dilation(occ, structure=_disk(r)) if r > 0 else occ
    return blocked, ~known_free & ~occ, f


def astar(blocked: np.ndarray, unknown: np.ndarray, start, goal, unknown_cost=1.5,
          max_expand=60000) -> Optional[list]:
    n = blocked.shape[0]
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < n and 0 <= sy < n and 0 <= gx < n and 0 <= gy < n):
        return None
    g = np.full(blocked.shape, np.inf)
    parent = -np.ones(blocked.shape + (2,), dtype=np.int32)
    g[sy, sx] = 0.0
    openl = [(math.hypot(gx - sx, gy - sy), 0.0, sx, sy)]
    SQ2 = math.sqrt(2.0)
    nb = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2))
    expanded = 0
    while openl:
        _, gc, x, y = heapq.heappop(openl)
        if gc > g[y, x]:
            continue
        if x == gx and y == gy:
            path = [(x, y)]
            while (x, y) != (sx, sy):
                x, y = parent[y, x]
                path.append((int(x), int(y)))
            return path[::-1]
        expanded += 1
        if expanded > max_expand:
            return None
        for dx, dy, c in nb:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < n and 0 <= ny < n) or blocked[ny, nx]:
                continue
            if dx and dy and (blocked[y, nx] or blocked[ny, x]):   # no corner cutting
                continue
            step = c * (unknown_cost if unknown[ny, nx] else 1.0)
            ng = gc + step
            if ng < g[ny, nx]:
                g[ny, nx] = ng
                parent[ny, nx] = (x, y)
                heapq.heappush(openl, (ng + math.hypot(gx - nx, gy - ny), ng, nx, ny))
    return None


def _los(blocked, a, b):
    x0, y0 = a
    x1, y1 = b
    n = int(max(abs(x1 - x0), abs(y1 - y0))) * 2 + 1
    for k in range(n + 1):
        t = k / n
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        if blocked[y, x]:
            return False
    return True


def shorten(blocked, cells):
    if len(cells) <= 2:
        return cells
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _los(blocked, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def _nearest_free(blocked, x, y, rmax):
    n = blocked.shape[0]
    for r in range(1, rmax + 1):
        best = None
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and not blocked[ny, nx]:
                    d = dx * dx + dy * dy
                    if best is None or d < best[0]:
                        best = (d, nx, ny)
        if best:
            return best[1], best[2]
    return None


class Planner:
    def __init__(self, P):
        self.P = P

    def plan(self, log_odds: np.ndarray, res: float, orig: int, start_xy, goal_xy) -> Optional[list]:
        P = self.P
        for inflate in (P.INFLATION_RADIUS_M, P.INFLATION_RADIUS_M * 0.6, 0.0):
            blocked, unknown, f = build_cost_grid(log_odds, res, P.PLAN_RES_M, P.MAP_OCC_THRESH,
                                                  P.MAP_FREE_THRESH, inflate)
            pr = res * f
            po = orig / f

            def w2p(x, y):
                return int(math.floor(x / pr + po + 0.5 / f)), int(math.floor(y / pr + po + 0.5 / f))

            s = w2p(*start_xy)
            g = w2p(*goal_xy)
            n = blocked.shape[0]
            if not (0 <= s[0] < n and 0 <= s[1] < n and 0 <= g[0] < n and 0 <= g[1] < n):
                return None
            if blocked[s[1], s[0]]:
                s2 = _nearest_free(blocked, s[0], s[1], 4)
                if s2 is None:
                    continue
                s = s2
            if blocked[g[1], g[0]]:
                g2 = _nearest_free(blocked, g[0], g[1], 5)
                if g2 is None:
                    continue
                g = g2
            cells = astar(blocked, unknown, s, g)
            if cells is None:
                continue
            cells = shorten(blocked, cells)
            pts = [((cx + 0.5) * pr - orig * res - 0.5 * res, (cy + 0.5) * pr - orig * res - 0.5 * res)
                   for cx, cy in cells]
            pts[-1] = tuple(goal_xy) if not blocked[g[1], g[0]] and g == w2p(*goal_xy) else pts[-1]
            if inflate < P.INFLATION_RADIUS_M:
                log.info("planned with reduced inflation %.2f m", inflate)
            return pts
        return None

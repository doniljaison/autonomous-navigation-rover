"""
Clearance-aware planner (drop-in for Planner.plan, installed by nav/guard.install_guard).
=======================================================================================
Why: the stock planner down-samples the grid and, when it fails, retries with inflation
0.6x and then 0, so paths could graze or cross single-cell obstacles (chair legs).

How:
  obstacles  = cells with log-odds > PLAN_OBST_THRESH (default half of MAP_OCC_THRESH, so
               weak "grey" leg cells count) + the remembered points from ObstacleLayer
  clearance  = distance (m) from every cell to the nearest obstacle (8-neighbour chamfer)
  hard limit = clearance < R_HARD = ROBOT_HALF_WIDTH_M + PLAN_MARGIN_M is never entered
               (the rover drives along the path, so its half width is what must fit)
  soft cost  = between R_HARD and R_HARD + PLAN_SOFT_M a cell costs extra, rising toward
               R_HARD: the path keeps away from things when there is room, but can still
               thread a gap (between chair legs) when that is the only way
  unknown    = allowed, PLAN_UNKNOWN_COST x more expensive
  A* (8-connected, numba) then shortcut smoothing: a straight shortcut is taken only if it
  touches no blocked cell AND costs no more than the A* stretch it replaces, so smoothing
  never drags the path toward obstacles; then resampled every 0.10 m.
Start inside the hard band (rover parked near a leg): the band (not the obstacle cells)
within PLAN_ESCAPE_M of the start may be crossed at a high cost.  Goal inside the band:
the nearest reachable cell within PLAN_GOAL_SNAP_M is used.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from numba import njit

DEFAULTS = dict(
    PLAN_MARGIN_M=0.04,        # hard side clearance beyond the half width (only tight gaps get this close)
    PLAN_SOFT_M=0.35,          # band beyond that in which cells cost extra: paths keep away when there is room
    PLAN_SOFT_WEIGHT=3.0,
    PLAN_ENCLOSED_WEIGHT=4.0,  # extra cost for enclosed cells (pockets, between chair legs): boxed-in risk
    PLAN_ENCLOSED_RANGE_M=0.9, # a direction is "closed" if an obstacle is within this distance
    PLAN_ENCLOSED_FREE_DIRS=4, # up to this many closed directions (of 8) cost nothing (a corridor has ~2-4)
    PLAN_TURN_ROOM=False,      # move path corners until the rover can spin there (only for FOLLOW_MODE skid)
    PLAN_ROUND=True,           # round the corners into smooth curves (only where clearance allows)
    PLAN_ROUND_CUT_M=0.45,     # how far before / after a corner the curve may start
    PLAN_UNKNOWN_COST=3.0,
    PLAN_ESCAPE_M=0.35,
    PLAN_GOAL_SNAP_M=0.30,
    PLAN_CELL_M=0.05,
    PLAN_PATH_STEP_M=0.10,
)


@njit(cache=True, nogil=True)
def _chamfer(obst, c):
    h, w = obst.shape
    d = np.empty((h, w))
    big = 1e9
    for i in range(h):
        for j in range(w):
            d[i, j] = 0.0 if obst[i, j] else big
    c2 = c * math.sqrt(2.0)
    for i in range(h):
        for j in range(w):
            v = d[i, j]
            if i > 0:
                v = min(v, d[i - 1, j] + c)
                if j > 0:
                    v = min(v, d[i - 1, j - 1] + c2)
                if j < w - 1:
                    v = min(v, d[i - 1, j + 1] + c2)
            if j > 0:
                v = min(v, d[i, j - 1] + c)
            d[i, j] = v
    for i in range(h - 1, -1, -1):
        for j in range(w - 1, -1, -1):
            v = d[i, j]
            if i < h - 1:
                v = min(v, d[i + 1, j] + c)
                if j > 0:
                    v = min(v, d[i + 1, j - 1] + c2)
                if j < w - 1:
                    v = min(v, d[i + 1, j + 1] + c2)
            if j < w - 1:
                v = min(v, d[i, j + 1] + c)
            d[i, j] = v
    return d


@njit(cache=True, nogil=True)
def _closed_dirs(obst, rng_cells):
    """For every cell: how many of 8 directions hit an obstacle within rng_cells."""
    h, w = obst.shape
    out = np.zeros((h, w), np.int8)
    di = (-1, -1, -1, 0, 0, 1, 1, 1)
    dj = (-1, 0, 1, -1, 1, -1, 0, 1)
    for i in range(h):
        for j in range(w):
            if obst[i, j]:
                out[i, j] = 8
                continue
            n = 0
            for k in range(8):
                step = 1.41421356 if di[k] != 0 and dj[k] != 0 else 1.0
                m = int(rng_cells / step)
                for t in range(1, m + 1):
                    a, b = i + di[k] * t, j + dj[k] * t
                    if a < 0 or b < 0 or a >= h or b >= w:
                        break
                    if obst[a, b]:
                        n += 1
                        break
            out[i, j] = n
    return out


@njit(cache=True, nogil=True)
def _astar(cost, blocked, si, sj, gi, gj):
    h, w = cost.shape
    n = h * w
    g = np.full(n, np.inf)
    par = np.full(n, -1, np.int64)
    closed = np.zeros(n, np.bool_)
    s = si * w + sj
    goal = gi * w + gj
    g[s] = 0.0
    heap = [(math.hypot(gi - si, gj - sj), s)]
    di = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dj = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    dl = np.array([1.41421356, 1.0, 1.41421356, 1.0, 1.0, 1.41421356, 1.0, 1.41421356])
    while len(heap) > 0:
        f, u = heapq.heappop(heap)
        if closed[u]:
            continue
        closed[u] = True
        if u == goal:
            break
        ui, uj = u // w, u % w
        for k in range(8):
            vi, vj = ui + di[k], uj + dj[k]
            if vi < 0 or vj < 0 or vi >= h or vj >= w:
                continue
            v = vi * w + vj
            if closed[v] or blocked[vi, vj]:
                continue
            if dl[k] > 1.0 and (blocked[ui, vj] or blocked[vi, uj]):   # no corner cutting
                continue
            ng = g[u] + dl[k] * 0.5 * (cost[ui, uj] + cost[vi, vj])
            if ng < g[v]:
                g[v] = ng
                par[v] = u
                heapq.heappush(heap, (ng + math.hypot(gi - vi, gj - vj), v))
    if not closed[goal]:
        return np.zeros((0, 2), np.int64)
    out = []
    u = goal
    while u != -1:
        out.append(u)
        u = par[u]
    res = np.empty((len(out), 2), np.int64)
    for k in range(len(out)):
        u = out[len(out) - 1 - k]
        res[k, 0] = u // w
        res[k, 1] = u % w
    return res


@njit(cache=True, nogil=True)
def _smooth(cells, cost, blocked):
    """Greedy shortcutting: jump from cell i to the farthest cell j whose straight segment
    touches no blocked cell AND costs no more than the A* path between them (so a shortcut
    can never pull the path closer to obstacles than A* chose to go)."""
    n = cells.shape[0]
    cum = np.zeros(n)
    for k in range(1, n):
        a0, b0 = cells[k - 1, 0], cells[k - 1, 1]
        a1, b1 = cells[k, 0], cells[k, 1]
        dl = math.hypot(a1 - a0, b1 - b0)
        cum[k] = cum[k - 1] + dl * 0.5 * (cost[a0, b0] + cost[a1, b1])
    keep = [0]
    i = 0
    while i < n - 1:
        best = i + 1
        for j in range(i + 2, n):
            i0, j0 = cells[i, 0], cells[i, 1]
            i1, j1 = cells[j, 0], cells[j, 1]
            L = math.hypot(i1 - i0, j1 - j0)
            m = int(L * 4) + 1
            ds = L / m
            c = 0.0
            ok = True
            for k in range(m + 1):
                t = k / m
                a = int(round(i0 + (i1 - i0) * t))
                b = int(round(j0 + (j1 - j0) * t))
                if blocked[a, b]:
                    ok = False
                    break
                w = 0.5 if (k == 0 or k == m) else 1.0
                c += w * ds * cost[a, b]
            if ok and c <= cum[j] - cum[i] + 1e-9:
                best = j
        keep.append(best)
        i = best
    out = np.empty((len(keep), 2), np.int64)
    for k in range(len(keep)):
        out[k] = cells[keep[k]]
    return out


class ClearancePlanner:
    def __init__(self, P, half_width, spin_radius=None):
        g = lambda k: getattr(P, k, DEFAULTS[k])
        self.r_hard = half_width + g("PLAN_MARGIN_M")
        self.spin_r = spin_radius if (spin_radius and g("PLAN_TURN_ROOM")) else None
        self.round = bool(g("PLAN_ROUND"))
        self.enc_w = g("PLAN_ENCLOSED_WEIGHT")
        self.enc_r = g("PLAN_ENCLOSED_RANGE_M")
        self.enc_free = int(g("PLAN_ENCLOSED_FREE_DIRS"))
        self.round_cut = g("PLAN_ROUND_CUT_M")
        self.soft = g("PLAN_SOFT_M")
        self.soft_w = g("PLAN_SOFT_WEIGHT")
        self.unk = g("PLAN_UNKNOWN_COST")
        self.esc = g("PLAN_ESCAPE_M")
        self.snap = g("PLAN_GOAL_SNAP_M")
        self.cell_target = g("PLAN_CELL_M")
        self.step = g("PLAN_PATH_STEP_M")
        occ = float(getattr(P, "MAP_OCC_THRESH", 1.0))
        self.obst_th = float(getattr(P, "PLAN_OBST_THRESH", 0.5 * occ if occ > 0 else occ))
        self.free_th = float(getattr(P, "MAP_FREE_THRESH", -0.5))
        self.last = None          # dict with the clearance grid of the last plan (for checks)
        self.info = ""

    def warmup(self):
        lo = np.zeros((60, 60), np.float32)
        lo[:, :] = self.free_th - 1
        lo[30, 20:40] = 10
        self.plan(lo, 0.05, 30, (-0.5, -0.5), (0.5, 0.5), None)

    # ── grid prep ──────────────────────────────────────────────────────────────
    def _prepare(self, lo, res, orig, pts_list, bbox_pts):
        f = max(1, int(round(self.cell_target / res)))
        c = res * f
        known = lo != 0
        rows = np.flatnonzero(known.any(1))
        cols = np.flatnonzero(known.any(0))
        extra = [np.asarray(p, float).reshape(-1, 2) for p in pts_list if p is not None]
        span = extra + [np.asarray(bbox_pts, float).reshape(-1, 2)]
        r0 = int(rows[0]) if len(rows) else orig
        r1 = int(rows[-1]) + 1 if len(rows) else orig + 1
        c0 = int(cols[0]) if len(cols) else orig
        c1 = int(cols[-1]) + 1 if len(cols) else orig + 1
        for e in span:
            if len(e):
                cx = np.round(e[:, 0] / res).astype(int) + orig
                cy = np.round(e[:, 1] / res).astype(int) + orig
                r0, r1 = min(r0, int(cy.min())), max(r1, int(cy.max()) + 1)
                c0, c1 = min(c0, int(cx.min())), max(c1, int(cx.max()) + 1)
        pad = int(math.ceil((self.r_hard + self.soft + 0.3) / res))
        n0, n1 = lo.shape
        r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
        r1, c1 = min(n0, r1 + pad), min(n1, c1 + pad)
        h = (r1 - r0 + f - 1) // f
        w = (c1 - c0 + f - 1) // f
        sub = np.zeros((h * f, w * f), np.float32)
        sub[: r1 - r0, : c1 - c0] = lo[r0:r1, c0:c1]
        blk = sub.reshape(h, f, w, f)
        obst = (blk > self.obst_th).any(axis=(1, 3))
        free = (blk < self.free_th).all(axis=(1, 3))
        for e in extra:
            if len(e):
                ci = ((np.round(e[:, 1] / res).astype(int) + orig) - r0) // f
                cj = ((np.round(e[:, 0] / res).astype(int) + orig) - c0) // f
                ok = (ci >= 0) & (ci < h) & (cj >= 0) & (cj < w)
                obst[ci[ok], cj[ok]] = True
        clear = _chamfer(obst, c)
        return dict(f=f, c=c, r0=r0, c0=c0, h=h, w=w, obst=obst, free=free, clear=clear,
                    res=res, orig=orig)

    def _to_cell(self, G, xy):
        i = ((int(round(xy[1] / G["res"])) + G["orig"]) - G["r0"]) // G["f"]
        j = ((int(round(xy[0] / G["res"])) + G["orig"]) - G["c0"]) // G["f"]
        return i, j

    def _to_xy(self, G, i, j):
        f, res = G["f"], G["res"]
        y = (G["r0"] + i * f + (f - 1) / 2.0 - G["orig"]) * res
        x = (G["c0"] + j * f + (f - 1) / 2.0 - G["orig"]) * res
        return x, y

    # ── plan ───────────────────────────────────────────────────────────────────
    def plan(self, lo, res, orig, start, goal, extra_pts=None):
        G = self._prepare(lo, res, int(orig), [extra_pts], np.array([start[:2], goal[:2]]))
        self.last = G
        h, w, c = G["h"], G["w"], G["c"]
        clear = G["clear"]
        si, sj = self._to_cell(G, start)
        gi, gj = self._to_cell(G, goal)
        if not (0 <= si < h and 0 <= sj < w and 0 <= gi < h and 0 <= gj < w):
            self.info = "start/goal outside grid"
            return None
        hard = clear < self.r_hard
        blocked = hard | G["obst"]
        cost = np.ones((h, w))
        soft = np.clip((self.r_hard + self.soft - clear) / self.soft, 0.0, 1.0)
        cost += self.soft_w * soft ** 2
        if self.enc_w > 0:
            cd = _closed_dirs(G["obst"], self.enc_r / c).astype(np.float64)
            cost += self.enc_w * (np.clip(cd - self.enc_free, 0, None) / (8 - self.enc_free)) ** 2
        cost[~G["free"]] *= self.unk
        # escape from the hard band around the start
        ii, jj = np.mgrid[0:h, 0:w]
        near_s = (ii - si) ** 2 + (jj - sj) ** 2 <= (self.esc / c) ** 2
        esc = near_s & hard & ~G["obst"]
        if blocked[si, sj]:
            blocked = blocked & ~esc
            cost[esc] += 30.0
            if G["obst"][si, sj]:
                G["obst"][si, sj] = False
                blocked[si, sj] = False
        # goal snapping
        if blocked[gi, gj]:
            cand = ~blocked & ((ii - gi) ** 2 + (jj - gj) ** 2 <= (self.snap / c) ** 2)
            if not cand.any():
                self.info = "goal is inside an obstacle's clearance band"
                return None
            d2 = np.where(cand, (ii - gi) ** 2 + (jj - gj) ** 2, 1 << 30)
            k = int(np.argmin(d2))
            gi, gj = k // w, k % w
        cells = _astar(cost, blocked, si, sj, gi, gj)
        if len(cells) == 0:
            self.info = "no path"
            return None
        sm = _smooth(cells, cost, blocked)
        if self.spin_r is not None and len(sm) > 2:
            sm = self._turn_room(sm, clear, blocked, c)
        smf = sm.astype(float)
        if self.round and len(sm) > 2:
            smf = self._round_corners(smf, blocked, self.round_cut / c)
        pts = np.array([self._to_xy(G, i, j) for i, j in smf])
        pts[0] = start[:2]
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            L = float(np.hypot(*(b - a)))
            n = max(1, int(math.ceil(L / self.step)))
            for k in range(1, n + 1):
                out.append(a + (b - a) * k / n)
        self.info = (f"{len(out)} pts, min clearance {float(min(clear[i, j] for i, j in cells)):.2f} m "
                     f"(hard {self.r_hard:.2f})")
        return [np.array(p, float) for p in out]

    @staticmethod
    def _seg_free(blocked, a, b):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        for k in range(n + 1):
            t = k / n
            if blocked[int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t))]:
                return False
        return True

    def _round_corners(self, pts, blocked, cut_cells, iters=4):
        """Chaikin corner cutting: each corner B (between A and C) is replaced by two points
        on BA and BC; a cut is made only if the new chord stays outside the hard band."""
        for it in range(iters):
            out = [pts[0]]
            for k in range(1, len(pts) - 1):
                A, B, C = pts[k - 1], pts[k], pts[k + 1]
                ab, cb = A - B, C - B
                la, lc = float(np.hypot(*ab)), float(np.hypot(*cb))
                if la < 1e-6 or lc < 1e-6:
                    out.append(B)
                    continue
                q = B + ab / la * min(0.25 * la, cut_cells)
                r = B + cb / lc * min(0.25 * lc, cut_cells)
                if self._seg_free(blocked, q, r):
                    out += [q, r]
                else:
                    out.append(B)
            out.append(pts[-1])
            pts = np.array(out)
        return pts

    def _turn_room(self, sm, clear, blocked, c):
        """The rover turns in place at path corners, and its corners swing out to spin_r.
        Move each corner (up to 0.20 m) to the nearest cell with clearance >= spin_r whose two
        segments stay clear; else to the reachable cell with the most clearance."""
        h, w = clear.shape
        out = [tuple(sm[0])]
        rmax = int(math.ceil(0.20 / c))
        offs = sorted(((di, dj) for di in range(-rmax, rmax + 1) for dj in range(-rmax, rmax + 1)
                       if di * di + dj * dj <= rmax * rmax), key=lambda d: d[0] ** 2 + d[1] ** 2)
        for k in range(1, len(sm) - 1):
            i, j = int(sm[k][0]), int(sm[k][1])
            if clear[i, j] >= self.spin_r:
                out.append((i, j))
                continue
            prev, nxt = out[-1], tuple(sm[k + 1])
            best, best_c = (i, j), clear[i, j]
            for di, dj in offs:
                a, b = i + di, j + dj
                if not (0 <= a < h and 0 <= b < w) or blocked[a, b] or clear[a, b] <= best_c:
                    continue
                if not (self._seg_free(blocked, prev, (a, b)) and self._seg_free(blocked, (a, b), nxt)):
                    continue
                best, best_c = (a, b), clear[a, b]
                if best_c >= self.spin_r:
                    break
            out.append(best)
        out.append(tuple(sm[-1]))
        return np.array(out, np.int64)

    # ── checks on the last grid ────────────────────────────────────────────────
    def path_min_clearance(self, path):
        """Min clearance (m) along a polyline on the last prepared grid; None if no grid."""
        G = self.last
        if G is None or path is None or len(path) == 0:
            return None
        m = 1e9
        P = np.asarray([np.asarray(p, float)[:2] for p in path])
        for a, b in zip(P[:-1] if len(P) > 1 else P, P[1:] if len(P) > 1 else P):
            n = max(1, int(np.hypot(*(b - a)) / (G["c"] * 0.5)))
            for k in range(n + 1):
                i, j = self._to_cell(G, a + (b - a) * k / n)
                if 0 <= i < G["h"] and 0 <= j < G["w"]:
                    m = min(m, float(G["clear"][i, j]))
        return m


def path_length(path):
    if path is None or len(path) < 2:
        return 0.0
    P = np.asarray([np.asarray(p, float)[:2] for p in path])
    return float(np.hypot(*np.diff(P, axis=0).T).sum())


def trim_path(path, xy):
    """Remaining part of a path from the projection of xy onto it."""
    P = np.asarray([np.asarray(p, float)[:2] for p in path])
    if len(P) < 2:
        return [p for p in P], 0.0
    best, bk, bpt = 1e9, 0, P[0]
    for k in range(len(P) - 1):
        a, b = P[k], P[k + 1]
        ab = b - a
        t = float(np.clip(np.dot(xy - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
        q = a + t * ab
        d = float(np.hypot(*(xy - q)))
        if d < best:
            best, bk, bpt = d, k, q
    return [np.array(bpt)] + [p.copy() for p in P[bk + 1:]], best

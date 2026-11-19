"""
Map and run metrics (used by the gates and, later, the paper).
==============================================================
    python -m phase3_onboard.tools.metrics MAP.npz [--room 4.0 4.0]
    python -m phase3_onboard.tools.metrics RUN_DIR/replay_v3      (reads summary + map)

Map metrics (no ground truth needed):
  * walls: RANSAC line extraction on occupied cells (lines >= --min-wall m)
  * wall_angle_err_deg : worst deviation of any wall from the dominant
                         orthogonal pair (0/90 deg family)
  * corner_angle_deg   : angle between the two longest roughly perpendicular walls
  * duplicated_walls   : walls rotated >3 deg off the dominant family, plus
                         parallel walls that double a longer one within 0.30 m
  * room_w/room_h      : spread of the wall offsets in each family
Run metrics: loop-closure error vs the start, acceptance rate, step timing.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


def occupied_points(log, res, orig, thresh=0.6):
    cy, cx = np.nonzero(log > thresh)
    return np.stack([(cx - orig) * res, (cy - orig) * res], axis=1)


def ransac_lines(pts, min_len=1.0, inlier=0.05, max_lines=12, iters=400, seed=0, res=0.05):
    rng = np.random.default_rng(seed)
    lines = []
    remaining = pts.copy()
    min_inl = max(8, int(min_len / res * 0.6))
    for _ in range(max_lines):
        if len(remaining) < min_inl:
            break
        best = None
        for _ in range(iters):
            i, j = rng.choice(len(remaining), 2, replace=False)
            p, q = remaining[i], remaining[j]
            d = q - p
            L = np.hypot(*d)
            if L < 0.2:
                continue
            nrm = np.array([-d[1], d[0]]) / L
            dist = np.abs((remaining - p) @ nrm)
            m = dist < inlier
            c = int(m.sum())
            if best is None or c > best[0]:
                best = (c, m, p, d / L)
        if best is None or best[0] < min_inl:
            break
        _, m, p, u = best
        inl = remaining[m]
        # refine with PCA
        mu = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - mu)
        u = vt[0]
        proj = (inl - mu) @ u
        # keep the longest contiguous run (gap <= 0.25 m) so separate walls are not merged
        order = np.sort(proj)
        gaps = np.diff(order)
        cuts = np.nonzero(gaps > 0.25)[0]
        segs = np.split(order, cuts + 1)
        segl = max(segs, key=lambda s: s[-1] - s[0])
        length = float(segl[-1] - segl[0])
        # dense walls only: a real wall fills >= 50 % of the cells along its length
        if length >= min_len and len(segl) >= 0.5 * length / res:
            ang = math.degrees(math.atan2(u[1], u[0])) % 180.0
            nrm = np.array([-u[1], u[0]])
            lines.append({"angle_deg": ang, "length_m": length, "n": int(m.sum()),
                          "center": mu.tolist(), "normal": nrm.tolist()})
        remaining = remaining[~m]
    return lines


def _mod90_diff(a, b):
    d = (a - b) % 90.0
    return min(d, 90.0 - d)


def map_metrics(log, res, orig, room=None, min_wall=1.0):
    pts = occupied_points(log, res, orig)
    out = {"occupied_cells": int(len(pts))}
    if len(pts) < 20:
        out["error"] = "map nearly empty"
        return out
    lines = ransac_lines(pts, min_len=min_wall, res=res)
    out["walls"] = lines
    if not lines:
        out["error"] = "no walls found"
        return out
    lines.sort(key=lambda l: -l["length_m"])
    ref = lines[0]["angle_deg"]
    devs = [_mod90_diff(l["angle_deg"], ref) for l in lines]
    out["wall_angle_err_deg"] = round(float(max(devs)), 2)
    # corner angle between longest wall and longest roughly-perpendicular wall
    corner = None
    for l in lines[1:]:
        d = abs((l["angle_deg"] - ref + 90) % 180 - 90)   # 0..90
        if d > 45:
            corner = d
            break
    out["corner_angle_deg"] = None if corner is None else round(float(corner), 2)
    # duplicates
    rotated = sum(1 for d in devs if d > 3.0)
    doubled = 0
    for i, a in enumerate(lines):
        for b in lines[:i]:
            if abs(((a["angle_deg"] - b["angle_deg"]) + 90) % 180 - 90) < 5:
                nb = np.array(b["normal"])
                off = abs((np.array(a["center"]) - np.array(b["center"])) @ nb)
                if 0.06 < off < 0.30:
                    doubled += 1
                    break
    out["duplicated_walls"] = int(rotated + doubled)
    out["rotated_walls"] = int(rotated)
    out["doubled_walls"] = int(doubled)
    # room dimensions from the two families
    fam = {0: [], 1: []}
    u0 = math.radians(ref)
    n0 = np.array([-math.sin(u0), math.cos(u0)])
    n1 = np.array([math.cos(u0), math.sin(u0)])
    for l in lines:
        d = abs((l["angle_deg"] - ref + 90) % 180 - 90)
        c = np.array(l["center"])
        if d < 10:
            fam[0].append(float(c @ n0))
        elif d > 80:
            fam[1].append(float(c @ n1))
    w = max(fam[1]) - min(fam[1]) if len(fam[1]) >= 2 else None
    h = max(fam[0]) - min(fam[0]) if len(fam[0]) >= 2 else None
    out["room_w_m"] = None if w is None else round(w, 3)
    out["room_h_m"] = None if h is None else round(h, 3)
    if room and w and h:
        a, b = sorted(room)
        m1, m2 = sorted([w, h])
        out["room_dim_err_pct"] = round(100 * max(abs(m1 - a) / a, abs(m2 - b) / b), 2)
    return out


def loop_error(path):
    """path: Nx4 (t, x, y, th).  Final pose vs the start pose."""
    p = np.asarray(path)
    if len(p) < 2:
        return None
    dx, dy = p[-1, 1] - p[0, 1], p[-1, 2] - p[0, 2]
    dth = (p[-1, 3] - p[0, 3] + math.pi) % (2 * math.pi) - math.pi
    return {"loop_xy_m": round(math.hypot(dx, dy), 3), "loop_yaw_deg": round(abs(math.degrees(dth)), 2)}


def percentiles(x):
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if len(x) == 0:
        return None
    return {"p50": round(float(np.percentile(x, 50)), 2), "p95": round(float(np.percentile(x, 95)), 2),
            "p99": round(float(np.percentile(x, 99)), 2), "max": round(float(x.max()), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--room", nargs=2, type=float, default=None, help="tape-measured W H (m)")
    ap.add_argument("--min-wall", type=float, default=1.0)
    a = ap.parse_args()
    p = a.path
    if os.path.isdir(p):
        p = os.path.join(p, "map.npz")
    d = np.load(p)
    m = map_metrics(d["log"], float(d["res"]), int(d["orig"]), a.room, a.min_wall)
    m.pop("walls", None)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()

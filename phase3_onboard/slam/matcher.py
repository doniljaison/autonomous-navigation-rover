"""
Scan-to-map matcher.
====================
1. Correlative search in a small (x, y, theta) window around the motion prior,
   scored on a likelihood field built from the map's distance field, with a
   mild Gaussian penalty for leaving the prior (breaks symmetric ties).
2. Gauss-Newton refinement minimising the point-to-surface distance (bilinear
   distance field; near walls this is point-to-line), Huber-weighted.
3. Diagnostics for the gate: inliers, RMS, ambiguity (second distinct peak),
   and the normalised translational Hessian eigenvalues (degeneracy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from phase3_onboard.slam.jit import njit


@dataclass
class MatchResult:
    pose: np.ndarray
    score: float
    second: float
    ambiguity: float
    n: int
    inliers: int
    inlier_ratio: float
    rms: float
    eig_min: float
    eig_vec_weak: np.ndarray
    degenerate: bool
    iters: int


@njit
def _correlative(lf, x0, y0, res, px, py, prior_x, prior_y, thetas, nxy,
                 lam_xy, lam_th, prior_th):
    """Exhaustive search.  Returns score volume [n_th, 2*nxy+1, 2*nxy+1]
    (already prior-penalised) — normalised by the number of points."""
    h, w = lf.shape
    n = px.shape[0]
    side = 2 * nxy + 1
    out = np.zeros((thetas.shape[0], side, side), dtype=np.float32)
    cxs = np.empty(n, dtype=np.int64)
    cys = np.empty(n, dtype=np.int64)
    for it in range(thetas.shape[0]):
        th = thetas[it]
        c = math.cos(th)
        s = math.sin(th)
        for k in range(n):
            wx = c * px[k] - s * py[k] + prior_x
            wy = s * px[k] + c * py[k] + prior_y
            cxs[k] = int(math.floor(wx / res + 0.5)) - x0
            cys[k] = int(math.floor(wy / res + 0.5)) - y0
        dth = th - prior_th
        pen_th = lam_th * dth * dth
        for iy in range(side):
            dy = iy - nxy
            for ix in range(side):
                dx = ix - nxy
                acc = 0.0
                for k in range(n):
                    cx = cxs[k] + dx
                    cy = cys[k] + dy
                    if 0 <= cx < w and 0 <= cy < h:
                        acc += lf[cy, cx]
                d2 = (dx * dx + dy * dy) * res * res
                out[it, iy, ix] = acc / n - lam_xy * d2 - pen_th
    return out


@njit
def _bilinear(D, x0, y0, res, wx, wy, cap):
    """Distance + gradient (metres) at world point; cap outside the window."""
    h, w = D.shape
    gx = wx / res - x0
    gy = wy / res - y0
    ix = int(math.floor(gx))
    iy = int(math.floor(gy))
    if ix < 0 or iy < 0 or ix + 1 >= w or iy + 1 >= h:
        return cap, 0.0, 0.0
    fx = gx - ix
    fy = gy - iy
    d00 = D[iy, ix]
    d10 = D[iy, ix + 1]
    d01 = D[iy + 1, ix]
    d11 = D[iy + 1, ix + 1]
    d = (d00 * (1 - fx) * (1 - fy) + d10 * fx * (1 - fy) + d01 * (1 - fx) * fy + d11 * fx * fy)
    ddx = ((d10 - d00) * (1 - fy) + (d11 - d01) * fy) / res
    ddy = ((d01 - d00) * (1 - fx) + (d11 - d10) * fx) / res
    return d, ddx, ddy


@njit
def _gn(D, x0, y0, res, px, py, pose, iters, huber, cap, inlier_thr):
    x = pose[0]
    y = pose[1]
    th = pose[2]
    n = px.shape[0]
    H = np.zeros((3, 3))
    used = 0
    for it in range(iters):
        H[:, :] = 0.0
        g = np.zeros(3)
        c = math.cos(th)
        s = math.sin(th)
        for k in range(n):
            wx = c * px[k] - s * py[k] + x
            wy = s * px[k] + c * py[k] + y
            d, ddx, ddy = _bilinear(D, x0, y0, res, wx, wy, cap)
            if d >= cap * 0.98:
                continue
            wgt = 1.0 if d <= huber else huber / d
            jt = (-s * px[k] - c * py[k]) * ddx + (c * px[k] - s * py[k]) * ddy
            J0 = ddx
            J1 = ddy
            J2 = jt
            H[0, 0] += wgt * J0 * J0; H[0, 1] += wgt * J0 * J1; H[0, 2] += wgt * J0 * J2
            H[1, 1] += wgt * J1 * J1; H[1, 2] += wgt * J1 * J2; H[2, 2] += wgt * J2 * J2
            g[0] += wgt * J0 * d; g[1] += wgt * J1 * d; g[2] += wgt * J2 * d
        H[1, 0] = H[0, 1]; H[2, 0] = H[0, 2]; H[2, 1] = H[1, 2]
        Hd = H.copy()
        for i in range(3):
            Hd[i, i] += 1e-6 + 1e-3 * H[i, i]
        delta = np.linalg.solve(Hd, -g)
        x += delta[0]
        y += delta[1]
        th += delta[2]
        used = it + 1
        if abs(delta[0]) < 1e-4 and abs(delta[1]) < 1e-4 and abs(delta[2]) < 1e-4:
            break
    # final statistics
    c = math.cos(th)
    s = math.sin(th)
    inl = 0
    ss = 0.0
    Ht = np.zeros((2, 2))
    for k in range(n):
        wx = c * px[k] - s * py[k] + x
        wy = s * px[k] + c * py[k] + y
        d, ddx, ddy = _bilinear(D, x0, y0, res, wx, wy, cap)
        if d < inlier_thr:
            inl += 1
            ss += d * d
            Ht[0, 0] += ddx * ddx; Ht[0, 1] += ddx * ddy; Ht[1, 1] += ddy * ddy
    Ht[1, 0] = Ht[0, 1]
    rms = math.sqrt(ss / inl) if inl > 0 else 1e9
    return np.array([x, y, th]), inl, rms, Ht, used


def warmup():
    """Trigger numba compilation (and cache) before the real-time loop starts."""
    D = np.zeros((20, 20), np.float32) + 0.1
    lf = np.exp(-D)
    px = np.array([0.1, 0.2, 0.3]); py = np.array([0.0, 0.1, 0.2])
    _correlative(lf, 0, 0, 0.05, px, py, 0.5, 0.5, np.array([0.0, 0.01]), 1, 0.1, 0.1, 0.0)
    _gn(D, 0, 0, 0.05, px, py, np.array([0.5, 0.5, 0.0]), 2, 0.05, 0.4, 0.1)


def match(D: np.ndarray, x0: int, y0: int, res: float, pts: np.ndarray, prior: np.ndarray,
          th_half: float, P) -> MatchResult:
    """D: distance field (metres) window with origin cell (x0, y0) in grid coords.
    NOTE: x0/y0 are expressed as cell indices relative to the grid origin cell,
    i.e. caller passes x0 - orig, y0 - orig so world = cell * res."""
    px = np.ascontiguousarray(pts[:, 0], dtype=np.float64)
    py = np.ascontiguousarray(pts[:, 1], dtype=np.float64)
    sigma = P.MATCH_SIGMA_M
    lf = np.exp(-(D.astype(np.float64) ** 2) / (2 * sigma * sigma)).astype(np.float32)

    step_th = math.radians(P.MATCH_COARSE_TH_STEP_DEG)
    n_th = max(1, int(math.ceil(th_half / step_th)))
    thetas = prior[2] + step_th * np.arange(-n_th, n_th + 1)
    nxy = max(1, int(round(P.MATCH_COARSE_XY_M / res)))
    # prior penalty: ~ -0.04 score at the window edge (score is in [0,1])
    lam_xy = 0.04 / max(P.MATCH_COARSE_XY_M ** 2, 1e-6)
    lam_th = 0.04 / max(th_half ** 2, 1e-6)
    vol = _correlative(lf, x0, y0, res, px, py, prior[0], prior[1], thetas, nxy,
                       lam_xy, lam_th, prior[2])
    best_idx = np.unravel_index(int(np.argmax(vol)), vol.shape)
    best = float(vol[best_idx])
    # Rotational ambiguity: second *local maximum* of the yaw profile outside
    # +/-3 steps of the best.  (Translational ambiguity along featureless walls is
    # handled by the Hessian degeneracy test instead.)
    it, iy, ix = best_idx
    prof = vol.reshape(vol.shape[0], -1).max(axis=1)
    second = -1.0
    for k in range(len(prof)):
        if abs(k - it) <= 3:
            continue
        left = prof[k - 1] if k > 0 else -np.inf
        right = prof[k + 1] if k < len(prof) - 1 else -np.inf
        if prof[k] >= left and prof[k] >= right and prof[k] > second:
            second = float(prof[k])
    coarse = np.array([prior[0] + (ix - nxy) * res, prior[1] + (iy - nxy) * res, thetas[it]])

    pose, inl, rms, Ht, iters = _gn(D, x0, y0, res, px, py, coarse, int(P.MATCH_GN_ITERS),
                                     float(P.MATCH_HUBER_M), float(P.MATCH_FIELD_MAX_M) * 0.999,
                                     float(P.MATCH_INLIER_M))
    n = len(px)
    if inl > 0:
        Hn = Ht / inl
        evals, evecs = np.linalg.eigh(Hn)
        eig_min = float(evals[0])
        weak = evecs[:, 0]
    else:
        eig_min, weak = 0.0, np.array([1.0, 0.0])
    amb = second / best if best > 1e-6 else 1.0
    return MatchResult(pose=pose, score=best, second=second, ambiguity=amb, n=n, inliers=int(inl),
                       inlier_ratio=inl / max(n, 1), rms=float(rms), eig_min=eig_min,
                       eig_vec_weak=weak, degenerate=eig_min < P.GATE_DEGENERATE_EIG, iters=int(iters))

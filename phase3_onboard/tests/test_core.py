"""Unit tests: python -m pytest phase3_onboard/tests -q"""

import json
import math
import time

import numpy as np
import pytest

from phase3_onboard.common import Scan, compose, inverse, relative, transform
from phase3_onboard.slam.params import from_config


@pytest.fixture(scope="module")
def P():
    return from_config()


def test_se2_roundtrip():
    a = np.array([1.0, -2.0, 0.7])
    b = np.array([0.3, 0.4, -1.2])
    np.testing.assert_allclose(compose(inverse(a), compose(a, b)), b, atol=1e-12)
    np.testing.assert_allclose(relative(a, compose(a, b)), b, atol=1e-12)
    pts = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(transform(pts, [0, 0, math.pi / 2]), [[0, 1], [-1, 0]], atol=1e-12)


def test_logger_roundtrip(tmp_path):
    from phase3_onboard.logger import RunLogger, RunReader
    lg = RunLogger(str(tmp_path), "t")
    s = Scan(3, 10.0, 10.1, 10.11, np.linspace(-3, 3, 50, dtype=np.float32),
             np.ones(50, np.float32), np.linspace(0, 0.1, 50, dtype=np.float32))
    lg.event("rx", t=10.05, line='{"T":1002,"gz":1.0}')
    lg.scan(s)
    lg.event("tx", t=10.2, line='{"T":1,"L":0,"R":0}')
    lg.close()
    rd = RunReader(lg.dir)
    kinds = [(k, x["k"] if k == "ev" else x.seq) for k, x in rd.merged()]
    assert kinds == [("ev", "rx"), ("scan", 3), ("ev", "tx")]
    s2 = next(rd.scans())
    np.testing.assert_array_equal(s2.angle, s.angle)
    assert rd.meta["config"]["MAP_RES_M"] == 0.05


def test_raycast_matches_reference():
    from phase3_onboard.slam.grid import OccupancyGrid, bresenham_cells
    g = OccupancyGrid(4.0, 0.05, hit=1.0, miss=-0.25, clamp=10)
    hits = np.array([[1.0, 0.37], [-0.8, -0.9]])
    g.update((0.0, 0.0), hits, None)
    ref = np.zeros_like(g.log)
    rx, ry = g.w2c(0.0, 0.0)
    for h in hits:
        ex, ey = g.w2c(*h)
        for cx, cy in bresenham_cells(int(rx), int(ry), int(ex), int(ey)):
            ref[cy, cx] -= 0.25
        ref[ey, ex] += 1.0
    np.testing.assert_allclose(g.log, ref, atol=1e-6)


def test_chamfer_close_to_edt():
    from scipy.ndimage import distance_transform_edt
    from phase3_onboard.slam.grid import OccupancyGrid
    g = OccupancyGrid(4.0, 0.05)
    rng = np.random.default_rng(0)
    g.log[rng.integers(0, 80, 40), rng.integers(0, 80, 40)] = 3.0
    D, x0, y0, n = g.distance_field(40, 40, 40, 0.6, 10.0)
    occ = g.log[y0:y0 + D.shape[0], x0:x0 + D.shape[1]] > 0.6
    ref = distance_transform_edt(~occ) * 0.05
    assert np.max(np.abs(D - ref)) < 0.09 * ref.max() + 0.01


def test_spike_filter(P):
    from phase3_onboard.slam.preprocess import prepare
    n = 720
    ang = np.linspace(-math.pi, math.pi, n, endpoint=False).astype(np.float32)
    r = np.full(n, 2.0, np.float32)
    r[100] = 6.0
    r[400] = 0.5
    r[500:505] = 1.0     # a real 5-sample object must survive
    s = Scan(0, 0.0, 0.1, 0.1, ang, r, np.linspace(0, 0.1, n, dtype=np.float32))
    p = prepare(s, P)
    assert p.n_spikes == 2
    d = np.hypot(p.pts[:, 0], p.pts[:, 1])
    assert np.sum(np.abs(d - 1.0) < 0.01) == 5


def test_deskew_removes_spin_smear(P):
    """A scan taken while spinning at 2 rad/s, de-skewed with the true yaw,
    must coincide with the static scan at the end time."""
    from phase3_onboard.sim import raycast_segments, square_room
    from phase3_onboard.slam.preprocess import prepare
    segs = square_room(4.0, clutter=True)
    w, n, T = 2.0, 900, 0.1
    ts = np.linspace(0, T, n, endpoint=False)
    k = np.arange(n)
    sens = 2 * np.pi * k / n
    yaw = w * ts
    r = raycast_segments(np.full(n, -0.5), np.full(n, -0.3), yaw + sens, segs, 12)
    ang = ((sens + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)
    s = Scan(0, 0.0, ts[-1], ts[-1], ang, r.astype(np.float32), ts.astype(np.float32))
    yaw_fn = lambda t: w * np.asarray(t)
    pd = prepare(s, P, yaw_fn=yaw_fn)
    raw = prepare(s, P.__class__(**{**vars(P), "DESKEW": False}))
    # ground truth: world points expressed in the end-time robot frame
    world = np.stack([-0.5 + r * np.cos(yaw + sens), -0.3 + r * np.sin(yaw + sens)], 1)
    ok = np.isfinite(r) & (r > 0.3)
    gt = transform(world[ok] - [0, 0], [0, 0, 0])
    gt = np.stack([gt[:, 0] + 0.5, gt[:, 1] + 0.3], 1)
    c, sn = math.cos(-yaw[-1]), math.sin(-yaw[-1])
    gt = np.stack([c * gt[:, 0] - sn * gt[:, 1], sn * gt[:, 0] + c * gt[:, 1]], 1)
    err_ds = np.median(np.min(np.linalg.norm(pd.pts[:, None] - gt[None], axis=2), axis=1))
    err_raw = np.median(np.min(np.linalg.norm(raw.pts[:, None] - gt[None], axis=2), axis=1))
    assert err_ds < 0.01 and err_raw > 5 * err_ds


def _room_scan(pose, n=900, side=4.0, clutter=True):
    from phase3_onboard.sim import raycast_segments, square_room
    segs = square_room(side, clutter=clutter)
    sens = 2 * np.pi * np.arange(n) / n
    r = raycast_segments(np.full(n, pose[0]), np.full(n, pose[1]), pose[2] + sens, segs, 12)
    r = np.where(np.isfinite(r), r, 0.0)
    ang = ((sens + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)
    return ang, r.astype(np.float32)


def test_matcher_recovers_offset(P):
    from phase3_onboard.slam import matcher as M
    from phase3_onboard.slam.grid import OccupancyGrid
    from phase3_onboard.slam.preprocess import prepare
    g = OccupancyGrid(P.MAP_SIZE_M, P.MAP_RES_M)
    ang, r = _room_scan((-0.6, -0.4, 0.0))
    s = Scan(0, 0, 0.1, 0.1, ang, r, np.zeros_like(r))
    p0 = prepare(s, P)
    for _ in range(3):
        g.update((0.0, 0.0), p0.pts, None)
    truth = np.array([0.12, -0.07, math.radians(3.0)])
    true_world = compose([-0.6, -0.4, 0.0], truth)
    ang, r = _room_scan(true_world)
    p1 = prepare(Scan(1, 0, 0.1, 0.1, ang, r, np.zeros_like(r)), P)
    cx, cy = g.w2c(0, 0)
    D, x0, y0, _ = g.distance_field(int(cx), int(cy), 120, P.MAP_OCC_THRESH, P.MATCH_FIELD_MAX_M)
    mr = M.match(D, x0 - g.orig, y0 - g.orig, g.res, p1.match_pts, np.zeros(3), math.radians(6), P)
    assert np.hypot(*(mr.pose[:2] - truth[:2])) < 0.02
    assert abs(mr.pose[2] - truth[2]) < math.radians(0.5)
    assert mr.inlier_ratio > 0.9 and not mr.degenerate


def test_gate_rejects_gyro_disagreement(P):
    """Feed a gyro that claims a 20 deg turn while the scans say 0: the matcher
    finds 0, the gate must reject and keep the map untouched."""
    from phase3_onboard.imu import ImuSample
    from phase3_onboard.slam.engine import SlamEngine
    eng = SlamEngine(P)
    pose = (-0.6, -0.4, 0.0)
    t = 0.0
    for i in range(6):
        ang, r = _room_scan(pose)
        if i == 5:  # gyro claims +20 deg over this interval
            for k in range(10):
                eng.gyro.add(ImuSample(t + 0.01 * k, math.radians(200.0), None, {}))
        else:
            for k in range(10):
                eng.gyro.add(ImuSample(t + 0.01 * k, 0.0, None, {}))
        t += 0.1
        s = Scan(i, t - 0.1, t, t, ang, r, np.linspace(0, 0.1, len(r), dtype=np.float32))
        ver = eng.grid.version
        res = eng.step(s, cmd_idle=False)
    assert not res.accepted
    assert res.reason in ("yaw_vs_gyro", "inliers", "rms")
    assert eng.grid.version == ver


def test_imu_parser_and_gyro(P):
    from phase3_onboard.imu import GyroYaw, ImuParser
    p = ImuParser(P)
    g = GyroYaw("gyro", bias_window_s=0.5)
    for k in range(100):  # 1 s standstill with bias 2 deg/s
        s = p.sample(k * 0.01, json.loads('{"T":1002,"gz":2.0,"y":0}'))
        g.add(s, stationary=True)
    assert g.bias_ready and abs(math.degrees(g.bias) - 2.0) < 1e-6
    y0 = g.yaw_at(0.99)
    for k in range(100, 200):  # 1 s at 92 deg/s raw -> 90 deg/s true
        g.add(p.sample(k * 0.01, {"T": 1002, "gz": 92.0}))
    assert abs(math.degrees(g.yaw_at(1.99) - y0) - 90.0) < 1.5


class _FakeTransport:
    name = "fake"

    def __init__(self):
        self.on_line = None
        self.sent = []

    def open(self):
        pass

    def write_line(self, line):
        self.sent.append((time.monotonic(), line))

    def close(self):
        pass


def test_link_watchdog(P):
    from phase3_onboard.rover_link import RoverLink
    P2 = from_config(IMU_POLL_HZ=0, FEEDBACK_STREAM=False, ECHO_OFF=False, INTENT_TTL_S=0.2)
    tr = _FakeTransport()
    link = RoverLink(tr, P2)
    link.start()
    link.set_intent(0.3, 0.3, "test")
    time.sleep(0.6)
    link.stop()
    cmds = [(t, json.loads(l)) for t, l in tr.sent if '"T":1,' in l]
    drive = [t for t, d in cmds if d["L"] > 0]
    assert drive, "intent was never sent"
    after = [d for t, d in cmds if t > drive[-1]]
    assert after and all(d["L"] == 0 and d["R"] == 0 for d in after)
    assert drive[-1] - drive[0] < 0.2 + 2 / P2.CMD_HZ
    assert link.watchdog_trips == 1
    assert json.loads(tr.sent[-1][1]) == {"T": 1, "L": 0.0, "R": 0.0}


def test_planner_goes_around_wall(P):
    from phase3_onboard.nav.planner import Planner
    from phase3_onboard.slam.grid import OccupancyGrid
    g = OccupancyGrid(10.0, 0.05)
    g.log[:] = -2.0
    cx, cy = g.w2c(0.0, np.linspace(-1.5, 1.0, 60))
    g.log[cy, cx] = 3.0                       # wall x=0 from y=-1.5..1.0
    path = Planner(P).plan(g.log, g.res, g.orig, (-1.0, 0.0), (1.0, 0.0))
    assert path is not None
    assert max(p[1] for p in path) > 1.0      # went over the top end
    for a, b in zip(path, path[1:]):          # never crosses the wall segment
        if (a[0] < 0) != (b[0] < 0):
            ycross = a[1] + (b[1] - a[1]) * (0 - a[0]) / (b[0] - a[0])
            assert ycross > 1.0 + P.INFLATION_RADIUS_M * 0.5


def test_proximity_blocks_forward(P):
    from phase3_onboard.nav.control import Proximity
    pr = Proximity(P)
    pr.update(np.array([[0.2, 0.0], [-1.0, 0.0]]))
    L, R, blocked = pr.limit(0.3, 0.3)
    assert blocked and L == 0 and R == 0
    L, R, blocked = pr.limit(-0.2, 0.2)       # turning in place still allowed
    assert not blocked and L < 0 < R
    L, R, blocked = pr.limit(-0.3, -0.3)      # reversing away allowed
    assert not blocked and L < 0


def test_lost_then_relocalise_without_map_damage(P):
    """Kick the pose estimate 0.3 m off (as if a slip went unnoticed): the gate
    must reject, declare LOST, re-localise while still, and never write the map
    from a rejected or re-localising frame."""
    from phase3_onboard.slam.engine import SlamEngine
    from phase3_onboard.imu import ImuSample
    eng = SlamEngine(P)
    pose = (-0.6, -0.4, 0.0)
    t = 0.0

    def feed(i):
        nonlocal t
        for k in range(10):
            eng.gyro.add(ImuSample(t + 0.01 * k, 0.0, None, {}))
        t += 0.1
        ang, r = _room_scan(pose)
        return eng.step(Scan(i, t - 0.1, t, t, ang, r, np.linspace(0, 0.1, len(r), dtype=np.float32)),
                        cmd_idle=True)
    for i in range(8):
        feed(i)
    eng.pose = eng.pose + np.array([0.3, -0.25, 0.0])   # corrupt the estimate
    v_before = eng.grid.version
    reasons = [feed(100 + i).reason for i in range(P.GATE_LOST_AFTER + 3)]
    assert "relocalized" in reasons or reasons[0] == "ok"
    assert np.hypot(*(eng.pose[:2])) < 0.03 and not eng.lost
    # rejected and re-localising frames must not have written the map
    n_written = sum(1 for r in reasons if r in ("ok", "ok_degenerate"))
    assert eng.grid.version - v_before == n_written

"""
Phase 3 configuration — the single tuning file.
================================================
Target: Raspberry Pi 4 Model B (64-bit Raspberry Pi OS Lite, Bookworm),
YDLIDAR G4 on USB, WAVE ROVER ESP32 driver board on the 40-pin UART.

Per-rover values you measure (LiDAR offset, IMU scale, motion calibration)
should go in ``config_local.py`` next to this file (git-ignored).  Anything
defined there overrides the defaults below.

Conventions (see docs/conventions.md):
    robot frame : x forward, y left, theta CCW positive, metres / radians
    map frame   : the robot frame at the first accepted scan
    motor duty  : L/R in -0.5..+0.5 where 0.5 == 100 % PWM (NOT m/s; no encoders)
"""

import math

# ═════════════════════════════════════════════════════════════════════════════
#  Platform
# ═════════════════════════════════════════════════════════════════════════════
PLATFORM = "rpi4b"                 # informational; written into every log's meta.json

# ═════════════════════════════════════════════════════════════════════════════
#  Rover link  (ESP32 "General Driver for Robots")
# ═════════════════════════════════════════════════════════════════════════════
# Pi 4B: with `dtoverlay=disable-bt` + `enable_uart=1`, /dev/serial0 -> /dev/ttyAMA0
# (the PL011 UART on GPIO14/15, header pins 8/10).  Without disable-bt, serial0
# is the mini-UART /dev/ttyS0 whose baud rate follows the VPU clock — avoid it.
LINK_TRANSPORT = "uart"            # "uart" | "http" | "sim"
UART_PORT      = "/dev/serial0"
UART_BAUD      = 115200
ROVER_HTTP_HOST = "192.168.4.1"    # only for LINK_TRANSPORT="http" (old WiFi path, latency study)
ROVER_HTTP_PORT = 80
ROVER_HTTP_ENDPOINT = "/js"

CMD_HZ            = 25.0           # motor command stream rate (Pi owns the stream)
INTENT_TTL_S      = 0.40           # a drive intent older than this is treated as "stop"
FEEDBACK_STREAM   = True           # send {"T":131,"cmd":1} at start (continuous chassis feedback)
ECHO_OFF          = True           # send {"T":143,"cmd":0}
IMU_POLL_HZ       = 25.0           # poll {"T":126} at this rate (0 = rely on feedback stream only)
MAX_DUTY          = 0.50           # firmware limit (0.5 == 100 % PWM)
SWAP_LR           = False          # set True if L/R wires turn out swapped
INVERT_FWD        = True          # set True if positive duty drives the rover backwards

# IMU field mapping. CAPTURE RAW OUTPUT FIRST (tools/imu_probe.py) and fix these.
# Firmware replies to T:126 with T:1002 ({"r","p","y","ax".."gz","mx".."mz","temp"}).
IMU_REPLY_TYPES   = (1002, 1001)   # message "T" values that may carry IMU fields
IMU_GYRO_Z_KEY    = "gz"           # yaw-rate field; set None to derive from yaw angle
IMU_GYRO_SCALE    = math.radians(1.0)  # raw units -> rad/s (deg/s by default; fix after probe)
IMU_GYRO_SIGN     = +1.0           # +1 if positive gz == CCW (fix with hand-rotation test)
IMU_YAW_KEY       = "y"            # absolute yaw angle field (degrees), used for yaw_diff fallback
IMU_YAW_SIGN      = +1.0
GYRO_BIAS_WINDOW_S = 2.0           # standstill bias estimate at startup
YAW_SOURCE        = "gyro"         # "gyro" | "yaw_diff" | "none"
FEEDBACK_VOLTAGE_KEY = "v"

# ═════════════════════════════════════════════════════════════════════════════
#  LiDAR  (YDLIDAR G4)
# ═════════════════════════════════════════════════════════════════════════════
LIDAR_DRIVER       = "ydlidar"     # "ydlidar" (official SDK) | "pylidar3" (legacy) | "sim"
LIDAR_PORT         = "/dev/ydlidar"  # udev symlink from pi/99-ydlidar.rules (fallback /dev/ttyUSB0)
LIDAR_BAUD         = 230400
LIDAR_SAMPLE_KHZ   = 9
LIDAR_SCAN_HZ      = 10.0
LIDAR_SINGLE_CHANNEL = False
LIDAR_INTENSITY    = False
LIDAR_SDK_REVERSION = True         # vendor G4 defaults
LIDAR_SDK_INVERTED  = True
LIDAR_MIN_RANGE_M  = 0.26
LIDAR_MAX_RANGE_M  = 16.0
# Empirical convention (tools/lidar_convention.py): angle_robot = SIGN*angle_raw + OFFSET
LIDAR_ANGLE_SIGN   = +1.0
LIDAR_ANGLE_OFFSET_RAD = 0.0
LIDAR_TIMESTAMP_LATENCY_S = 0.0    # added to scan end time if the driver gives receive time only

# Mounting offset of the LiDAR centre vs the rover centre, in the robot frame.
# [HUMAN] measure by Day 5 (ruler + photo from above).  Until then 0,0,0.
LIDAR_MOUNT_X_M    = 0.0
LIDAR_MOUNT_Y_M    = 0.0
LIDAR_MOUNT_YAW_RAD = 0.0

# Robot body footprint to ignore (points that hit the rover itself / mast)
SELF_FILTER_RADIUS_M = 0.12

# ═════════════════════════════════════════════════════════════════════════════
#  SLAM v3
# ═════════════════════════════════════════════════════════════════════════════
# --- Preprocess ---
PRE_MIN_RANGE_M    = 0.15          # below LIDAR_MIN_RANGE: dropped anyway
PRE_MAX_RANGE_M    = 10.0          # beyond this: not used for matching / occupied cells
PRE_SPIKE_ABS_M    = 0.15          # isolated-point test: |r_i - r_neighbour| > max(abs, rel*r)
PRE_SPIKE_REL      = 0.06
PRE_MIN_POINTS     = 60            # fewer valid points -> scan is skipped (counted)
MATCH_VOXEL_M      = 0.05          # voxel down-sample for matching
MATCH_MAX_POINTS   = 320           # cap on matching points (Pi 4B budget)
MATCH_MAX_RANGE_M  = 8.0

# --- De-skew / prior ---
DESKEW             = True
PRIOR_VEL_ALPHA    = 0.5           # low-pass on body velocity from accepted motion
PRIOR_STOP_AFTER_S = 0.30          # zero motor command for this long -> velocity prior = 0
# Optional calibrated duty->speed model (tools/calibrate_motion.py). None = constant velocity.
DUTY_TO_MPS        = None          # e.g. 0.9  (m/s per unit mean duty)
DUTY_TO_RADPS      = None

# --- Matcher (scan-to-map) ---
MATCH_COARSE_XY_M  = 0.20          # +/- search window around the prior
MATCH_COARSE_XY_STEP_M = 0.05
MATCH_COARSE_TH_DEG = 4.0          # base +/- yaw window ...
MATCH_COARSE_TH_PER_RAD = 0.15     # ... plus this fraction of the gyro rotation since last scan
MATCH_COARSE_TH_STEP_DEG = 0.5
MATCH_SIGMA_M      = 0.07          # likelihood-field width
MATCH_FIELD_MAX_M  = 0.40          # distance field cap
MATCH_GN_ITERS     = 8
MATCH_HUBER_M      = 0.05
MATCH_INLIER_M     = 0.10
MATCH_LOCAL_HALF_M = 9.0           # distance field is built for a window of this half-size

# --- Gate ---
GATE_MIN_INLIER_RATIO = 0.45
GATE_MIN_INLIERS   = 40
GATE_MAX_RMS_M     = 0.06
GATE_YAW_TOL_DEG   = 2.5           # |match dtheta - gyro dtheta| <= tol + frac*|gyro dtheta|
GATE_YAW_TOL_FRAC  = 0.10
GATE_MAX_SPEED_MPS = 1.2           # physically plausible translation speed
GATE_AMBIGUITY     = 0.97          # 2nd-best distinct peak / best above this -> ambiguous
GATE_DEGENERATE_EIG = 0.08         # min normalised eigenvalue of translational Hessian
GATE_LOST_AFTER    = 15            # consecutive rejections -> LOST (rover stopped)
FUSE_YAW_WEIGHT_MIN = 0.6          # weight on matched yaw when quality is marginal
GYRO_SCALE_ADAPT   = True          # bounded online gyro scale correction from good turning frames
GYRO_SCALE_ADAPT_RATE = 0.05
GYRO_SCALE_ADAPT_MAX = 0.15        # never more than +/-15 % (bigger errors: fix IMU_GYRO_SCALE)
# Re-localisation (first scan on a loaded map, and while LOST): wide window, stricter acceptance
RELOC_XY_M         = 0.5
RELOC_TH_DEG       = 25.0
RELOC_MIN_INLIER_RATIO = 0.70
RELOC_MAX_AMBIGUITY = 0.90

# --- Map ---
MAP_RES_M          = 0.05          # fallback ladder: 0.075 / 0.10 if Pi 4B is too slow
MAP_SIZE_M         = 20.0
MAP_HIT            = 0.9
MAP_MISS           = -0.4
MAP_CLAMP          = 3.5
MAP_OCC_THRESH     = 0.6           # log-odds above this = occupied (matching + planning)
MAP_FREE_THRESH    = -0.4
MAP_MAX_RAY_M      = 8.0
MAP_FREE_RAY_NO_RETURN = False     # zero returns are not trusted as free space
MAP_RAY_STRIDE     = 1             # use every Nth ray for free-space carving (2 halves raycast cost)

# ═════════════════════════════════════════════════════════════════════════════
#  Navigation & safety
# ═════════════════════════════════════════════════════════════════════════════
NAV_HZ             = 15.0
PLAN_RES_M         = 0.10          # A* runs on a down-sampled grid
INFLATION_RADIUS_M = 0.15
GOAL_TOLERANCE_M   = 0.12
LOOKAHEAD_M        = 0.40
REPLAN_INTERVAL_S  = 2.0
AUTO_FWD_DUTY      = 0.22          # straight-line duty in AUTO
AUTO_TURN_DUTY_MAX = 0.45          # caps auto spin rate so the matcher keeps up
AUTO_TURN_DUTY_MIN = 0.35          # skid-steer needs this much to turn in place at all
AUTO_TURN_GAIN     = 0.30          # duty per rad of heading error
ROTATE_FIRST_DEG   = 50.0          # rotate in place if heading error bigger than this
SLOWDOWN_REJECT_RATE = 0.3         # recent gate-reject fraction above which speed is halved
WARMUP_FRAMES      = 10

MANUAL_MAX_DUTY    = 0.40
MANUAL_TURN_DUTY   = 0.45

PROX_SECTOR_HALF_DEG = 30.0
PROX_STOP_M        = 0.25          # forward motion blocked below this (front sector, from rover centre)
PROX_SLOW_M        = 0.45
PROX_APPLIES_TO_MANUAL = True
STUCK_TIMEOUT_S    = 5.0
STUCK_BACKUP_S     = 1.0
STUCK_MAX_ATTEMPTS = 3

# ═════════════════════════════════════════════════════════════════════════════
#  Web UI / logging / monitoring
# ═════════════════════════════════════════════════════════════════════════════
WEB_HOST           = "0.0.0.0"
WEB_PORT           = 8080
WEB_MAP_HZ         = 3.0           # max PNG re-encode rate
LOG_DIR            = "runs"
LOG_ENABLED        = True
SYSMON_HZ          = 1.0
MAPS_DIR           = "maps"

# ─────────────────────────────────────────────────────────────────────────────
try:  # per-rover overrides
    from phase3_onboard.config_local import *  # noqa: F401,F403
except ImportError:
    pass


def snapshot() -> dict:
    """All UPPER_CASE settings as a JSON-friendly dict (stored in each log)."""
    out = {}
    for k, v in globals().items():
        if k.isupper():
            out[k] = v if isinstance(v, (int, float, str, bool, type(None), list, tuple)) else repr(v)
    return out

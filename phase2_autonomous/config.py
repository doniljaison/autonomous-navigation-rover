"""
Centralised Configuration — Edit this file to match your hardware
=================================================================
All rover, LiDAR, and navigation settings live here.
Every other module imports from config instead of hard-coding values.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Hardware connections  — CHANGE THESE TO MATCH YOUR SETUP
# ─────────────────────────────────────────────────────────────────────────────

# LiDAR serial port  (check Device Manager on Windows)
LIDAR_PORT = "COM5"           # Windows: "COM5"  |  Linux: "/dev/ttyUSB0"

# Waveshare rover WiFi
ROVER_IP   = "10.109.154.53"  # rover's IP on your network
ROVER_PORT = 80
ROVER_ENDPOINT = "/js"        # Waveshare JSON command endpoint

# ─────────────────────────────────────────────────────────────────────────────
#  SLAM parameters
# ─────────────────────────────────────────────────────────────────────────────

MIN_RANGE_MM  = 150           # ignore readings closer than this (mm)
MAX_RANGE_MM  = 8000          # LiDAR effective range (mm)
MAP_RES_M     = 0.05          # metres per grid cell (5 cm)
MAP_SIZE_M    = 20.0          # map side length in metres
ICP_MAX_ITER  = 30            # ICP iterations per frame
ICP_TOLERANCE = 1e-5          # ICP convergence threshold
ICP_MAX_DIST  = 0.30          # max ICP correspondence distance (m) — tighter = less drift
MIN_POINTS    = 30            # skip frame if fewer scan points
FPS_WINDOW    = 30            # rolling-average FPS window
TRAIL_LEN     = 500           # robot path history length

# ── IMU fusion ────────────────────────────────────────────────────────────────
IMU_ALPHA     = 0.7           # complementary filter: 0.7 = trust ICP 70%, IMU 30%
IMU_POLL_HZ   = 10.0          # how often to poll IMU from rover (Hz)

# ── Obstacle / proximity thresholds ──────────────────────────────────────────
OBSTACLE_WARN_M  = 0.30       # yellow alert (metres) — slow down
OBSTACLE_CRIT_M  = 0.12       # red alert (metres) — stop
SECTOR_HALF_DEG  = 25         # sector half-width (degrees)

# ─────────────────────────────────────────────────────────────────────────────
#  Navigation parameters
# ─────────────────────────────────────────────────────────────────────────────

MAX_LINEAR_SPEED   = 0.45     # m/s cap for motor commands
MAX_ANGULAR_SPEED  = 1.5      # rad/s cap for turning
LOOKAHEAD_M        = 0.50     # pure-pursuit lookahead distance (m)
GOAL_TOLERANCE_M   = 0.30     # distance to consider goal reached (m)
INFLATION_RADIUS_M = 0.10     # obstacle inflation for A* planning (m) — smaller for indoor
REPLAN_INTERVAL_S  = 3.0      # replan path every N seconds
STUCK_TIMEOUT_S    = 6.0      # declare stuck after N seconds without movement
STUCK_ESCAPE_S     = 1.5      # spin duration when stuck

# ─────────────────────────────────────────────────────────────────────────────
#  Manual control
# ─────────────────────────────────────────────────────────────────────────────

MANUAL_SPEED  = 0.45          # max speed in manual mode
MANUAL_TURN   = 0.45          # turn sensitivity (keyboard is binary, so go full)
AUTO_TURN_SCALE = 0.35        # how aggressively auto mode turns (same as gamepad)
CMD_HZ        = 30.0          # motor command rate (Hz)
NAV_HZ        = 20.0          # autonomous nav loop rate (Hz)
VIS_INTERVAL  = 67            # visualiser refresh interval (ms)
WARMUP_FRAMES = 15            # wait for SLAM to stabilise before allowing goals

# ─────────────────────────────────────────────────────────────────────────────
#  Gamepad  (Xbox layout — adjust axis indices for your controller)
# ─────────────────────────────────────────────────────────────────────────────

JOYSTICK_INDEX  = 0
AXIS_TURN       = 0           # left stick X
AXIS_FORWARD    = 5           # right trigger
AXIS_BACKWARD   = 2           # left trigger
ENABLE_BUTTON   = 4           # LB bumper
REQUIRE_ENABLE  = True        # must hold LB to drive
TRIGGER_MODE    = "1_to_-1"   # "1_to_-1" (Linux) or "0_to_1" (Windows/XInput)
DEADZONE        = 0.05

# ─────────────────────────────────────────────────────────────────────────────
#  Wheel geometry (for differential drive mixing)
# ─────────────────────────────────────────────────────────────────────────────

WHEEL_BASE_M = 0.17           # centre-to-centre wheel distance (metres)
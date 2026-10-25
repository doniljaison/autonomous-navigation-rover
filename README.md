# Autonomous Navigation Rover -- Intelligent and Autonomous Systems

A LiDAR-based autonomous mobile rover capable of real-time SLAM (Simultaneous Localization and Mapping), manual teleoperation via Xbox controller or keyboard, and autonomous point-to-point navigation through click-to-navigate and frontier-based exploration. Built entirely in Python without ROS dependencies for the core navigation stack.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Hardware](#hardware)
- [Software Architecture](#software-architecture)
  - [Phase 1 -- Manual Teleoperation and LiDAR Mapping](#phase-1----manual-teleoperation-and-lidar-mapping)
  - [Phase 2 -- Autonomous Navigation](#phase-2----autonomous-navigation)
- [Module Reference](#module-reference)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Running Manual Teleoperation (Phase 1)](#running-manual-teleoperation-phase-1)
  - [Running Hybrid Control (Phase 2)](#running-hybrid-control-phase-2)
  - [Running Autonomous Frontier Exploration (Phase 2)](#running-autonomous-frontier-exploration-phase-2)
- [Controls](#controls)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
  - [SLAM Pipeline](#slam-pipeline)
  - [Path Planning](#path-planning)
  - [Path Tracking](#path-tracking)
  - [Obstacle Avoidance](#obstacle-avoidance)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project was developed as part of the **Intelligent and Autonomous Systems (IAS)** course. The objective was to build a mobile rover system that can:

1. Be manually controlled via an Xbox gamepad or keyboard.
2. Build a real-time 2D occupancy grid map of its environment using LiDAR.
3. Autonomously navigate from its current position to a user-specified goal point on the map.
4. Autonomously explore unmapped areas using frontier-based exploration.

The project was executed in two phases:

- **Phase 1 (Mid-Evaluation):** Manual teleoperation of the rover using an Xbox controller, with simultaneous LiDAR-based SLAM for real-time mapping and localization.
- **Phase 2 (Final Evaluation):** Full autonomous navigation where a user clicks a destination on the live map and the rover plans and follows a collision-free path to reach it, along with a fully autonomous frontier exploration mode.

---

## Hardware

| Component         | Specification                                                    |
|-------------------|------------------------------------------------------------------|
| **Rover Base**    | Waveshare 4WD Mobile Rover, differential drive                   |
| **Driver Board**  | ESP32 (Waveshare General Driver for Robots)                      |
| **Compute**       | Host PC (Windows/Linux); Raspberry Pi 5 migration in progress    |
| **LiDAR**         | 2D YDLiDAR G4 (360-degree, connected via USB)                    |
| **IMU**           | 9-axis IMU onboard the ESP32, polled over HTTP / UART            |
| **Communication** | WiFi (HTTP JSON commands) or wired UART (115200 baud, pyserial)  |
| **Wheel Base**    | 0.17 m (centre-to-centre wheel distance)                         |

The rover accepts JSON motor commands over HTTP:
```
GET http://<ROVER_IP>/js?json={"T":1,"L":<left_speed>,"R":<right_speed>}
```
where `L` and `R` are floating-point wheel speeds (e.g., -0.45 to +0.45 m/s).

---

## Software Architecture

The entire core navigation and SLAM stack is written in **pure Python** with no ROS dependency. This was a deliberate design choice to maintain a lean, custom, low-latency control pipeline.

### Phase 1 -- Manual Teleoperation and LiDAR Mapping

Located in the **`phase1_teleop/`** directory. Provides:

- **Xbox controller teleoperation** via ROS2 `joy` node for gamepad input, translated to HTTP motor commands.
- **Real-time ICP-SLAM** using the YDLiDAR G4 sensor, with a live Matplotlib dashboard showing the occupancy map, polar scan, robot pose, ICP error, and system statistics.
- **Simulation mode** for testing the SLAM pipeline without hardware.

### Phase 2 -- Autonomous Navigation

Located in the **`phase2_autonomous/`** directory. Extends Phase 1 with:

- **Click-to-navigate:** Click any point on the live occupancy map and the rover autonomously drives there.
- **A\* path planning** on the inflated occupancy grid with 8-connected movement.
- **Pure Pursuit and Rotate-and-Drive controllers** for smooth path following.
- **IMU-ICP complementary filter fusion** for improved heading estimation.
- **Multi-sector obstacle proximity detection** with graduated alerts (clear/warning/critical).
- **Stuck detection and escape manoeuvres** (reverse, spin) when the rover is unable to make progress.
- **Frontier-based autonomous exploration** that detects boundaries between known and unknown space, plans paths to them, and systematically explores the entire reachable environment.
- **Map save/resume** via `.npz` files for persistent mapping sessions.

---

## Module Reference

### `phase1_teleop/` Directory (Phase 1)

| File                    | Description                                                                                   |
|-------------------------|-----------------------------------------------------------------------------------------------|
| `joy_rover_http.py`     | ROS2 node that subscribes to Xbox controller input and sends HTTP motor commands to the rover. Uses a `PreconnectPool` for low-latency TCP communication. |
| `joy_rover_launch.py`   | ROS2 launch file that starts both the `joy` node (gamepad driver) and `joy_rover_http` node.  |
| `LiDAR.py`              | Standalone ICP-SLAM implementation with occupancy grid mapping, a simulated LiDAR for offline testing, and a real-time Matplotlib dashboard. |

### `phase2_autonomous/` Directory (Phase 2)

| File         | Description                                                                                          |
|--------------|------------------------------------------------------------------------------------------------------|
| `config.py`  | Centralised configuration for all hardware, SLAM, navigation, and control parameters. All other modules import from here. |
| `slam.py`    | Production SLAM engine with ICP scan-matching, occupancy grid mapping, IMU fusion via complementary filter, multi-sector proximity detection, and a real-time Matplotlib visualiser. No simulation fallback. |
| `nav.py`     | Navigation utilities: A\* planner with obstacle inflation, `RotateAndDriveController` for path tracking, `PreconnectPool` for motor communication, and HTTP request helpers. |
| `hybrid.py`  | **Main entry point.** Hybrid manual + autonomous control. Supports keyboard (WASD), Xbox gamepad, and click-to-navigate on the live map. Runs SLAM, IMU polling, manual control, and autonomous navigation as concurrent threads. |
| `auto.py`    | Fully autonomous frontier-based exploration. Detects frontiers (boundaries of explored/unexplored space), plans paths via A\*, follows them with Pure Pursuit, and handles obstacle stops and stuck-escape recovery through a finite-state machine. |

### `docs/` Directory

| File                     | Description                                                                     |
|--------------------------|---------------------------------------------------------------------------------|
| `Commands.txt`           | Quick-reference commands for fetching IMU data and starting the controller nodes. |
| `system description.txt` | Detailed system architecture notes, known issues, and planned improvements.     |

---

## Prerequisites

- Python 3.10 or later
- A YDLiDAR G4 (or compatible) sensor connected via USB
- A Waveshare 4WD rover on the same WiFi network (or connected via UART)
- An Xbox-compatible gamepad (optional; keyboard control is always available)

---

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/IAS-Autonomous-Rover.git
   cd IAS-Autonomous-Rover
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Phase 1 only) Install ROS2 and the `joy` package if using the Xbox controller via ROS2:
   ```bash
   sudo apt install ros-<distro>-joy
   ```

4. Configure hardware parameters in `phase2_autonomous/config.py`:
   - Set `LIDAR_PORT` to match your LiDAR's serial port (e.g., `COM5` on Windows, `/dev/ttyUSB0` on Linux).
   - Set `ROVER_IP` to your rover's IP address on the network.

5. On Linux, add your user to the `dialout` group for serial port access:
   ```bash
   sudo usermod -aG dialout $USER
   ```
   Log out and back in for the change to take effect.

---

## Usage

### Running Manual Teleoperation (Phase 1)

Start the LiDAR SLAM dashboard (standalone, no rover control):
```bash
cd phase1_teleop
python LiDAR.py
```

Start the Xbox controller teleoperation (requires ROS2):
```bash
# Terminal 1
ros2 run joy joy_node --ros-args -p deadzone:=0.05

# Terminal 2
cd phase1_teleop
python3 joy_rover_http.py
```

### Running Hybrid Control (Phase 2)

This is the primary mode for operating the rover with both manual and autonomous capabilities:
```bash
cd phase2_autonomous
python hybrid.py
```

To resume from a previously saved map:
```bash
python hybrid.py --map mymap.npz
```

### Running Autonomous Frontier Exploration (Phase 2)

For fully autonomous exploration without manual intervention:
```bash
cd phase2_autonomous
python auto.py
```

---

## Controls

When running `hybrid.py`:

| Input                    | Action                                         |
|--------------------------|-------------------------------------------------|
| `W` / Arrow Up           | Drive forward                                  |
| `S` / Arrow Down         | Drive backward                                 |
| `A` / Arrow Left         | Turn left                                      |
| `D` / Arrow Right        | Turn right                                     |
| `Space`                  | Emergency stop / cancel autonomous navigation  |
| `M`                      | Save current map to `.npz` file                |
| `Escape`                 | Shut down and exit                             |
| Left-click on map        | Set autonomous navigation goal                 |
| Xbox LB + Triggers/Stick | Manual driving (gamepad)                       |

Manual input (keyboard or gamepad) automatically overrides and cancels any active autonomous navigation.

---

## Configuration

All tunable parameters are centralised in `phase2_autonomous/config.py`:

| Parameter              | Default   | Description                                        |
|------------------------|-----------|----------------------------------------------------|
| `MAP_RES_M`            | 0.05 m    | Occupancy grid resolution (metres per cell)        |
| `MAP_SIZE_M`           | 20.0 m    | Total map side length                              |
| `ICP_MAX_ITER`         | 30        | Maximum ICP iterations per scan frame              |
| `ICP_MAX_DIST`         | 0.30 m    | Maximum correspondence distance for ICP matching   |
| `IMU_ALPHA`            | 0.7       | Complementary filter weight (0.7 = 70% ICP trust)  |
| `MAX_LINEAR_SPEED`     | 0.45 m/s  | Maximum rover forward/reverse speed                |
| `MAX_ANGULAR_SPEED`    | 1.5 rad/s | Maximum turning speed                              |
| `INFLATION_RADIUS_M`   | 0.10 m    | Obstacle inflation for A\* planning                |
| `GOAL_TOLERANCE_M`     | 0.30 m    | Distance threshold to consider a goal reached      |
| `STUCK_TIMEOUT_S`      | 6.0 s     | Time without movement before stuck-escape triggers |
| `OBSTACLE_WARN_M`      | 0.30 m    | Warning proximity threshold                        |
| `OBSTACLE_CRIT_M`      | 0.12 m    | Critical proximity threshold (emergency stop)      |

---

## How It Works

### SLAM Pipeline

1. **Scan Acquisition:** Raw 360-degree distance readings are obtained from the YDLiDAR G4 and converted from polar (angle, distance) to Cartesian (x, y) coordinates, filtering out invalid readings outside the 0.15--8.0 m range.

2. **ICP Scan Matching:** Each new scan is aligned against the previous scan using the Iterative Closest Point (ICP) algorithm. A KD-Tree accelerates nearest-neighbour queries. The SVD-based Procrustes solution computes the optimal rigid-body transformation (translation + rotation) between consecutive scans.

3. **IMU Fusion:** The heading estimate from ICP is blended with the IMU's absolute yaw reading using a complementary filter (configurable weight via `IMU_ALPHA`), correcting for ICP drift over time.

4. **Occupancy Grid Mapping:** Each scan is projected into a global 2D occupancy grid using Bresenham ray-casting. Cells along each ray are marked as free (log-odds decremented), and the endpoint cell is marked as occupied (log-odds incremented). Log-odds are clamped to prevent over-confidence.

### Path Planning

The A\* planner operates on the occupancy grid with 8-connected movement. Obstacles are inflated by a configurable radius to ensure the planned path maintains safe clearance from walls and objects. If planning fails with inflation, the planner retries without inflation as a fallback.

### Path Tracking

Two controllers are available:

- **Rotate-and-Drive** (`nav.py`): Faces each waypoint by spinning in place, then drives straight to it. Simple and reliable.
- **Pure Pursuit** (`auto.py`): Computes smooth steering commands using a lookahead point on the planned path, allowing the rover to follow curved trajectories without stopping at each waypoint.

### Obstacle Avoidance

The proximity detector partitions the scan into four sectors (front, left, right, rear) and computes the minimum distance in each. Three alert levels are used:

- **Clear** (green): No obstacles within 0.30 m. Full speed.
- **Warning** (yellow): Obstacle within 0.30 m. Speed reduced to 40%.
- **Critical** (red): Obstacle within 0.12 m. Emergency stop, followed by escape manoeuvre (reverse or spin away from the closest obstacle).

---

## Project Structure

```
IAS/
├── .gitignore
├── requirements.txt
├── README.md
│
├── docs/
│   ├── Commands.txt               # Quick-reference commands
│   └── system description.txt     # Detailed system architecture notes
│
├── phase1_teleop/                 # Phase 1: Manual teleoperation & mapping
│   ├── LiDAR.py                   # Standalone SLAM with simulated/real LiDAR
│   ├── joy_rover_http.py          # ROS2 Xbox controller to rover HTTP bridge
│   └── joy_rover_launch.py        # ROS2 launch file for controller setup
│
└── phase2_autonomous/             # Phase 2: Autonomous navigation
    ├── config.py                  # Centralised configuration
    ├── slam.py                    # SLAM engine (ICP + occupancy grid + IMU fusion)
    ├── nav.py                     # A* planner, path controllers, motor pool
    ├── hybrid.py                  # Main entry: manual + click-to-navigate + SLAM
    └── auto.py                    # Autonomous frontier-based exploration
```

---

## Acknowledgements

This project was completed as a group effort for the Intelligent and Autonomous Systems course.

import math

IMU_GYRO_SIGN  = +1
IMU_GYRO_SCALE = math.radians(1.0) / 1.0028   # 0.0174043 rad/s per raw unit
IMU_YAW_SIGN   = +1

LIDAR_MOUNT_X_M     = 0.00
LIDAR_MOUNT_Y_M     = 0.00
LIDAR_MOUNT_YAW_RAD = 0.0

MANUAL_TURN_DUTY   = 0.45
AUTO_TURN_DUTY_MAX = 0.45
AUTO_TURN_DUTY_MIN = 0.35

ROBOT_FRONT_M      = 0.10   # LiDAR centre -> front-most point (bumper, payload)
ROBOT_BACK_M       = 0.10
ROBOT_HALF_WIDTH_M = 0.15   # centre line -> outside of the wheels
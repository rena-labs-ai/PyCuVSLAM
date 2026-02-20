import math
import time
from typing import Optional, List, Tuple

import cuvslam as vslam


DEFAULT_ODOM_LOG_PATH = "/mnt/jetson_data/rena-control/odom.log"
DEFAULT_ODOM_LOG_INTERVAL_S = 2.0


def quat_to_rpy(
    qx: float, qy: float, qz: float, qw: float
) -> Tuple[float, float, float]:
    """Convert quaternion [x, y, z, w] to roll, pitch, yaw in radians."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def rpy_to_quat(
    roll: float, pitch: float, yaw: float
) -> Tuple[float, float, float, float]:
    """Convert roll, pitch, yaw (radians) to quaternion [x, y, z, w]."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


class Pose(vslam.Pose):
    """Pose in camera frame."""

    def __init__(self, pose: vslam.Pose) -> None:
        super().__init__(pose.rotation, pose.translation)

    def to_robot_frame(self) -> "Pose":
        """Convert the pose to robot frame.

        Returns:
            Pose in robot frame.
        """
        robot_roll, robot_pitch, robot_yaw = quat_to_rpy(
            self.rotation[0], self.rotation[1], self.rotation[2], self.rotation[3]
        )
        robot_pose = vslam.Pose(
            rpy_to_quat(robot_yaw, robot_roll, -robot_pitch),
            [self.translation[2], -self.translation[0], self.translation[1]],
        )
        return Pose(robot_pose)


class Landmark:
    """Landmark in camera frame."""

    def __init__(self, id: int, coords: List[float]) -> None:
        self.id = id
        self.coords = coords

    def to_robot_frame(self) -> "Landmark":
        """Convert the landmark to robot frame."""
        robot = object.__new__(Landmark)
        robot.id = self.id
        robot.coords = [self.coords[2], -self.coords[0], -self.coords[1]]
        return robot


class OdomLogger:
    """Periodically logs odometry pose (in robot frame) to a file.

    Camera-to-robot frame convention:
        robot_x =  odom_z
        robot_y = -odom_x
        robot_z =  odom_y
    """

    def __init__(
        self,
        log_path: str = DEFAULT_ODOM_LOG_PATH,
        interval_s: float = DEFAULT_ODOM_LOG_INTERVAL_S,
    ) -> None:
        self._log_path = log_path
        self._interval_s = interval_s
        self._last_log_time = 0.0
        self._file = open(self._log_path, "a")

    @staticmethod
    def _yaw_from_quaternion(rotation) -> float:
        """Extract yaw (degrees) from a camera-frame quaternion [qx, qy, qz, qw]."""
        siny_cosp = 2.0 * (rotation[3] * rotation[2] + rotation[0] * rotation[1])
        cosy_cosp = 1.0 - 2.0 * (rotation[1] * rotation[1] + rotation[2] * rotation[2])
        return math.degrees(math.atan2(siny_cosp, cosy_cosp))

    def log(
        self,
        frame_id: int,
        vo_pose: Pose,
        slam_pose: Optional[Pose] = None,
    ) -> None:
        """Write an odom entry if enough time has elapsed since the last one.

        Args:
            frame_id: Current frame number.
            vo_pose: VO pose in camera frame.
            slam_pose: SLAM pose in camera frame.
        """
        now = time.time()
        if now - self._last_log_time < self._interval_s:
            return

        slam_pose = slam_pose.to_robot_frame()
        vo_pose = vo_pose.to_robot_frame()

        slam_str = (
            f"slam_pose: x:{slam_pose.translation[0]:.4f} y:{slam_pose.translation[1]:.4f} z:{slam_pose.translation[2]:.4f} yaw:{self._yaw_from_quaternion(slam_pose.rotation):.2f}deg"
            if slam_pose is not None
            else "slam_pose: None"
        )
        line = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"frame={frame_id} "
            f"vo_pose: x:{vo_pose.translation[0]:.4f} y:{vo_pose.translation[1]:.4f} z:{vo_pose.translation[2]:.4f} yaw:{self._yaw_from_quaternion(vo_pose.rotation):.2f}deg "
            f"{slam_str}"
        )

        self._file.write(line + "\n")
        self._file.flush()
        self._last_log_time = now

    def close(self) -> None:
        """Flush and close the log file."""
        self._file.close()

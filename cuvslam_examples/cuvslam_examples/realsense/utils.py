import math
import time
from typing import Optional, List, Tuple

import cuvslam as vslam


DEFAULT_ODOM_LOG_PATH = "./odom.log"
DEFAULT_ODOM_LOG_INTERVAL_S = 2.0


# cuVSLAM (OpenCV) -> canonical ROS; inverse of cuvslam_pose_canonical in
# isaac_ros_visual_slam/impl/cuvslam_ros_conversion.hpp (v_ros = T @ v_cuv).
_T_CUV_TO_ROS = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)


def _mat3_mul(
    a: Tuple[Tuple[float, ...], ...], b: Tuple[Tuple[float, ...], ...]
) -> Tuple[Tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _mat3_transpose(a: Tuple[Tuple[float, ...], ...]) -> Tuple[Tuple[float, float, float], ...]:
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))


def _quat_to_rotmat(
    qx: float, qy: float, qz: float, qw: float
) -> Tuple[Tuple[float, float, float], ...]:
    """Column-vector convention: v' = R @ v."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def _rotmat_to_quat(
    r: Tuple[Tuple[float, ...], ...],
) -> Tuple[float, float, float, float]:
    """Rotation matrix (row-major rows) to quaternion (qx, qy, qz, qw)."""
    m = r
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        s = 0.5 / math.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (m[2][1] - m[1][2]) * s
        qy = (m[0][2] - m[2][0]) * s
        qz = (m[1][0] - m[0][1]) * s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
        qw = (m[2][1] - m[1][2]) / s
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
        qw = (m[0][2] - m[2][0]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s
        qw = (m[1][0] - m[0][1]) / s
    return qx, qy, qz, qw


def _cuv_to_ros_rotation(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float, float]:
    """R_ros = T @ R_cuv @ T.T with Isaac canonical T (cuVSLAM -> ROS)."""
    t = _T_CUV_TO_ROS
    r_cuv = _quat_to_rotmat(qx, qy, qz, qw)
    r_ros = _mat3_mul(t, _mat3_mul(r_cuv, _mat3_transpose(t)))
    return _rotmat_to_quat(r_ros)


class Pose(vslam.Pose):
    """Pose in camera frame."""

    def __init__(self, pose: vslam.Pose) -> None:
        super().__init__(pose.rotation, pose.translation)

    def to_robot_frame(self) -> "Pose":
        """Convert pose from cuVSLAM (OpenCV) camera frame to canonical ROS frame.

        Matches isaac_ros_visual_slam ``canonical_pose_cuvslam``: translation
        ``(z, -x, -y)`` and rotation ``R_ros = T R_cuv T^T``.
        """
        qx, qy, qz, qw = self.rotation
        tx, ty, tz = self.translation[0], self.translation[1], self.translation[2]
        robot_rot = _cuv_to_ros_rotation(qx, qy, qz, qw)
        robot_pose = vslam.Pose(robot_rot, [tz, -tx, -ty])
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

    Camera-to-robot matches Isaac ``canonical_pose_cuvslam``:
        robot_x =  odom_z
        robot_y = -odom_x
        robot_z = -odom_y
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
        """Extract yaw (degrees) from a canonical-ROS quaternion [qx, qy, qz, qw]."""
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

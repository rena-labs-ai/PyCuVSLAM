"""ROS2 node that runs cuVSLAM and publishes odometry to /cuvslam/odometry.

Supports RosMulticamTracker, RosZedStereoTracker, RosZedVIOTracker,
RosRealSenseStereoTracker, RosHawkStereoTracker, and RosOakStereoTracker.
Stereo image topics are derived from parameter camera (model id).

Publishes only Odometry (no TF). Stamps match the tracker pipeline timestamp
(e.g. OAK: synced left IR header time from ApproximateTimeSynchronizer).
"""

import subprocess
import threading
import time
import yaml

import cuvslam as vslam
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseWithCovariance,
    Quaternion,
    TwistWithCovariance,
)
from nav_msgs.msg import Odometry
from rena_msgs.srv import LocalizeInMap, SaveSlamMap

from cuvslam_examples.realsense.pipeline import Pipeline
from cuvslam_examples.realsense.tracker import RosRealsenseStereoTracker, RosRealsenseRGBDTracker
from cuvslam_examples.zed.tracker import RosZedStereoTracker, RosZedVIOTracker
from cuvslam_examples.hawk.tracker import RosHawkMulticamTracker, RosHawkStereoTracker
from cuvslam_examples.oak.tracker import RosOakRGBDTracker, RosOakStereoTracker

ODOM_TOPIC = "/cuvslam/odometry"
ODOM_FRAME = "odom"

# Default ROS topic namespace (prefix) per RealSense camera model.
# Trackers append "camera/infra1/..." etc., so the base must not include
# the trailing /camera segment (matches the multi-cam convention).
_REALSENSE_TOPIC_BASE_DEFAULT = {
    "realsensed435": "/realsensed435_base",
    "realsensed455": "/realsensed455_base",
}


def _zed_stem(camera: str) -> str:
    c = camera.lower().strip()
    if c in ("zed", "zedm", "zed2i"):
        return f"/{c}_base/zed_node"
    return "/zed/zed_node"


def _realsense_topic_base(camera: str) -> str:
    c = camera.lower().strip()
    if c not in _REALSENSE_TOPIC_BASE_DEFAULT:
        raise ValueError(
            f"Unknown RealSense camera model {camera!r}; expected one of "
            f"{sorted(_REALSENSE_TOPIC_BASE_DEFAULT)}"
        )
    return _REALSENSE_TOPIC_BASE_DEFAULT[c]


def _stamp_from_ns(timestamp_ns: int) -> Time:
    t = Time()
    t.sec = int(timestamp_ns // 1_000_000_000)
    t.nanosec = int(timestamp_ns % 1_000_000_000)
    return t


def main() -> None:
    rclpy.init()

    param_node = Node("vslam")
    config_file_param = param_node.declare_parameter("config_file", "")
    enable_viz_param = param_node.declare_parameter("enable_visualization", False)
    tracker_param = param_node.declare_parameter("tracker", "ros_multicam")
    camera_param = param_node.declare_parameter("camera", "zed2i")
    rig_base_path_param = param_node.declare_parameter("rig_base_path", "")
    odom_child_frame_param = param_node.declare_parameter("odom_child_frame", "nav_base_link")

    config_file = config_file_param.value
    tracker_type = str(tracker_param.value)
    rig_base_path = str(rig_base_path_param.value)
    camera_model = str(camera_param.value)
    odom_child_frame = str(odom_child_frame_param.value)

    # Per-tracker rig file lives at {rig_base_path}/<model>_rig.yaml
    # (e.g. hawk_rig.yaml).
    def _rig_file(model: str) -> str:
        if not rig_base_path:
            return ""
        import os as _os
        return _os.path.join(rig_base_path, f"{model}_rig.yaml")

    if tracker_type == "ros_multicam" and not config_file:
        param_node.get_logger().error(
            "config_file parameter is required for ros_multicam"
        )
        param_node.destroy_node()
        rclpy.shutdown()
        return
    if tracker_type == "ros_hawk_multicam" and not _rig_file("hawk"):
        param_node.get_logger().error(
            "rig_base_path must be set for ros_hawk_multicam (expects <path>/hawk_rig.yaml)"
        )
        param_node.destroy_node()
        rclpy.shutdown()
        return

    enable_viz = (
        enable_viz_param.value
        if isinstance(enable_viz_param.value, bool)
        else str(enable_viz_param.value).lower() == "true"
    )
    param_node.destroy_node()

    zed_stem = _zed_stem(camera_model)
    zed_left = f"{zed_stem}/left/color/rect/image/compressed"
    zed_right = f"{zed_stem}/right/color/rect/image/compressed"
    zed_imu = f"{zed_stem}/imu/data"
    hawk_left = "/left/image_rect"
    hawk_right = "/right/image_rect"
    # OAK topics are derived from rena_bringup/config/config.yaml inside the
    # RosOakStereoTracker itself (serial + image_mode → image_raw | image_rect).

    # For ros_hawk_multicam, launch compressed→raw republishers for every topic in the rig.
    # image_transport republish subscribes to <topic>/compressed and publishes <topic> (raw).
    republisher_procs: list[subprocess.Popen] = []
    if tracker_type == "ros_hawk_multicam":
        with open(_rig_file("hawk")) as f:
            rig_data = yaml.safe_load(f)
        for cam in rig_data.get("stereo_cameras", []):
            for topic in (cam["left_topic"], cam["right_topic"]):
                cmd = [
                    "ros2", "run", "image_transport", "republish",
                    "compressed", "raw",
                    "--ros-args",
                    "-r", f"in/compressed:={topic}/compressed",
                    "-r", f"out:={topic}",
                ]
                republisher_procs.append(subprocess.Popen(cmd))

    match tracker_type:
        case "ros_hawk_stereo":
            tracker = RosHawkStereoTracker(left_topic=hawk_left, right_topic=hawk_right)
        case "ros_hawk_multicam":
            tracker = RosHawkMulticamTracker(rig_file=_rig_file("hawk"))
        case "ros_oak_stereo":
            tracker = RosOakStereoTracker()
        case "ros_oak_rgbd":
            tracker = RosOakRGBDTracker()
        case "ros_zed_stereo":
            tracker = RosZedStereoTracker(left_topic=zed_left, right_topic=zed_right)
        case "ros_zed_vio":
            tracker = RosZedVIOTracker(
                left_topic=zed_left,
                right_topic=zed_right,
                imu_topic=zed_imu,
                camera=camera_model,
            )
        case "ros_realsense_stereo":
            tracker = RosRealsenseStereoTracker(
                topic_base=_realsense_topic_base(camera_model),
            )
        case "ros_realsense_rgbd":
            tracker = RosRealsenseRGBDTracker(
                topic_base=_realsense_topic_base(camera_model),
            )
        case _:
            raise ValueError(f"Unknown tracker type: {tracker_type}")

    class VslamNode(Node):
        def __init__(self, child_frame: str) -> None:
            super().__init__("vslam")
            self._child_frame = child_frame
            self._odom_pub = self.create_publisher(Odometry, ODOM_TOPIC, 10)
            self._pipeline = Pipeline(tracker, enable_visualization=enable_viz)
            self._pipeline.start()

            self._latest_images = None
            self._latest_images_lock = threading.Lock()

            self._save_map_srv = self.create_service(
                SaveSlamMap, "/vslam/save_map", self._on_save_map
            )
            self._localize_srv = self.create_service(
                LocalizeInMap, "/vslam/localize_in_map", self._on_localize_in_map
            )

            self.get_logger().info(
                f"Publishing odometry only on {ODOM_TOPIC} "
                f"(child_frame={child_frame}, no TF). "
                f"Services: /vslam/save_map, /vslam/localize_in_map"
            )
            self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self._thread.start()

        def _tracking_loop(self) -> None:
            while rclpy.ok():
                result = self._pipeline.get(timeout=1.0)
                if result is None or result.slam_pose is None:
                    continue
                if result.images is not None:
                    with self._latest_images_lock:
                        self._latest_images = result.images
                stamp = _stamp_from_ns(result.timestamp)
                pose_robot = result.slam_pose.to_robot_frame()
                t = pose_robot.translation
                r = pose_robot.rotation

                msg = Odometry()
                msg.header.stamp = stamp
                msg.header.frame_id = ODOM_FRAME
                msg.child_frame_id = self._child_frame
                msg.pose = PoseWithCovariance()
                msg.pose.pose = Pose()
                msg.pose.pose.position = Point(
                    x=float(t[0]), y=float(t[1]), z=float(t[2])
                )
                msg.pose.pose.orientation = Quaternion(
                    x=float(r[0]), y=float(r[1]), z=float(r[2]), w=float(r[3])
                )
                msg.twist = TwistWithCovariance()
                self._odom_pub.publish(msg)

        def _on_save_map(self, request, response):
            t = self._pipeline.tracker
            if t is None:
                response.success = False
                response.message = "tracker not ready"
                return response
            done = threading.Event()
            ok = [False]
            def cb(success: bool) -> None:
                ok[0] = bool(success)
                done.set()
            try:
                t.save_map(request.folder_path, cb)
            except Exception as exc:
                response.success = False
                response.message = f"save_map raised: {exc}"
                return response
            if not done.wait(timeout=120.0):
                response.success = False
                response.message = "save_map timed out after 120s"
                return response
            response.success = ok[0]
            response.message = "saved" if ok[0] else "cuvslam reported failure"
            return response

        def _on_localize_in_map(self, request, response):
            t = self._pipeline.tracker
            if t is None:
                response.success = False
                response.message = "tracker not ready"
                return response
            with self._latest_images_lock:
                images = self._latest_images
            if images is None:
                response.success = False
                response.message = "no recent images from tracker; cannot relocalize"
                return response
            q = request.guess_pose.orientation
            p = request.guess_pose.position
            guess = vslam.Pose(
                rotation=[float(q.x), float(q.y), float(q.z), float(q.w)],
                translation=[float(p.x), float(p.y), float(p.z)],
            )
            done = threading.Event()
            result_pose = [None]
            err_msg = [""]
            def cb(pose, msg: str) -> None:
                result_pose[0] = pose
                err_msg[0] = msg or ""
                done.set()
            try:
                settings = vslam.Slam.LocalizationSettings()
                t.localize_in_map(request.folder_path, guess, images, settings, cb)
            except Exception as exc:
                response.success = False
                response.message = f"localize_in_map raised: {exc}"
                return response
            if not done.wait(timeout=60.0):
                response.success = False
                response.message = "localize_in_map timed out after 60s"
                return response
            response.success = result_pose[0] is not None
            response.message = err_msg[0] if not response.success else "localized"
            return response

        def destroy_node(self) -> None:
            self._pipeline.stop()
            super().destroy_node()

    node = VslamNode(child_frame=odom_child_frame)
    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        for proc in republisher_procs:
            proc.terminate()
        for proc in republisher_procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

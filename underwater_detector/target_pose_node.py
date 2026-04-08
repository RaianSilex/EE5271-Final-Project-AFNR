#!/usr/bin/env python3
"""
Target ArUco Pose Estimator
===========================
Listens for YOLO detections, finds ALL non-'LoCo' detections above the
confidence threshold, searches each bounding-box region for ArUco marker
ID 3, and publishes a PoseArray containing the 6-DOF pose of every target
whose marker is visible.  The motor-command node downstream can then pick
the closest pose from the array.

Marker convention
-----------------
  The marker defines the target frame directly:
    +X = marker right edge direction
    +Y = marker downward direction   (OpenCV ArUco convention)
    +Z = marker face normal (pointing away from the surface it is stuck on)

  Override 'marker_to_target_rpy_deg' (intrinsic XYZ Euler, degrees) to
  rotate from the raw marker frame into a different target body frame.

Topics
------
  Subscribes:
    /zed/zed_node/left/camera_info      (sensor_msgs/CameraInfo)
    /zed/zed_node/left/image_rect_color (sensor_msgs/Image)
    /detections/json                     (std_msgs/String)  – YOLO detections

  Publishes:
    /targets/poses      (geometry_msgs/PoseArray)  – one Pose per visible target
    /targets/pose_image (sensor_msgs/Image)        – visualisation with axes drawn

Parameters
----------
  marker_id                 int      default 3      ArUco ID affixed to every target
  marker_size               float    default 0.10   Printed marker side length in metres
  aruco_dict_id             int      default 0      cv2.aruco dict enum (0=DICT_4X4_50)
  conf_threshold            float    default 0.30   Min YOLO confidence to consider a detection
  marker_to_target_rpy_deg  list[3]  default [0,0,0]
      Intrinsic XYZ Euler (deg) rotating the raw marker frame into the
      desired target body frame.  Leave as zeros to publish the marker
      frame itself.
"""

import json

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from scipy.spatial.transform import Rotation as ScipyR
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

# The YOLO class name reserved for LoCo – excluded from target search
_LOCO_LABEL = 'LoCo'


class TargetPoseEstimator(Node):
    """ROS2 node that estimates all non-LoCo targets' 6-DOF poses via ArUco marker ID 3."""

    def __init__(self):
        super().__init__('target_pose_estimator')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('marker_id',   3)
        self.declare_parameter('marker_size', 0.10)
        self.declare_parameter('aruco_dict_id', 0)
        self.declare_parameter('conf_threshold', 0.30)
        self.declare_parameter('marker_to_target_rpy_deg', [0.0, 0.0, 0.0])

        self.marker_id   = self.get_parameter('marker_id').value
        self.marker_size = self.get_parameter('marker_size').value
        dict_id          = self.get_parameter('aruco_dict_id').value
        self.conf_thr    = self.get_parameter('conf_threshold').value
        rpy              = self.get_parameter('marker_to_target_rpy_deg').value
        self.R_target_from_marker = ScipyR.from_euler(
            'xyz', rpy, degrees=True).as_matrix()

        # ── ArUco detector ────────────────────────────────────────────────
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params     = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        # ── State ─────────────────────────────────────────────────────────
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs:   np.ndarray | None = None
        # All non-LoCo detections above threshold: list of {'bbox', 'label', 'confidence'}
        self.target_dets: list[dict] = []
        self.bridge = CvBridge()

        # ── Subscriptions ─────────────────────────────────────────────────
        self.create_subscription(
            CameraInfo, '/zedm/zed_node/left_raw/camera_info',
            self._camera_info_cb, 1)
        self.create_subscription(
            String, '/detections/json',
            self._detections_cb, 10)
        self.create_subscription(
            Image, '/zedm/zed_node/left_raw/image_raw_color',
            self._image_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_poses = self.create_publisher(PoseArray, '/targets/poses', 10)
        self.pub_vis   = self.create_publisher(Image, '/targets/pose_image', 10)

        self.get_logger().info(
            f'Target pose estimator ready | '
            f'marker_id={self.marker_id} | '
            f'marker_size={self.marker_size} m | '
            f'aruco_dict_id={dict_id} | '
            f'conf_threshold={self.conf_thr}')

    # ── Callbacks ────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=float).reshape(3, 3)
            self.dist_coeffs   = np.array(msg.d, dtype=float)
            self.get_logger().info('Camera intrinsics received.')

    def _detections_cb(self, msg: String):
        """Cache all non-LoCo detections above confidence threshold."""
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        self.target_dets = [
            det for det in detections
            if det.get('label') != _LOCO_LABEL
            and det.get('confidence', 0.0) >= self.conf_thr
        ]

    def _image_cb(self, msg: Image):
        """For each cached target detection, find ArUco ID 3 in its ROI and publish pose."""
        if self.camera_matrix is None or not self.target_dets:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        h, w = frame.shape[:2]
        vis = frame.copy()

        pose_array = PoseArray()
        pose_array.header          = msg.header
        pose_array.header.frame_id = 'zed_left_camera_optical_frame'

        vis_row_offset = 0  # stack per-target text lines at top of image

        for det in self.target_dets:
            x1, y1, x2, y2 = det['bbox']
            x1c, y1c = max(0, int(x1)), max(0, int(y1))
            x2c, y2c = min(w, int(x2)), min(h, int(y2))
            label = det.get('label', 'Target')

            # Draw the YOLO bounding box
            cv2.rectangle(vis, (x1c, y1c), (x2c, y2c), (0, 128, 255), 2)
            cv2.putText(vis, label, (x1c, y1c - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 128, 255), 2)

            roi = frame[y1c:y2c, x1c:x2c]
            if roi.size == 0:
                continue

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            corners_roi, ids, _ = self.detector.detectMarkers(gray_roi)

            if ids is None or len(ids) == 0:
                continue

            ids_flat = ids.flatten()
            matches = np.where(ids_flat == self.marker_id)[0]
            if len(matches) == 0:
                continue

            idx = matches[0]  # one marker per target

            # Shift ROI-local corners back to full-frame coordinates
            corners_full = corners_roi[idx].copy()
            corners_full[0, :, 0] += x1c
            corners_full[0, :, 1] += y1c

            # ── Pose of marker in camera frame ────────────────────────────
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners_full], self.marker_size,
                self.camera_matrix, self.dist_coeffs)
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]

            R_cam_marker, _ = cv2.Rodrigues(rvec)

            # ── Apply optional marker→target rotation ─────────────────────
            R_cam_target = R_cam_marker @ self.R_target_from_marker.T
            q       = ScipyR.from_matrix(R_cam_target).as_quat()  # [x, y, z, w]
            rpy_deg = ScipyR.from_matrix(R_cam_target).as_euler('xyz', degrees=True)

            # ── Append to PoseArray ───────────────────────────────────────
            pose = Pose()
            pose.position.x = float(tvec[0])
            pose.position.y = float(tvec[1])
            pose.position.z = float(tvec[2])
            pose.orientation.x = float(q[0])
            pose.orientation.y = float(q[1])
            pose.orientation.z = float(q[2])
            pose.orientation.w = float(q[3])
            pose_array.poses.append(pose)

            self.get_logger().info(
                f'Target ({label}) | '
                f'XYZ [{tvec[0]:+.3f}, {tvec[1]:+.3f}, {tvec[2]:+.3f}] m | '
                f'RPY [{rpy_deg[0]:+.1f}, {rpy_deg[1]:+.1f}, {rpy_deg[2]:+.1f}]°')

            # ── Visualisation ─────────────────────────────────────────────
            cv2.drawFrameAxes(
                vis, self.camera_matrix, self.dist_coeffs,
                rvec, tvec, self.marker_size * 0.6)
            cx = int(np.mean(corners_full[0, :, 0]))
            cy = int(np.mean(corners_full[0, :, 1]))
            cv2.putText(vis, f'ID{self.marker_id}', (cx, cy - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
            txt = (f'[{label}] '
                   f'X={tvec[0]:+.2f} Y={tvec[1]:+.2f} Z={tvec[2]:+.2f} m  '
                   f'R={rpy_deg[0]:+.0f} P={rpy_deg[1]:+.0f} Yaw={rpy_deg[2]:+.0f} deg')
            cv2.putText(vis, txt, (10, 30 + vis_row_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 128, 255), 2)
            vis_row_offset += 24

        if pose_array.poses:
            self.pub_poses.publish(pose_array)

        self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = TargetPoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

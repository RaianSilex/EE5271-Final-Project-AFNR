#!/usr/bin/env python3
"""
Target ArUco Pose Estimator
===========================
Searches the full camera frame for a single ArUco marker (ID 3) that is
affixed to a target object in the scene.  When the marker is detected the
node publishes the 6-DOF pose of the *target frame* (coincident with the
marker face) in the ZED left-camera optical frame.

Marker convention
-----------------
  The marker is treated as defining the target frame directly:
    +X = marker right edge direction
    +Y = marker downward direction   (OpenCV ArUco convention)
    +Z = marker face normal (pointing away from the surface it is stuck on)

  If you need a different target frame orientation you can override the
  'marker_to_target_rpy_deg' parameter (intrinsic XYZ Euler angles, degrees)
  to rotate from the raw marker frame into your desired target frame.

Topics
------
  Subscribes:
    /zed/zed_node/left/camera_info     (sensor_msgs/CameraInfo)
    /zed/zed_node/left/image_rect_color (sensor_msgs/Image)

  Publishes:
    /target/pose        (geometry_msgs/PoseStamped)  – 6-DOF pose in camera frame
    /target/pose_image  (sensor_msgs/Image)          – visualisation with axes drawn

Parameters
----------
  marker_id              int    default 3      ArUco ID of the target marker
  marker_size            float  default 0.10   Printed marker side length in metres
  aruco_dict_id          int    default 0      cv2.aruco dict enum (0=DICT_4X4_50)
  marker_to_target_rpy_deg  list[3]  default [0,0,0]
      Intrinsic XYZ Euler (deg) rotating the raw marker frame into the
      desired target body frame.  Leave as zeros to publish the marker
      frame itself.
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as ScipyR
from sensor_msgs.msg import CameraInfo, Image


class TargetPoseEstimator(Node):
    """ROS2 node that estimates a target object's 6-DOF pose via ArUco marker ID 3."""

    def __init__(self):
        super().__init__('target_pose_estimator')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('marker_id',   3)
        self.declare_parameter('marker_size', 0.10)
        self.declare_parameter('aruco_dict_id', 0)
        # Optional rigid offset from marker frame to target body frame
        self.declare_parameter('marker_to_target_rpy_deg', [0.0, 0.0, 0.0])

        self.marker_id   = self.get_parameter('marker_id').value
        self.marker_size = self.get_parameter('marker_size').value
        dict_id          = self.get_parameter('aruco_dict_id').value
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
        self.bridge = CvBridge()

        # ── Subscriptions ─────────────────────────────────────────────────
        self.create_subscription(
            CameraInfo, '/zed/zed_node/left/camera_info',
            self._camera_info_cb, 1)
        self.create_subscription(
            Image, '/zed/zed_node/left/image_rect_color',
            self._image_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_pose = self.create_publisher(PoseStamped, '/target/pose', 10)
        self.pub_vis  = self.create_publisher(Image, '/target/pose_image', 10)

        self.get_logger().info(
            f'Target pose estimator ready | '
            f'watching marker_id={self.marker_id} | '
            f'marker_size={self.marker_size} m | '
            f'aruco_dict_id={dict_id}')

    # ── Callbacks ────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=float).reshape(3, 3)
            self.dist_coeffs   = np.array(msg.d, dtype=float)
            self.get_logger().info('Camera intrinsics received.')

    def _image_cb(self, msg: Image):
        """Detect target marker in full frame and publish 6-DOF pose."""
        if self.camera_matrix is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self.detector.detectMarkers(gray)

        vis = frame.copy()

        if ids is None:
            self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))
            return

        ids_flat = ids.flatten()

        # Find our target marker among all detected markers
        matches = np.where(ids_flat == self.marker_id)[0]
        if len(matches) == 0:
            self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))
            return

        idx = matches[0]   # take the first (should be unique)
        corners = corners_list[idx]

        # ── Pose of marker in camera frame ────────────────────────────────
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners], self.marker_size,
            self.camera_matrix, self.dist_coeffs)
        rvec = rvecs[0][0]
        tvec = tvecs[0][0]

        R_cam_marker, _ = cv2.Rodrigues(rvec)

        # ── Apply optional marker→target rotation ─────────────────────────
        # T_cam←target = T_cam←marker * T_marker←target
        #   R_C_T = R_C_M @ R_M_T  (where R_M_T = R_target_from_marker.T)
        #   t stays the same (we treat the marker centre as the target origin)
        R_cam_target = R_cam_marker @ self.R_target_from_marker.T
        q = ScipyR.from_matrix(R_cam_target).as_quat()  # [x, y, z, w]
        rpy_deg = ScipyR.from_matrix(R_cam_target).as_euler('xyz', degrees=True)

        # ── Publish PoseStamped ───────────────────────────────────────────
        pose = PoseStamped()
        pose.header          = msg.header
        pose.header.frame_id = 'zed_left_camera_optical_frame'
        pose.pose.position.x = float(tvec[0])
        pose.pose.position.y = float(tvec[1])
        pose.pose.position.z = float(tvec[2])
        pose.pose.orientation.x = float(q[0])
        pose.pose.orientation.y = float(q[1])
        pose.pose.orientation.z = float(q[2])
        pose.pose.orientation.w = float(q[3])
        self.pub_pose.publish(pose)

        self.get_logger().info(
            f'Target | XYZ [{tvec[0]:+.3f}, {tvec[1]:+.3f}, {tvec[2]:+.3f}] m | '
            f'RPY [{rpy_deg[0]:+.1f}, {rpy_deg[1]:+.1f}, {rpy_deg[2]:+.1f}]°')

        # ── Visualisation ─────────────────────────────────────────────────
        cv2.drawFrameAxes(
            vis, self.camera_matrix, self.dist_coeffs,
            rvec, tvec, self.marker_size * 0.6)
        cx = int(np.mean(corners[0, :, 0]))
        cy = int(np.mean(corners[0, :, 1]))
        cv2.putText(vis, f'TARGET (ID {self.marker_id})', (cx, cy - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 255), 2)
        txt = (f'X={tvec[0]:+.2f} Y={tvec[1]:+.2f} Z={tvec[2]:+.2f} m  '
               f'R={rpy_deg[0]:+.0f} P={rpy_deg[1]:+.0f} Yaw={rpy_deg[2]:+.0f} deg')
        cv2.putText(vis, txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 128, 255), 2)

        self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = TargetPoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

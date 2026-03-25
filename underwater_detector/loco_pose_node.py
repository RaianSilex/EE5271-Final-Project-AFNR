#!/usr/bin/env python3
"""
LoCo ArUco Pose Estimator
=========================
Listens for YOLO 'LoCo' detections, searches the bounding box region for
ArUco markers, and publishes the full 6-DOF pose (X,Y,Z,Roll,Pitch,Yaw)
of LoCo's body frame w.r.t. the ZED left camera optical frame.

Marker layout (3 markers recommended for full coverage):
  ID 0 – Starboard (right) side:
          Left edge  → tail, right edge → nose (so marker +X points toward nose).
          Face normal (marker +Z) points away from the hull, to starboard.
  ID 1 – Nose (front):
          Marker face normal (+Z) points forward.
          Marker +X points to port (left when looking at the nose head-on).
  ID 2 – Top:
          Marker face normal (+Z) points upward.
          Marker +X points toward the nose (forward).

Body frame convention (ROS standard):
  +X = forward (toward nose)
  +Y = left    (port)
  +Z = up

Why 3 markers?
  A single ArUco marker gives full 6-DOF when visible.
  Side covers ~180° broadside arc.
  Nose covers ~90° forward cone.
  Top covers steep downward angles (pool overhead camera) and partial aft gaps.
  Together they give robust coverage from any practical ZED camera position.

Tuning checklist before running:
  1. Set 'marker_size' to the printed square side length in metres.
  2. Verify 'aruco_dict_id' matches the dictionary you printed
     (default 0 = DICT_4X4_50).
  3. Measure and set marker_X_offset params (position of each marker centre
     in the body frame so multi-marker pose fusion is accurate).
  4. Verify the R_body_from_marker matrices match your physical placement.
"""

import json

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as ScipyR
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


# ---------------------------------------------------------------------------
# Helper: build rigid-body transforms (R_body_from_marker, t_marker_in_body)
# ---------------------------------------------------------------------------
def _rot(rx_deg, ry_deg, rz_deg):
    """Intrinsic XYZ rotation convenience wrapper."""
    return ScipyR.from_euler('xyz', [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()


# Rotation matrices (columns = marker-axis directions expressed in body frame)
#
# Marker 0 – starboard side
#   marker +X → body +X (nose direction, because left=tail right=nose)
#   marker +Y → body +Z (up)
#   marker +Z → body −Y (starboard, out of hull)
R_BODY_FROM_M0 = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0],
], dtype=float)

# Marker 1 – nose / front
#   marker +X → body +Y (port; when facing the nose, left is port)
#   marker +Y → body +Z (up)
#   marker +Z → body +X (forward, out of nose)
R_BODY_FROM_M1 = np.array([
    [0,  0,  1],
    [1,  0,  0],
    [0,  1,  0],
], dtype=float)

# Marker 2 – top
#   marker +X → body +X (forward; top edge of marker points toward nose)
#   marker +Y → body +Y (port)
#   marker +Z → body +Z (up, out of top hull)
R_BODY_FROM_M2 = np.eye(3, dtype=float)


class LocoPoseEstimator(Node):
    """ROS2 node that estimates LoCo's 6-DOF pose via ArUco markers."""

    def __init__(self):
        super().__init__('loco_pose_estimator')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('marker_size',    0.10)   # side length in metres
        self.declare_parameter('aruco_dict_id',  0)      # cv2.aruco dict enum value
        self.declare_parameter('conf_threshold', 0.30)   # min YOLO confidence for LoCo

        # Marker centre positions in body frame [x_fwd, y_left, z_up] (metres)
        # Measure these on the physical robot and update accordingly.
        self.declare_parameter('marker0_offset', [0.0, -0.15, 0.0])  # starboard side
        self.declare_parameter('marker1_offset', [0.20,  0.0,  0.0]) # nose
        self.declare_parameter('marker2_offset', [0.0,   0.0,  0.10])# top

        marker_size   = self.get_parameter('marker_size').value
        dict_id       = self.get_parameter('aruco_dict_id').value
        self.conf_thr = self.get_parameter('conf_threshold').value
        self.marker_size = marker_size

        offsets = {
            0: np.array(self.get_parameter('marker0_offset').value, dtype=float),
            1: np.array(self.get_parameter('marker1_offset').value, dtype=float),
            2: np.array(self.get_parameter('marker2_offset').value, dtype=float),
        }

        # (R_body_from_marker,  translation_of_marker_in_body)
        self.marker_tf = {
            0: (R_BODY_FROM_M0, offsets[0]),
            1: (R_BODY_FROM_M1, offsets[1]),
            2: (R_BODY_FROM_M2, offsets[2]),
        }

        # ── ArUco detector ────────────────────────────────────────────────
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        # ── State ─────────────────────────────────────────────────────────
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs:   np.ndarray | None = None
        self.loco_bbox: list | None = None   # [x1,y1,x2,y2] pixels, updated by detector
        self.bridge = CvBridge()

        # ── Subscriptions ─────────────────────────────────────────────────
        self.create_subscription(
            CameraInfo, '/zed/zed_node/left/camera_info',
            self._camera_info_cb, 1)
        self.create_subscription(
            String, '/detections/json',
            self._detections_cb, 10)
        self.create_subscription(
            Image, '/zed/zed_node/left/image_rect_color',
            self._image_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_pose = self.create_publisher(PoseStamped, '/loco/pose', 10)
        self.pub_vis  = self.create_publisher(Image, '/loco/pose_image', 10)

        self.get_logger().info(
            f'LoCo pose estimator ready | marker_size={marker_size} m | '
            f'aruco_dict_id={dict_id}')

    # ── Callbacks ────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=float).reshape(3, 3)
            self.dist_coeffs   = np.array(msg.d, dtype=float)
            self.get_logger().info('Camera intrinsics received.')

    def _detections_cb(self, msg: String):
        """Cache the most-confident LoCo bounding box from the detector node."""
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        best_conf = self.conf_thr
        self.loco_bbox = None
        for det in detections:
            if det.get('label') == 'LoCo' and det['confidence'] > best_conf:
                self.loco_bbox = det['bbox']
                best_conf = det['confidence']

    def _image_cb(self, msg: Image):
        """Main pipeline: detect ArUco in LoCo ROI → estimate + publish pose."""
        if self.camera_matrix is None or self.loco_bbox is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.loco_bbox
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)

        roi = frame[y1c:y2c, x1c:x2c]
        if roi.size == 0:
            return

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        corners_roi, ids, _ = self.detector.detectMarkers(gray_roi)

        vis = frame.copy()
        cv2.rectangle(vis, (x1c, y1c), (x2c, y2c), (255, 80, 0), 2)
        cv2.putText(vis, 'LoCo ROI', (x1c, y1c - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 0), 2)

        if ids is None or len(ids) == 0:
            self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))
            return

        # Shift ROI-local corners back to full-frame pixel coordinates
        full_corners = []
        for c in corners_roi:
            fc = c.copy()
            fc[0, :, 0] += x1c
            fc[0, :, 1] += y1c
            full_corners.append(fc)

        body_positions = []
        body_rotmats   = []

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id not in self.marker_tf:
                self.get_logger().warn(
                    f'Unknown ArUco ID {marker_id} – add it to marker_tf if intentional.')
                continue

            R_B_M, t_M_in_B = self.marker_tf[marker_id]

            # Pose of marker in camera frame
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [full_corners[i]], self.marker_size,
                self.camera_matrix, self.dist_coeffs)
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]

            R_C_M, _ = cv2.Rodrigues(rvec)

            # Chain: T_camera←body = T_camera←marker * T_marker←body
            #   R_C_B  = R_C_M  @ R_B_M.T
            #   t_B_C  = t_M_C  − R_C_M @ R_B_M.T @ t_M_B
            R_C_B  = R_C_M @ R_B_M.T
            t_B_in_C = tvec - R_C_M @ R_B_M.T @ t_M_in_B

            body_positions.append(t_B_in_C)
            body_rotmats.append(R_C_B)

            # Draw the marker's own axis on the visualisation frame
            cv2.drawFrameAxes(
                vis, self.camera_matrix, self.dist_coeffs,
                rvec, tvec, self.marker_size * 0.6)
            cx = int(np.mean(full_corners[i][0, :, 0]))
            cy = int(np.mean(full_corners[i][0, :, 1]))
            cv2.putText(vis, f'M{marker_id}', (cx, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if body_positions:
            # ── Fuse estimates from all visible markers ──────────────────
            t_fused = np.mean(body_positions, axis=0)

            rots    = ScipyR.from_matrix(body_rotmats)
            r_fused = rots.mean()          # geodesic mean on SO(3)
            q       = r_fused.as_quat()   # [x, y, z, w]
            rpy_deg = r_fused.as_euler('xyz', degrees=True)

            # ── Publish PoseStamped ──────────────────────────────────────
            pose = PoseStamped()
            pose.header           = msg.header
            pose.header.frame_id  = 'zed_left_camera_optical_frame'
            pose.pose.position.x  = float(t_fused[0])
            pose.pose.position.y  = float(t_fused[1])
            pose.pose.position.z  = float(t_fused[2])
            pose.pose.orientation.x = float(q[0])
            pose.pose.orientation.y = float(q[1])
            pose.pose.orientation.z = float(q[2])
            pose.pose.orientation.w = float(q[3])
            self.pub_pose.publish(pose)

            self.get_logger().info(
                f'LoCo | XYZ [{t_fused[0]:+.3f}, {t_fused[1]:+.3f}, {t_fused[2]:+.3f}] m | '
                f'RPY [{rpy_deg[0]:+.1f}, {rpy_deg[1]:+.1f}, {rpy_deg[2]:+.1f}]° | '
                f'markers seen: {list(ids.flatten())}')

            # Annotate fused pose on visualisation
            txt = (f'X={t_fused[0]:+.2f} Y={t_fused[1]:+.2f} Z={t_fused[2]:+.2f} m  '
                   f'R={rpy_deg[0]:+.0f} P={rpy_deg[1]:+.0f} Yaw={rpy_deg[2]:+.0f} deg')
            cv2.putText(vis, txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

        self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = LocoPoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

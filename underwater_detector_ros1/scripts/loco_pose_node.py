#!/usr/bin/env python3
"""
LoCo ArUco Pose Estimator (ROS1 Melodic)
=========================================
Listens for YOLO 'LoCo' detections, searches the bounding box region for
ArUco markers, and publishes the full 6-DOF pose (X,Y,Z,Roll,Pitch,Yaw)
of LoCo's body frame w.r.t. the ZED left camera optical frame.

Marker layout (3 markers recommended for full coverage):
  ID 0 - Starboard (right) side
  ID 1 - Nose (front)
  ID 2 - Top

Body frame convention (ROS standard):
  +X = forward (toward nose)
  +Y = left    (port)
  +Z = up

Topics
------
  Subscribes:
    /zed/zed_node/left/camera_info      (sensor_msgs/CameraInfo)
    /zed/zed_node/left/image_rect_color (sensor_msgs/Image)
    /detections/json                     (std_msgs/String)

  Publishes:
    /loco/pose        (geometry_msgs/PoseStamped)
    /loco/pose_image  (sensor_msgs/Image)
"""

import json

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as ScipyR


# ---------------------------------------------------------------------------
# Helper: build rigid-body transforms (R_body_from_marker, t_marker_in_body)
# ---------------------------------------------------------------------------
def _rot(rx_deg, ry_deg, rz_deg):
    """Intrinsic XYZ rotation convenience wrapper."""
    return ScipyR.from_euler('xyz', [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()


# Rotation matrices (columns = marker-axis directions expressed in body frame)
#
# Marker 0 - starboard side
R_BODY_FROM_M0 = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0],
], dtype=float)

# Marker 1 - nose / front
R_BODY_FROM_M1 = np.array([
    [0,  0,  1],
    [1,  0,  0],
    [0,  1,  0],
], dtype=float)

# Marker 2 - top
R_BODY_FROM_M2 = np.eye(3, dtype=float)


class LocoPoseEstimator(object):
    """ROS1 node that estimates LoCo's 6-DOF pose via ArUco markers."""

    def __init__(self):
        rospy.init_node('loco_pose_estimator')

        # -- Parameters --------------------------------------------------------
        marker_size       = rospy.get_param('~marker_size',    0.10)
        dict_id           = rospy.get_param('~aruco_dict_id',  0)
        self.conf_thr     = rospy.get_param('~conf_threshold', 0.30)
        self.marker_size  = marker_size

        # Marker centre positions in body frame [x_fwd, y_left, z_up] (metres)
        m0_off = rospy.get_param('~marker0_offset', [0.0, -0.15, 0.0])
        m1_off = rospy.get_param('~marker1_offset', [0.20,  0.0,  0.0])
        m2_off = rospy.get_param('~marker2_offset', [0.0,   0.0,  0.10])

        offsets = {
            0: np.array(m0_off, dtype=float),
            1: np.array(m1_off, dtype=float),
            2: np.array(m2_off, dtype=float),
        }

        # (R_body_from_marker,  translation_of_marker_in_body)
        self.marker_tf = {
            0: (R_BODY_FROM_M0, offsets[0]),
            1: (R_BODY_FROM_M1, offsets[1]),
            2: (R_BODY_FROM_M2, offsets[2]),
        }

        # -- ArUco detector (OpenCV 3.x / Melodic style) ----------------------
        self.aruco_dict = cv2.aruco.Dictionary_get(dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # -- State -------------------------------------------------------------
        self.camera_matrix = None
        self.dist_coeffs   = None
        self.loco_bbox     = None
        self.bridge = CvBridge()

        # -- Subscriptions -----------------------------------------------------
        rospy.Subscriber('/zedm/zed_node/left_raw/camera_info',
                         CameraInfo, self._camera_info_cb, queue_size=1)
        rospy.Subscriber('/detections/json',
                         String, self._detections_cb, queue_size=10)
        rospy.Subscriber('/zedm/zed_node/left_raw/image_raw_color',
                         Image, self._image_cb, queue_size=1,
                         buff_size=2**24)

        # -- Publishers --------------------------------------------------------
        self.pub_pose = rospy.Publisher('/loco/pose', PoseStamped, queue_size=10)
        self.pub_vis  = rospy.Publisher('/loco/pose_image', Image, queue_size=10)

        rospy.loginfo(
            'LoCo pose estimator ready | marker_size=%.2f m | aruco_dict_id=%d',
            marker_size, dict_id)

    # -- Callbacks -------------------------------------------------------------

    def _camera_info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K, dtype=float).reshape(3, 3)
            self.dist_coeffs   = np.array(msg.D, dtype=float)
            rospy.loginfo('Camera intrinsics received.')

    def _detections_cb(self, msg):
        """Cache the most-confident LoCo bounding box from the detector node."""
        try:
            detections = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        best_conf = self.conf_thr
        self.loco_bbox = None
        for det in detections:
            if det.get('label') == 'LoCo' and det['confidence'] > best_conf:
                self.loco_bbox = det['bbox']
                best_conf = det['confidence']

    def _image_cb(self, msg):
        """Main pipeline: detect ArUco in LoCo ROI -> estimate + publish pose."""
        if self.camera_matrix is None or self.loco_bbox is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr('cv_bridge: %s', e)
            return

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.loco_bbox
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)

        roi = frame[y1c:y2c, x1c:x2c]
        if roi.size == 0:
            return

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        corners_roi, ids, _ = cv2.aruco.detectMarkers(
            gray_roi, self.aruco_dict, parameters=self.aruco_params)

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
                rospy.logwarn(
                    'Unknown ArUco ID %d - add it to marker_tf if intentional.',
                    marker_id)
                continue

            R_B_M, t_M_in_B = self.marker_tf[marker_id]

            # Pose of marker in camera frame
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [full_corners[i]], self.marker_size,
                self.camera_matrix, self.dist_coeffs)
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]

            R_C_M, _ = cv2.Rodrigues(rvec)

            # Chain: T_camera<-body = T_camera<-marker * T_marker<-body
            R_C_B  = R_C_M.dot(R_B_M.T)
            t_B_in_C = tvec - R_C_M.dot(R_B_M.T).dot(t_M_in_B)

            body_positions.append(t_B_in_C)
            body_rotmats.append(R_C_B)

            # Draw the marker's own axis on the visualisation frame
            cv2.aruco.drawAxis(
                vis, self.camera_matrix, self.dist_coeffs,
                rvec, tvec, self.marker_size * 0.6)
            cx = int(np.mean(full_corners[i][0, :, 0]))
            cy = int(np.mean(full_corners[i][0, :, 1]))
            cv2.putText(vis, 'M%d' % marker_id, (cx, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if body_positions:
            # -- Fuse estimates from all visible markers -----------------------
            t_fused = np.mean(body_positions, axis=0)

            rots    = ScipyR.from_matrix(body_rotmats)
            r_fused = rots.mean()
            q       = r_fused.as_quat()   # [x, y, z, w]
            rpy_deg = r_fused.as_euler('xyz', degrees=True)

            # -- Publish PoseStamped -------------------------------------------
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

            rospy.loginfo(
                'LoCo | XYZ [%+.3f, %+.3f, %+.3f] m | '
                'RPY [%+.1f, %+.1f, %+.1f] deg | '
                'markers seen: %s',
                t_fused[0], t_fused[1], t_fused[2],
                rpy_deg[0], rpy_deg[1], rpy_deg[2],
                list(ids.flatten()))

            # Annotate fused pose on visualisation
            txt = ('X=%+.2f Y=%+.2f Z=%+.2f m  '
                   'R=%+.0f P=%+.0f Yaw=%+.0f deg' %
                   (t_fused[0], t_fused[1], t_fused[2],
                    rpy_deg[0], rpy_deg[1], rpy_deg[2]))
            cv2.putText(vis, txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

        self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))


def main():
    node = LocoPoseEstimator()
    rospy.spin()


if __name__ == '__main__':
    main()

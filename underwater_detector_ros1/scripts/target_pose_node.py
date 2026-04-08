#!/usr/bin/env python3
"""
Target ArUco Pose Estimator (ROS1 Melodic)
==========================================
Listens for YOLO detections, finds ALL non-'LoCo' detections above the
confidence threshold, searches each bounding-box region for ArUco marker
ID 3, and publishes a PoseArray containing the 6-DOF pose of every target
whose marker is visible.

Topics
------
  Subscribes:
    /zed/zed_node/left/camera_info      (sensor_msgs/CameraInfo)
    /zed/zed_node/left/image_rect_color (sensor_msgs/Image)
    /detections/json                     (std_msgs/String)

  Publishes:
    /targets/poses      (geometry_msgs/PoseArray)
    /targets/pose_image (sensor_msgs/Image)
"""

import json

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as ScipyR

_LOCO_LABEL = 'LoCo'


class TargetPoseEstimator(object):
    """ROS1 node that estimates all non-LoCo targets' 6-DOF poses via ArUco marker ID 3."""

    def __init__(self):
        rospy.init_node('target_pose_estimator')

        # -- Parameters --------------------------------------------------------
        self.marker_id   = rospy.get_param('~marker_id',   3)
        self.marker_size = rospy.get_param('~marker_size', 0.10)
        dict_id          = rospy.get_param('~aruco_dict_id', 0)
        self.conf_thr    = rospy.get_param('~conf_threshold', 0.30)
        rpy              = rospy.get_param('~marker_to_target_rpy_deg', [0.0, 0.0, 0.0])
        self.R_target_from_marker = ScipyR.from_euler(
            'xyz', rpy, degrees=True).as_matrix()

        # -- ArUco detector (OpenCV 3.x / Melodic style) -----------------------
        self.aruco_dict = cv2.aruco.Dictionary_get(dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # -- State -------------------------------------------------------------
        self.camera_matrix = None
        self.dist_coeffs   = None
        self.target_dets   = []
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
        self.pub_poses = rospy.Publisher('/targets/poses', PoseArray, queue_size=10)
        self.pub_vis   = rospy.Publisher('/targets/pose_image', Image, queue_size=10)

        rospy.loginfo(
            'Target pose estimator ready | '
            'marker_id=%d | marker_size=%.2f m | '
            'aruco_dict_id=%d | conf_threshold=%.2f',
            self.marker_id, self.marker_size, dict_id, self.conf_thr)

    # -- Callbacks -------------------------------------------------------------

    def _camera_info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K, dtype=float).reshape(3, 3)
            self.dist_coeffs   = np.array(msg.D, dtype=float)
            rospy.loginfo('Camera intrinsics received.')

    def _detections_cb(self, msg):
        """Cache all non-LoCo detections above confidence threshold."""
        try:
            detections = json.loads(msg.data)
        except (ValueError, TypeError):
            return

        self.target_dets = [
            det for det in detections
            if det.get('label') != _LOCO_LABEL
            and det.get('confidence', 0.0) >= self.conf_thr
        ]

    def _image_cb(self, msg):
        """For each cached target detection, find ArUco ID 3 in its ROI and publish pose."""
        if self.camera_matrix is None or not self.target_dets:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr('cv_bridge: %s', e)
            return

        h, w = frame.shape[:2]
        vis = frame.copy()

        pose_array = PoseArray()
        pose_array.header          = msg.header
        pose_array.header.frame_id = 'zed_left_camera_optical_frame'

        vis_row_offset = 0

        for det in self.target_dets:
            x1, y1, x2, y2 = det['bbox']
            x1c, y1c = max(0, int(x1)), max(0, int(y1))
            x2c, y2c = min(w, int(x2)), min(h, int(y2))
            label = det.get('label', 'Target')

            cv2.rectangle(vis, (x1c, y1c), (x2c, y2c), (0, 128, 255), 2)
            cv2.putText(vis, label, (x1c, y1c - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 128, 255), 2)

            roi = frame[y1c:y2c, x1c:x2c]
            if roi.size == 0:
                continue

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            corners_roi, ids, _ = cv2.aruco.detectMarkers(
                gray_roi, self.aruco_dict, parameters=self.aruco_params)

            if ids is None or len(ids) == 0:
                continue

            ids_flat = ids.flatten()
            matches = np.where(ids_flat == self.marker_id)[0]
            if len(matches) == 0:
                continue

            idx = matches[0]

            # Shift ROI-local corners back to full-frame coordinates
            corners_full = corners_roi[idx].copy()
            corners_full[0, :, 0] += x1c
            corners_full[0, :, 1] += y1c

            # -- Pose of marker in camera frame --------------------------------
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners_full], self.marker_size,
                self.camera_matrix, self.dist_coeffs)
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]

            R_cam_marker, _ = cv2.Rodrigues(rvec)

            # -- Apply optional marker->target rotation ------------------------
            R_cam_target = R_cam_marker.dot(self.R_target_from_marker.T)
            q       = ScipyR.from_matrix(R_cam_target).as_quat()  # [x, y, z, w]
            rpy_deg = ScipyR.from_matrix(R_cam_target).as_euler('xyz', degrees=True)

            # -- Append to PoseArray -------------------------------------------
            pose = Pose()
            pose.position.x = float(tvec[0])
            pose.position.y = float(tvec[1])
            pose.position.z = float(tvec[2])
            pose.orientation.x = float(q[0])
            pose.orientation.y = float(q[1])
            pose.orientation.z = float(q[2])
            pose.orientation.w = float(q[3])
            pose_array.poses.append(pose)

            rospy.loginfo(
                'Target (%s) | '
                'XYZ [%+.3f, %+.3f, %+.3f] m | '
                'RPY [%+.1f, %+.1f, %+.1f] deg',
                label, tvec[0], tvec[1], tvec[2],
                rpy_deg[0], rpy_deg[1], rpy_deg[2])

            # -- Visualisation -------------------------------------------------
            cv2.aruco.drawAxis(
                vis, self.camera_matrix, self.dist_coeffs,
                rvec, tvec, self.marker_size * 0.6)
            cx = int(np.mean(corners_full[0, :, 0]))
            cy = int(np.mean(corners_full[0, :, 1]))
            cv2.putText(vis, 'ID%d' % self.marker_id, (cx, cy - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
            txt = ('[%s] '
                   'X=%+.2f Y=%+.2f Z=%+.2f m  '
                   'R=%+.0f P=%+.0f Yaw=%+.0f deg' %
                   (label, tvec[0], tvec[1], tvec[2],
                    rpy_deg[0], rpy_deg[1], rpy_deg[2]))
            cv2.putText(vis, txt, (10, 30 + vis_row_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 128, 255), 2)
            vis_row_offset += 24

        if pose_array.poses:
            self.pub_poses.publish(pose_array)

        self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))


def main():
    node = TargetPoseEstimator()
    rospy.spin()


if __name__ == '__main__':
    main()

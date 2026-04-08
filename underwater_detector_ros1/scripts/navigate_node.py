#!/usr/bin/env python3
"""
LoCo Target Navigation Node (ROS1 Melodic)
===========================================
Subscribes to LoCo's pose and the target's pose, computes the relative
error, and publishes loco_pilot/Command messages to drive LoCo toward
the closest target.

LoCo has three controllable degrees of freedom:
  - throttle : forward/reverse thrust  (-1.0 to 1.0)
  - yaw      : left/right turning      (-1.0 to 1.0)
  - pitch    : pitch up/down            (-1.0 to 1.0)

Strategy (all in the camera optical frame: +X right, +Y down, +Z forward):
  1. Yaw to align horizontally with the target  (error in X)
  2. Pitch to align vertically with the target   (error in Y)
  3. Thrust forward to close the distance         (error in Z)

We use proportional control on each axis, clamped to [-max, +max].

Subscriptions
-------------
  /loco/pose        (geometry_msgs/PoseStamped)  - LoCo's 6-DOF pose
  /targets/poses    (geometry_msgs/PoseArray)     - target(s) 6-DOF pose(s)

Publications
------------
  /loco/command     (loco_pilot/Command)          - motion command
    throttle  = forward/reverse  (-1 to 1)
    yaw       = left/right       (-1 to 1)
    pitch     = up/down           (-1 to 1)

Parameters
----------
  ~kp_yaw            float   default 0.8   Proportional gain for yaw
  ~kp_pitch          float   default 0.8   Proportional gain for pitch
  ~kp_throttle       float   default 0.5   Proportional gain for throttle
  ~max_yaw           float   default 0.4   Max yaw command magnitude
  ~max_pitch         float   default 0.4   Max pitch command magnitude
  ~max_throttle      float   default 0.4   Max throttle command magnitude
  ~goal_distance     float   default 0.30  Distance (m) at which we stop
  ~control_rate      float   default 10.0  Control loop frequency (Hz)
"""

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, PoseArray
from loco_pilot.msg import Command


def clamp(value, limit):
    return max(-limit, min(limit, value))


class LocoNavigator(object):
    def __init__(self):
        rospy.init_node('loco_navigator')

        # -- Parameters --------------------------------------------------------
        self.kp_yaw       = rospy.get_param('~kp_yaw',       0.8)
        self.kp_pitch     = rospy.get_param('~kp_pitch',     0.8)
        self.kp_throttle  = rospy.get_param('~kp_throttle',  0.5)
        self.max_yaw      = rospy.get_param('~max_yaw',      0.4)
        self.max_pitch    = rospy.get_param('~max_pitch',     0.4)
        self.max_throttle = rospy.get_param('~max_throttle',  0.4)
        self.goal_dist    = rospy.get_param('~goal_distance', 0.30)
        rate_hz           = rospy.get_param('~control_rate',  10.0)

        # -- State -------------------------------------------------------------
        self.loco_pose    = None  # PoseStamped
        self.target_poses = None  # PoseArray

        # -- Subscriptions -----------------------------------------------------
        rospy.Subscriber('/loco/pose', PoseStamped, self._loco_pose_cb,
                         queue_size=1)
        rospy.Subscriber('/targets/poses', PoseArray, self._targets_cb,
                         queue_size=1)

        # -- Publisher (loco_pilot/Command on /loco/command) -------------------
        self.cmd_pub = rospy.Publisher('/loco/command', Command, queue_size=5)

        # -- Control loop ------------------------------------------------------
        self.rate = rospy.Rate(rate_hz)

        rospy.loginfo(
            'Navigator ready | '
            'kp=[%.2f, %.2f, %.2f] | '
            'max=[%.2f, %.2f, %.2f] | '
            'goal_dist=%.2f m | rate=%.1f Hz',
            self.kp_yaw, self.kp_pitch, self.kp_throttle,
            self.max_yaw, self.max_pitch, self.max_throttle,
            self.goal_dist, rate_hz)

    # -- Callbacks -------------------------------------------------------------

    def _loco_pose_cb(self, msg):
        self.loco_pose = msg

    def _targets_cb(self, msg):
        self.target_poses = msg

    # -- Main loop -------------------------------------------------------------

    def run(self):
        while not rospy.is_shutdown():
            cmd = Command()
            cmd.header.stamp = rospy.Time.now()

            if self.loco_pose is None or self.target_poses is None:
                self.cmd_pub.publish(cmd)
                self.rate.sleep()
                continue

            if len(self.target_poses.poses) == 0:
                self.cmd_pub.publish(cmd)
                self.rate.sleep()
                continue

            # Both poses are in camera optical frame
            # Camera optical: +X right, +Y down, +Z forward
            lp = self.loco_pose.pose.position
            loco_pos = np.array([lp.x, lp.y, lp.z])

            # Find the closest target
            best_dist = float('inf')
            best_target = None
            for pose in self.target_poses.poses:
                tp = pose.position
                target_pos = np.array([tp.x, tp.y, tp.z])
                dist = np.linalg.norm(target_pos - loco_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_target = target_pos

            if best_target is None:
                self.cmd_pub.publish(cmd)
                self.rate.sleep()
                continue

            # Error vector from LoCo to target in camera optical frame
            error = best_target - loco_pos
            err_x = error[0]  # horizontal (positive = target is to the right)
            err_y = error[1]  # vertical   (positive = target is below)
            err_z = error[2]  # forward    (positive = target is ahead)

            distance = np.linalg.norm(error)

            if distance > self.goal_dist:
                # Yaw: target to the right (+X) -> yaw right (+yaw)
                yaw_cmd = clamp(self.kp_yaw * err_x, self.max_yaw)

                # Pitch: target below (+Y optical) -> pitch down
                # LoCo pitch: positive = pitch up, so negate
                pitch_cmd = clamp(self.kp_pitch * (-err_y), self.max_pitch)

                # Throttle: target ahead (+Z) -> thrust forward
                throttle_cmd = clamp(self.kp_throttle * err_z, self.max_throttle)

                cmd.yaw      = yaw_cmd
                cmd.pitch    = pitch_cmd
                cmd.throttle = throttle_cmd

                rospy.loginfo(
                    'Nav | dist=%.2f m | '
                    'err=[%+.2f, %+.2f, %+.2f] | '
                    'cmd: thr=%+.2f pitch=%+.2f yaw=%+.2f',
                    distance, err_x, err_y, err_z,
                    throttle_cmd, pitch_cmd, yaw_cmd)
            else:
                rospy.loginfo('Target reached! distance=%.2f m', distance)

            self.cmd_pub.publish(cmd)
            self.rate.sleep()


def main():
    node = LocoNavigator()
    node.run()


if __name__ == '__main__':
    main()

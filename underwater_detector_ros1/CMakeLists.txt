cmake_minimum_required(VERSION 2.8.3)
project(underwater_detector)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  loco_pilot
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/detector_node.py
  scripts/loco_pose_node.py
  scripts/target_pose_node.py
  scripts/navigate_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
)

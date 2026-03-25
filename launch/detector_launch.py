from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    return LaunchDescription([

        # ── YOLO detector ─────────────────────────────────────────────────
        Node(
            package='underwater_detector',
            executable='detector_node',
            name='underwater_detector',
            output='screen',
            parameters=[{
                'weights':    os.path.expanduser('~/EE5271-Final-Project-AFNR/yolov9/best.pt'),
                'imgsz':      640,
                'conf_thres': 0.25,
                'iou_thres':  0.45,
                'device':     'cpu',   # change to 'cuda:0' if GPU available
            }],
        ),

        # ── LoCo ArUco pose estimator ─────────────────────────────────────
        # Subscribes to:
        #   /detections/json                        (from detector_node)
        #   /zed/zed_node/left/image_rect_color     (raw ZED image)
        #   /zed/zed_node/left/camera_info          (ZED intrinsics)
        # Publishes:
        #   /loco/pose        geometry_msgs/PoseStamped  (6-DOF body pose in camera frame)
        #   /loco/pose_image  sensor_msgs/Image          (visualisation with axes drawn)
        Node(
            package='underwater_detector',
            executable='loco_pose_node',
            name='loco_pose_estimator',
            output='screen',
            parameters=[{
                # ── ArUco settings ────────────────────────────────────────
                # Must match the dictionary you printed the markers from.
                #   0  = DICT_4X4_50   (recommended – fast, low false-positive rate)
                #   2  = DICT_5X5_100
                #   10 = DICT_6X6_250
                'aruco_dict_id': 0,

                # Physical side length of each printed marker square (metres).
                # Measure from black border edge to black border edge.
                'marker_size': 0.10,

                # Min YOLO confidence required before running ArUco search.
                'conf_threshold': 0.30,

                # ── Marker positions in body frame ────────────────────────
                # Body frame: X=forward(nose), Y=left(port), Z=up
                # Measure from robot centre to each marker centre in metres.
                # Update these after physical measurement!
                #
                # ID 0 – starboard side  [x_fwd, y_left, z_up]
                'marker0_offset': [0.0, -0.15, 0.0],
                # ID 1 – nose / front
                'marker1_offset': [0.20,  0.0,  0.0],
                # ID 2 – top
                'marker2_offset': [0.0,   0.0,  0.10],
            }],
        ),
    ])

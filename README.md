# underwater_detector

A ROS2 package for real-time underwater object detection using a YOLOv9 model exported to ONNX, with ArUco-based 6-DOF pose estimation for the LoCo AUV.

---

## Requirements

- **Ubuntu 22.04**
- **ROS2 Humble** — [install guide](https://docs.ros.org/en/humble/Installation.html)
- **Git LFS** — required to download the model weights on clone (`sudo apt-get install git-lfs && git lfs install`)

---

## Dependencies

Install Python dependencies:

```bash
pip install onnxruntime opencv-python numpy
```

> For GPU inference, replace `onnxruntime` with `onnxruntime-gpu`.

Install ROS2 package dependencies:

```bash
sudo apt install ros-humble-cv-bridge ros-humble-vision-opencv
```

---

## Installation

The model weights (`best.onnx`) are stored in this repo using **Git LFS** (Large File Storage). You need to install it once before cloning, otherwise you'll get a small placeholder file instead of the real model.

```bash
# Install Git LFS (one-time setup)
sudo apt-get install git-lfs
git lfs install
```

```bash
# Clone the repo — the model will download automatically
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/RaianSilex/underwater_detector.git

# Build
cd ~/ros2_ws
colcon build --packages-select underwater_detector
source install/setup.bash
```

---

## Usage

### Launch everything (detector + pose estimator)

```bash
ros2 launch underwater_detector detector_launch.py
```

### Run the detector node only

```bash
ros2 run underwater_detector detector_node
```

---

## Topics

| Direction | Topic | Type | Description |
|-----------|-------|------|-------------|
| Subscribes | `/zed/zed_node/left/image_rect_color` | `sensor_msgs/Image` | Input feed from ZED camera |
| Publishes | `/detections/image` | `sensor_msgs/Image` | Annotated image with bounding boxes |
| Publishes | `/detections/json` | `std_msgs/String` | JSON array of detections |
| Publishes | `/loco/pose` | `geometry_msgs/PoseStamped` | 6-DOF LoCo pose in camera frame |
| Publishes | `/loco/pose_image` | `sensor_msgs/Image` | Pose visualisation with axes drawn |

---

## Parameters

### detector_node

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weights` | `models/best.onnx` | Path to the ONNX model |
| `imgsz` | `640` | Input image size (square) |
| `conf_thres` | `0.25` | Confidence threshold |
| `iou_thres` | `0.45` | NMS IoU threshold |
| `device` | `cpu` | `cpu` or `cuda` |

### loco_pose_node

| Parameter | Default | Description |
|-----------|---------|-------------|
| `aruco_dict_id` | `0` | ArUco dictionary (`0` = DICT_4X4_50) |
| `marker_size` | `0.10` | Printed marker side length in metres |
| `conf_threshold` | `0.30` | Min YOLO confidence before ArUco search |
| `marker0_offset` | `[0.0, -0.15, 0.0]` | Marker ID 0 position in body frame (m) |
| `marker1_offset` | `[0.20, 0.0, 0.0]` | Marker ID 1 position in body frame (m) |
| `marker2_offset` | `[0.0, 0.0, 0.10]` | Marker ID 2 position in body frame (m) |

---

## GPU Inference

In `detector_launch.py`, change:

```python
'device': 'cpu'
```
to:
```python
'device': 'cuda'
```

And make sure `onnxruntime-gpu` is installed instead of `onnxruntime`.

---

## Detectable Classes

`Unknown Instance`, `Scissors`, `Plastic Cup`, `Metal Rod`, `Fork`, `Bottle`, `Soda Can`, `Case`, `Plastic Bag`, `Cup`, `Goggles`, `Flipper`, `LoCo`, `Aqua`, `Pipe`, `Snorkel`, `Spoon`, `Lure`, `Screwdriver`, `Car`, `Tripod`, `ROV`, `Knife`, `Dive Weight`

---

## ArUco Markers

Printable marker PDFs (DICT_4X4_50) are in [`aruco_markers/`](aruco_markers/).

- `ID0.pdf` — starboard side
- `ID1.pdf` — nose / front
- `ID2.pdf` — top

Measure the printed marker size (black border edge to edge) and update `marker_size` in the launch file accordingly.

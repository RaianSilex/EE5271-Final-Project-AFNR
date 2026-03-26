# ~/ros2_ws/src/underwater_detector/setup.py
from setuptools import setup
from glob import glob
import os

package_name = 'underwater_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_data={package_name: ['models/*.onnx']},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'onnxruntime'],
    zip_safe=True,
    maintainer='RaianSilex',
    maintainer_email='chowd207@umn.edu',
    description='YOLOv9 underwater object detector',
    license='MIT',
    entry_points={
        'console_scripts': [
            'detector_node     = underwater_detector.detector_node:main',
            'loco_pose_node    = underwater_detector.loco_pose_node:main',
            'target_pose_node  = underwater_detector.target_pose_node:main',
        ],
    },
)
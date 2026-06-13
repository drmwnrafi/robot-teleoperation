import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'robot_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zz',
    maintainer_email='zz@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    package_data={
        package_name: ['models/*.task'],
    },
    entry_points={
        'console_scripts': [
            'landmarks_node = robot_teleop.landmarks_node:main',
            'hand_landmarks_node = robot_teleop.hand_landmarks_node:main',
            'landmark_marker = robot_teleop.landmark_marker:main',
            'landmark_processor = robot_teleop.landmark_processor:main',
            'servo_keyboard = robot_teleop.servo_keyboard:main',
            'hand_pose_tracker = robot_teleop.hand_pose_tracker:main',
        ],
    },
)
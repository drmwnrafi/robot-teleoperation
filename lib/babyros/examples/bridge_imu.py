"""
Bridge babyros 'imu' topic to ROS 2 '/imu' topic
"""
from babyrosbridge import run_bridge

if __name__ == "__main__":
    print("Starting babyros -> ROS 2 bridge for IMU...")
    run_bridge(
        babyros_to_ros=[
            # (babyros_topic, ros_topic, ros_msg_type)
            ("imu", "/imu", "sensor_msgs/msg/Imu"),
        ]
    )
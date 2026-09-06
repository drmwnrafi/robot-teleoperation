"""
Zenoh Publisher Example - IMU data compatible with ROS 2 sensor_msgs/msg/Imu
"""
import time
import babyros
import numpy as np

if __name__ == "__main__":
    imu_pub = babyros.node.Publisher(topic="imu")

    topics = babyros.get_topics_in_session()
    print("Active topics in current session:", topics)

    print("Starting sensor stream... (Press Ctrl+C to stop)")
    count = 0
    try:
        while True:
            data = {
                "linear_acceleration": {
                    "x":np.random.rand(), "y":np.random.rand(), "z":np.random.rand()
                },
            }
            
            imu_pub.publish(data=data)
            print(f"Sent seq: {count}")
            
            count += 1
            time.sleep(0.1)  # 10 Hz
            
    except KeyboardInterrupt:
        print("\n[Publisher] Interrupted by user.")
    finally:
        imu_pub.delete()
        print("[Publisher] Cleanup complete.")
"""
babyrosbridge.py
Standalone module for bridging ROS 2 topics to babyros (Zenoh) and vice versa.
"""
import importlib
import numpy as np
import rclpy
from rclpy.node import Node
from typing import List, Tuple, Optional
import babyros
from babyros.node import Publisher, Subscriber, SessionManager
from loguru import logger
import yaml

def load_config(config_path: str) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """
    Load bridge configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Tuple of (ros_to_babyros, babyros_to_ros) lists
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ros_to_babyros = []
    babyros_to_ros = []
    
    # Parse ros_to_babyros mappings
    if 'ros_to_babyros' in config:
        for mapping in config['ros_to_babyros']:
            ros_to_babyros.append((
                mapping['ros_topic'],
                mapping['ros_msg_type'],
                mapping['babyros_topic']
            ))
    
    # Parse babyros_to_ros mappings
    if 'babyros_to_ros' in config:
        for mapping in config['babyros_to_ros']:
            babyros_to_ros.append((
                mapping['babyros_topic'],
                mapping['ros_topic'],
                mapping['ros_msg_type']
            ))
    
    # Handle shorthand format if provided (explicit direction only)
    if 'topics' in config:
        direction = config.get('direction', 'ros2babyros')
        
        if direction not in ('ros2babyros', 'babyros2ros'):
            raise ValueError(f"Invalid direction '{direction}'. Must be 'ros2babyros' or 'babyros2ros'")
        
        for topic_str in config['topics']:
            parts = topic_str.split(':')
            if len(parts) != 3:
                raise ValueError(f"Invalid topic format: {topic_str}. Expected 'topic1:type:topic2'")
            
            if direction == 'ros2babyros':
                ros_to_babyros.append((parts[0], parts[1], parts[2]))
            elif direction == 'babyros2ros':
                babyros_to_ros.append((parts[2], parts[0], parts[1]))
    
    return ros_to_babyros, babyros_to_ros

def get_ros2_msg_type(type_str: str):
    """
    Dynamically import a ROS 2 message type from a string.
    Handles both 'sensor_msgs/msg/Imu' and legacy/shorthand 'std_msgs/Header' formats.
    """
    try:
        # Handle standard 'package/msg/MessageName' format
        if '/msg/' in type_str:
            package, msg_name = type_str.split('/msg/')
            module_name = f"{package}.msg"
        else:
            # Handle shorthand 'package/MessageName' format (e.g., 'std_msgs/Header')
            package, msg_name = type_str.rsplit('/', 1)
            module_name = f"{package}.msg"
            
        module = importlib.import_module(module_name)
        return getattr(module, msg_name)
    except Exception as e:
        raise ValueError(f"Could not load ROS 2 message type '{type_str}': {e}")


def ros2_msg_to_dict(msg) -> dict:
    """Recursively convert a ROS 2 message to a dictionary."""
    if hasattr(msg, 'get_fields_and_field_types'):
        res = {}
        for field in msg.get_fields_and_field_types().keys():
            val = getattr(msg, field)
            res[field] = ros2_msg_to_dict(val)
        return res
    elif isinstance(msg, (list, tuple)):
        return [ros2_msg_to_dict(v) for v in msg]
    elif hasattr(msg, 'tolist'):
        return msg.tolist()
    elif isinstance(msg, bytes):
        return list(msg)
    else:
        return msg


def dict_to_ros2_msg(msg_type, data: dict):
    """Recursively convert a dictionary to a ROS 2 message."""
    if not isinstance(data, dict):
        return data
        
    msg = msg_type()
    fields = msg.get_fields_and_field_types()
    
    for field, f_type_str in fields.items():
        if field not in data:
            continue
            
        val = data[field]
        
        if f_type_str.startswith('sequence<'):
            inner_type_str = f_type_str[9:-1]
            if '/' in inner_type_str:
                inner_msg_type = get_ros2_msg_type(inner_type_str)
                setattr(msg, field, [dict_to_ros2_msg(inner_msg_type, v) for v in val])
            else:
                setattr(msg, field, val)
        elif '/' in f_type_str:
            nested_msg_type = get_ros2_msg_type(f_type_str)
            nested_msg = dict_to_ros2_msg(nested_msg_type, val)
            setattr(msg, field, nested_msg)
        else:
            setattr(msg, field, val)
            
    return msg


def ros2_image_to_np(msg) -> np.ndarray:
    """Convert sensor_msgs/msg/Image to numpy array."""
    if msg.encoding in ['rgb8', 'bgr8']:
        dtype = np.uint8
        channels = 3
    elif msg.encoding in ['mono8', '8UC1']:
        dtype = np.uint8
        channels = 1
    elif msg.encoding in ['mono16', '16UC1']:
        dtype = np.uint16
        channels = 1
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    
    arr = np.frombuffer(msg.data, dtype=dtype)
    if channels > 1:
        return arr.reshape((msg.height, msg.width, channels))
    return arr.reshape((msg.height, msg.width))


def np_to_ros2_image(arr: np.ndarray, msg_type) -> object:
    """Convert numpy array to sensor_msgs/msg/Image."""
    msg = msg_type()
    if arr.ndim == 2:
        msg.height, msg.width = arr.shape
        channels = 1
        msg.encoding = 'mono8' if arr.dtype == np.uint8 else 'mono16'
    else:
        msg.height, msg.width, channels = arr.shape
        msg.encoding = 'rgb8' if channels == 3 else 'bgr8'
        
    msg.step = msg.width * channels * arr.dtype.itemsize
    msg.data = arr.tobytes()
    return msg


class BabyRosBridge(Node):
    def __init__(
        self,
        ros_to_babyros: Optional[List[Tuple[str, str, str]]] = None,
        babyros_to_ros: Optional[List[Tuple[str, str, str]]] = None
    ):
        super().__init__('babyros_bridge')
        
        self.ros_subs = []
        self.babyros_pubs = {}
        self.babyros_subs = []
        self.ros_pubs = {}

        SessionManager.get_session()

        ros_to_babyros = ros_to_babyros or []
        babyros_to_ros = babyros_to_ros or []

        # --- ROS 2 -> babyros ---
        for ros_topic, ros_type_str, babyros_topic in ros_to_babyros:
            # Strip leading slash from babyros topic
            babyros_topic = babyros_topic.lstrip('/')
            
            msg_type = get_ros2_msg_type(ros_type_str)
            pub = Publisher(babyros_topic)
            self.babyros_pubs[babyros_topic] = pub

            def make_ros_callback(publisher, m_type, is_image=False):
                def cb(msg):
                    if is_image:
                        data = ros2_image_to_np(msg)
                    else:
                        data = ros2_msg_to_dict(msg)
                    publisher.publish(data)
                return cb

            is_image = (ros_type_str == 'sensor_msgs/msg/Image')
            sub = self.create_subscription(
                msg_type, 
                ros_topic, 
                make_ros_callback(pub, msg_type, is_image), 
                10
            )
            self.ros_subs.append(sub)
            self.get_logger().info(f"Bridging ROS 2 '{ros_topic}' -> babyros '{babyros_topic}'")

        # --- babyros -> ROS 2 ---
        for babyros_topic, ros_topic, ros_type_str in babyros_to_ros:
            # Strip leading slash from babyros topic
            babyros_topic = babyros_topic.lstrip('/')
            
            msg_type = get_ros2_msg_type(ros_type_str)
            pub = self.create_publisher(msg_type, ros_topic, 10)
            self.ros_pubs[ros_topic] = pub

            def make_babyros_callback(publisher, m_type, is_image=False):
                def cb(data):
                    if is_image:
                        msg = np_to_ros2_image(data, m_type)
                    else:
                        msg = dict_to_ros2_msg(m_type, data)
                    publisher.publish(msg)
                return cb

            is_image = (ros_type_str == 'sensor_msgs/msg/Image')
            sub = Subscriber(
                babyros_topic, 
                make_babyros_callback(pub, msg_type, is_image)
            )
            self.babyros_subs.append(sub)
            self.get_logger().info(f"Bridging babyros '{babyros_topic}' -> ROS 2 '{ros_topic}'")

    def destroy_bridge(self):
        for sub in self.babyros_subs:
            sub.delete()
        self.destroy_node()


def run_bridge(
    ros_to_babyros: Optional[List[Tuple[str, str, str]]] = None,
    babyros_to_ros: Optional[List[Tuple[str, str, str]]] = None
):
    """
    Convenience function to run the bridge with automatic initialization/cleanup.
    """
    rclpy.init()
    bridge = BabyRosBridge(ros_to_babyros, babyros_to_ros)
    
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_bridge()
        SessionManager.delete(force=True)
        rclpy.shutdown()
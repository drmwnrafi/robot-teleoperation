#!/usr/bin/env python3

import sys
import select
import tty
import termios
import threading
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import TwistStamped, PoseStamped
from control_msgs.msg import JointJog
from moveit_msgs.srv import ServoCommandType
from visualization_msgs.msg import Marker
import tf2_ros
from geometry_msgs.msg import Point, Quaternion

# Constants
MAX_VEL = 0.2
PUBLISH_RATE = 50
POSE_STEP = 0.02

# Left Arm Config
LEFT_TWIST_TOPIC = "/servo_node_left/delta_twist_cmds"
LEFT_POSE_TOPIC = "/servo_node_left/pose_cmd"
LEFT_JOINT_TOPIC = "/servo_node_left/delta_joint_cmds"
LEFT_SWITCH_SERVICE = "/servo_node_left/switch_command_type"
LEFT_EE_FRAME = "obotx_left_tool0"
LEFT_JOINTS = [
    "obotx_left_arm_mount_joint", "obotx_left_joint_slider_left_slide",
    "obotx_left_joint_slider_left_hinge", "obotx_left_joint_telescopic_slide",
    "obotx_left_joint_hinge_telescopic_hand", "obotx_left_palm_wrist_roll_joint",
    "obotx_left_palm_wrist_pitch_joint", "obotx_left_palm_wrist_yaw_joint"
]

# Right Arm Config
RIGHT_TWIST_TOPIC = "/servo_node_right/delta_twist_cmds"
RIGHT_POSE_TOPIC = "/servo_node_right/pose_cmd"
RIGHT_JOINT_TOPIC = "/servo_node_right/delta_joint_cmds"
RIGHT_SWITCH_SERVICE = "/servo_node_right/switch_command_type"
RIGHT_EE_FRAME = "obotx_right_tool0"
RIGHT_JOINTS = [
    "obotx_right_arm_mount_joint", "obotx_right_joint_slider_left_slide",
    "obotx_right_joint_slider_left_hinge", "obotx_right_joint_telescopic_slide",
    "obotx_right_joint_hinge_telescopic_hand", "obotx_right_palm_wrist_roll_joint",
    "obotx_right_palm_wrist_pitch_joint", "obotx_right_palm_wrist_yaw_joint"
]

PLANNING_FRAME_ID = "odom_gt"


class KeyboardServoPython(Node):
    def __init__(self):
        super().__init__('servo_keyboard_input')

        # Publishers
        self.twist_pubs = {
            'left': self.create_publisher(TwistStamped, LEFT_TWIST_TOPIC, 10),
            'right': self.create_publisher(TwistStamped, RIGHT_TWIST_TOPIC, 10)
        }
        self.pose_pubs = {
            'left': self.create_publisher(PoseStamped, LEFT_POSE_TOPIC, 10),
            'right': self.create_publisher(PoseStamped, RIGHT_POSE_TOPIC, 10)
        }
        self.joint_pubs = {
            'left': self.create_publisher(JointJog, LEFT_JOINT_TOPIC, 10),
            'right': self.create_publisher(JointJog, RIGHT_JOINT_TOPIC, 10)
        }

        # Marker Publisher for Pose Visualization
        self.marker_pub = self.create_publisher(Marker, '/servo_pose_markers', 10)

        # Service Clients
        self.switch_clients = {
            'left': self.create_client(ServoCommandType, LEFT_SWITCH_SERVICE),
            'right': self.create_client(ServoCommandType, RIGHT_SWITCH_SERVICE)
        }

        # TF Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State
        self.target_arm = 'left'
        self.mode = 'twist'
        self.joint_vel_cmd = 0.5
        self.command_frame_id = PLANNING_FRAME_ID

        # Pose Targets
        self.pose_targets = {
            'left': PoseStamped(),
            'right': PoseStamped()
        }
        self.pose_targets['left'].header.frame_id = PLANNING_FRAME_ID
        self.pose_targets['right'].header.frame_id = PLANNING_FRAME_ID

        self.get_logger().info("Dual-Arm Servo Keyboard Node Initialized.")
        self.print_help()

    def print_help(self):
        print("\n===========================================")
        print(" Dual-Arm Servo Keyboard Node")
        print("===========================================")
        print(" 'L' : Target LEFT  |  'R' : Target RIGHT")
        print(" MODES:")
        print("   't' : Twist (Velocity)")
        print("   'j' : Joint (Joint Velocity)")
        print("   'p' : Pose (Native MoveIt Target Pose)")
        print(" CONTROLS:")
        print("   Arrows + u/d : Move X/Y/Z")
        print("   1-8          : Joint jog (in Joint Mode)")
        print("   'Q'          : Quit")
        print("===========================================\n")
        self.update_status_msg()

    def update_status_msg(self):
        arm_str = "LEFT" if self.target_arm == 'left' else "RIGHT"
        self.get_logger().info(f"--> ACTIVE: {arm_str} Arm | Mode: {self.mode.upper()}")

    def get_current_ee_transform(self):
        try:
            ee_frame = LEFT_EE_FRAME if self.target_arm == 'left' else RIGHT_EE_FRAME
            return self.tf_buffer.lookup_transform(
                PLANNING_FRAME_ID, ee_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.5)
            )
        except Exception:
            return None

    def publish_pose_marker(self, arm, pose_stamped):
        marker = Marker()
        marker.header.frame_id = pose_stamped.header.frame_id
        marker.header.stamp = pose_stamped.header.stamp
        marker.ns = f"servo_pose_{arm}"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = pose_stamped.pose.position
        marker.pose.orientation = pose_stamped.pose.orientation
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        if arm == 'left':
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            marker.color.r = 0.0
            marker.color.g = 0.5
            marker.color.b = 1.0
        marker.color.a = 0.8
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0
        self.marker_pub.publish(marker)

    def run_keyboard_loop(self):
        old_settings = termios.tcgetattr(sys.stdin)
        rate = self.create_rate(PUBLISH_RATE)

        try:
            while rclpy.ok():
                tty.setraw(sys.stdin.fileno())
                rlist, _, _ = select.select([sys.stdin], [], [], 0.01)

                current_twist = TwistStamped()
                current_twist.header.frame_id = self.command_frame_id
                key_pressed = False

                if rlist:
                    key = sys.stdin.read(1)
                    key_pressed = True

                    # --- FIX: non-blocking escape sequence ---
                    if key == '\x1b':
                        key = 'ESC'
                        ready2, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if ready2:
                            c2 = sys.stdin.read(1)
                            ready3, _, _ = select.select([sys.stdin], [], [], 0.05)
                            if ready3:
                                c3 = sys.stdin.read(1)
                                if c2 == '[':
                                    if c3 == 'A':   key = 'UP'
                                    elif c3 == 'B': key = 'DOWN'
                                    elif c3 == 'C': key = 'RIGHT'
                                    elif c3 == 'D': key = 'LEFT'
                    # --- end fix ---

                    # Arm Switching
                    if key in ['l', 'L']:
                        self.target_arm = 'left'
                        self.update_status_msg()
                        key_pressed = False
                    elif key in ['r', 'R']:
                        self.target_arm = 'right'
                        self.update_status_msg()
                        key_pressed = False

                    # Mode Switching
                    elif key == 'j':
                        self.call_switch_service(0, 'joint')
                        self.mode = 'joint'
                        self.update_status_msg()
                        key_pressed = False
                    elif key == 't':
                        self.call_switch_service(1, 'twist')
                        self.mode = 'twist'
                        self.update_status_msg()
                        key_pressed = False
                    elif key == 'p':
                        self.call_switch_service(2, 'pose')
                        self.mode = 'pose'
                        tf_trans = self.get_current_ee_transform()
                        if tf_trans:
                            self.pose_targets[self.target_arm].pose.position = Point(
                                x=tf_trans.transform.translation.x,
                                y=tf_trans.transform.translation.y,
                                z=tf_trans.transform.translation.z
                            )
                            self.pose_targets[self.target_arm].pose.orientation = Quaternion(
                                x=tf_trans.transform.rotation.x,
                                y=tf_trans.transform.rotation.y,
                                z=tf_trans.transform.rotation.z,
                                w=tf_trans.transform.rotation.w
                            )
                            self.get_logger().info("Pose target initialized to current EE pose.")
                            self.publish_pose_marker(self.target_arm, self.pose_targets[self.target_arm])
                        self.update_status_msg()
                        key_pressed = False

                    elif key in ['q', 'Q']:
                        break

                    # Action Keys
                    elif self.mode == 'twist':
                        if key == 'UP':    current_twist.twist.linear.x = MAX_VEL
                        elif key == 'DOWN':  current_twist.twist.linear.x = -MAX_VEL
                        elif key == 'LEFT':  current_twist.twist.linear.y = MAX_VEL
                        elif key == 'RIGHT': current_twist.twist.linear.y = -MAX_VEL
                        elif key == 'u':     current_twist.twist.linear.z = MAX_VEL
                        elif key == 'd':     current_twist.twist.linear.z = -MAX_VEL
                        else: key_pressed = False

                    elif self.mode == 'pose':
                        target_pose = self.pose_targets[self.target_arm].pose
                        marker_updated = False
                        if key == 'UP':    target_pose.position.x += POSE_STEP; marker_updated = True
                        elif key == 'DOWN':  target_pose.position.x -= POSE_STEP; marker_updated = True
                        elif key == 'LEFT':  target_pose.position.y += POSE_STEP; marker_updated = True
                        elif key == 'RIGHT': target_pose.position.y -= POSE_STEP; marker_updated = True
                        elif key == 'u':     target_pose.position.z += POSE_STEP; marker_updated = True
                        elif key == 'd':     target_pose.position.z -= POSE_STEP; marker_updated = True
                        else: key_pressed = False
                        if marker_updated:
                            self.publish_pose_marker(self.target_arm, self.pose_targets[self.target_arm])

                    elif self.mode == 'joint':
                        if key in ['1', '2', '3', '4', '5', '6', '7', '8']:
                            idx = int(key) - 1
                            joint_msg = JointJog()
                            joint_msg.header.stamp = self.get_clock().now().to_msg()
                            joint_msg.header.frame_id = PLANNING_FRAME_ID
                            active_joints = LEFT_JOINTS if self.target_arm == 'left' else RIGHT_JOINTS
                            joint_msg.joint_names = active_joints
                            joint_msg.velocities = [0.0] * 8
                            joint_msg.velocities[idx] = self.joint_vel_cmd
                            self.joint_pubs[self.target_arm].publish(joint_msg)
                            key_pressed = False
                        elif key == 'r':
                            self.joint_vel_cmd *= -1
                            key_pressed = False

                # Publish commands
                now_stamp = self.get_clock().now().to_msg()
                if self.mode == 'twist':
                    current_twist.header.stamp = now_stamp
                    self.twist_pubs[self.target_arm].publish(current_twist)
                elif self.mode == 'pose':
                    self.pose_targets[self.target_arm].header.stamp = now_stamp
                    self.pose_pubs[self.target_arm].publish(self.pose_targets[self.target_arm])

                rate.sleep()

        except Exception as e:
            self.get_logger().error(f"Keyboard loop error: {e}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            if rclpy.ok():
                try:
                    stop_twist = TwistStamped()
                    stop_twist.header.stamp = self.get_clock().now().to_msg()
                    self.twist_pubs['left'].publish(stop_twist)
                    self.twist_pubs['right'].publish(stop_twist)
                except Exception:
                    pass

    def call_switch_service(self, cmd_type_int, mode_str):
        target = self.target_arm
        client = self.switch_clients[target]
        if not client.wait_for_service(timeout_sec=1.0):
            return
        request = ServoCommandType.Request()
        request.command_type = cmd_type_int
        try:
            response = client.call(request)
            if response.success:
                self.get_logger().info(f"[{target.upper()}] Switched to: {mode_str.upper()}")
        except Exception:
            pass


def main():
    rclpy.init()
    node = KeyboardServoPython()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    kb_thread = threading.Thread(target=node.run_keyboard_loop, daemon=True)
    kb_thread.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C received.")
    except Exception as e:
        node.get_logger().error(f"Spin error: {e}")
    finally:
        kb_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()
        print("\n[INFO] Node shut down cleanly.\n")


if __name__ == '__main__':
    main()
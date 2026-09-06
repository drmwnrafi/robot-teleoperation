#!/usr/bin/env python3

import rclpy
import math
import sys
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Point, Quaternion, TwistStamped, PointStamped
from visualization_msgs.msg import Marker
from moveit_msgs.srv import ServoCommandType
from landmark_msgs.msg import LandmarkMsg, HandLandmark
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
import tf2_ros
import tf2_geometry_msgs
from scipy.optimize import minimize

# ------------------------- Global Frames & Constants -------------------------
PLANNING_FRAME = "odom_gt"
HAND_DATA_FRAME = "landmark"
BASE_FRAME = "obotx_base_link_platform"
FOOTPRINT_FRAME = "obotx_base_footprint_platform"
LEFT_ROOT_LINK = "obotx_left_arm_root_link"
RIGHT_ROOT_LINK = "obotx_right_arm_root_link"

CMD_VEL_TOPIC = "/cmd_vel"

ALPHA = 0.3                  # Low‑pass filter for hand position
MAX_STEP_PER_UPDATE = 0.02   # Max movement per update (m)

# ----------------------------- ArmTracker Class -----------------------------
class ArmTracker:
    def __init__(self, node, side):
        self.state = "IDLE"
        self.side = side
        self.node = node
        self.pose_topic = f"/servo_node_{side}/pose_cmd"
        self.switch_service = f"/servo_node_{side}/switch_command_type"
        self.ee_frame = f"obotx_{side}_tool0"
        self.root_link = LEFT_ROOT_LINK if side == 'left' else RIGHT_ROOT_LINK

        # Publishers & clients
        self.pose_pub = node.create_publisher(PoseStamped, self.pose_topic, 10)
        self.switch_client = node.create_client(ServoCommandType, self.switch_service)
        self.marker_pub = node.create_publisher(Marker, f'/hand_target_marker_{side}', 10)

        self.is_tracking = False
        self.filtered_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.initialized_filter = False
        self.debug_counter = 0
        self.reachability_error = 0.0

        self.last_gesture = None

        self.gripper_joint_names = [
            f"obotx_{side}_palm_finger_1_joint",
            f"obotx_{side}_palm_finger_2_joint",
            f"obotx_{side}_finger_1_joint_1",
            f"obotx_{side}_finger_1_joint_2",
            f"obotx_{side}_finger_1_joint_3",
            f"obotx_{side}_finger_2_joint_1",
            f"obotx_{side}_finger_2_joint_2",
            f"obotx_{side}_finger_2_joint_3",
            f"obotx_{side}_finger_middle_joint_1",
            f"obotx_{side}_finger_middle_joint_2",
            f"obotx_{side}_finger_middle_joint_3",
        ]

        self.open_positions = [
            0.0, 0.0,
            0.0610865, 0.0, -0.0872665,
            0.0610865, 0.0, -0.0872665,
            0.0610865, 0.0, -0.0872665,
        ]

        self.close_positions = [
            0.0, 0.0,
            0.8066, 0.174533, -0.610865,
            0.8066, 0.174533, -0.610865,
            0.8066, 0.174533, -0.610865,
        ]

        # Create action client with correct server name
        action_name = f"/gripper_{side}_controller/follow_joint_trajectory"
        self.gripper_action_client = ActionClient(node, FollowJointTrajectory, action_name)

        if not self.gripper_action_client.wait_for_server(timeout_sec=3.0):
            node.get_logger().error(f"❌ Gripper action server {action_name} not available for {side}. Gripper disabled.")
            self.gripper_available = False
        else:
            self.gripper_available = True
            node.get_logger().info(f"✅ Gripper controller initialised for {side.upper()} using {action_name}")

        # Switch servo node to POSE mode
        self.call_switch_service()

    def call_switch_service(self):
        self.node.get_logger().info(f"Waiting for {self.side.upper()} Servo switch service...")
        if not self.switch_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error(f"❌ {self.side.upper()} Servo switch service not available!")
            return
        request = ServoCommandType.Request()
        request.command_type = ServoCommandType.Request.POSE
        future = self.switch_client.call_async(request)
        future.add_done_callback(self.switch_callback)

    def switch_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.node.get_logger().info(f"Successfully switched {self.side.upper()} MoveIt Servo to POSE mode.")
        except Exception as e:
            self.node.get_logger().error(f"Service call failed for {self.side.upper()}: {e}")

    def send_gripper_command(self, state, duration=0.1):
        if not self.gripper_available:
            return
        try:
            goal_msg = FollowJointTrajectory.Goal()
            goal_msg.trajectory = JointTrajectory()
            goal_msg.trajectory.joint_names = self.gripper_joint_names

            point = JointTrajectoryPoint()
            if state == "open":
                point.positions = self.open_positions
            elif state == "close":
                point.positions = self.close_positions
            else:
                return

            point.time_from_start = Duration(seconds=duration).to_msg()
            goal_msg.trajectory.points = [point] 

            self.gripper_action_client.send_goal_async(goal_msg)
            self.node.get_logger().debug(f"{self.side.upper()} gripper -> {state} in {duration}s")
        except Exception as e:
            self.node.get_logger().error(f"Gripper command failed for {self.side.upper()}: {e}")

    def get_current_ee_orientation(self):
        try:
            trans = self.node.tf_buffer.lookup_transform(PLANNING_FRAME, self.ee_frame,
                                                         rclpy.time.Time(), timeout=Duration(seconds=1.0))
            return trans.transform.rotation
        except Exception:
            return None

    def transform_to_planning_frame(self, x, y, z):
        if HAND_DATA_FRAME == PLANNING_FRAME:
            return float(x), float(y), float(z)
        try:
            pose_in = PoseStamped()
            pose_in.header.frame_id = HAND_DATA_FRAME
            pose_in.header.stamp = rclpy.time.Time()
            pose_in.pose.position.x = float(x)
            pose_in.pose.position.y = float(y)
            pose_in.pose.position.z = float(z)
            pose_in.pose.orientation.w = 1.0
            pose_out = self.node.tf_buffer.transform(pose_in, PLANNING_FRAME, timeout=Duration(seconds=1.0))
            return pose_out.pose.position.x, pose_out.pose.position.y, pose_out.pose.position.z
        except Exception as e:
            if self.debug_counter % 50 == 0:
                self.node.get_logger().warn(f"[{self.side.upper()}] TF transform failed. Error: {e}")
            return None, None, None

    def publish_target_marker(self, pose_msg: PoseStamped):
        try:
            root_tf = self.node.tf_buffer.lookup_transform(PLANNING_FRAME, self.root_link,
                                                           rclpy.time.Time(), timeout=Duration(seconds=1.0))
            root_x = root_tf.transform.translation.x
            root_y = root_tf.transform.translation.y
            root_z = root_tf.transform.translation.z
        except Exception:
            return
        target_x = pose_msg.pose.position.x
        target_y = pose_msg.pose.position.y
        target_z = pose_msg.pose.position.z

        dist_xy = math.hypot(target_x - root_x, target_y - root_y)
        dist_xyz = math.hypot(target_x - root_x, target_y - root_y, target_z - root_z)

        p_root = Point(x=root_x, y=root_y, z=root_z)
        p_corner = Point(x=target_x, y=target_y, z=root_z)
        p_target = Point(x=target_x, y=target_y, z=target_z)

        line_marker = Marker()
        line_marker.header.frame_id = PLANNING_FRAME
        line_marker.header.stamp = pose_msg.header.stamp
        line_marker.ns = f"hand_target_line_{self.side}"
        line_marker.id = 1
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.01
        if self.side == 'left':
            line_marker.color.r = 0.0; line_marker.color.g = 1.0; line_marker.color.b = 0.0
        else:
            line_marker.color.r = 0.0; line_marker.color.g = 0.5; line_marker.color.b = 1.0
        line_marker.color.a = 0.8
        line_marker.points = [p_root, p_corner, p_target, p_root]
        self.marker_pub.publish(line_marker)

        text_marker = Marker()
        text_marker.header.frame_id = PLANNING_FRAME
        text_marker.header.stamp = pose_msg.header.stamp
        text_marker.ns = f"hand_target_text_{self.side}"
        text_marker.id = 2
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.scale.z = 0.05
        if self.side == 'left':
            text_marker.color.r = 0.0; text_marker.color.g = 1.0; text_marker.color.b = 0.0
        else:
            text_marker.color.r = 0.0; text_marker.color.g = 0.5; text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.pose.position = Point(x=target_x, y=target_y, z=root_z + 0.05)
        text_marker.pose.orientation.w = 1.0
        text_marker.text = f"XY:{dist_xy:.2f}m\nXYZ:{dist_xyz:.2f}m"
        self.marker_pub.publish(text_marker)

    def update(self, hand_data: HandLandmark):
        workspace_recovery = ((self.side == "left" and self.node.base_controller.left_workspace_recovery) or
                              (self.side == "right" and self.node.base_controller.right_workspace_recovery))

        self.debug_counter += 1
        if not hand_data.present:
            self.state = "NO_HAND"
            self.is_tracking = False
            self.reachability_error = 0.0
            self.last_gesture = None
            return

        self.is_tracking = True

        # ----- Gesture-based gripper command -----
        if hand_data.gesture:
            gesture = hand_data.gesture
            if gesture != self.last_gesture:
                if gesture == "GRAB":
                    self.send_gripper_command("close", duration=0.1)
                elif gesture == "OPEN":
                    self.send_gripper_command("open", duration=0.1)
                self.last_gesture = gesture

        # ----- Position tracking (same logic as original) -----
        raw_x, raw_y, raw_z = hand_data.wrist_m.x, hand_data.wrist_m.y, hand_data.wrist_m.z
        trans_x, trans_y, trans_z = self.transform_to_planning_frame(raw_x, raw_y, raw_z)
        if trans_x is None:
            return

        try:
            root_tf = self.node.tf_buffer.lookup_transform(PLANNING_FRAME, self.root_link,
                                                           rclpy.time.Time(), timeout=Duration(seconds=1.0))
            root_x = root_tf.transform.translation.x
            root_y = root_tf.transform.translation.y
            root_z = root_tf.transform.translation.z
        except Exception:
            return

        dx = trans_x - root_x
        dy = trans_y - root_y
        dist_xy = math.hypot(dx, dy)

        if workspace_recovery and dist_xy < self.node.base_controller.min_xy_distance:
            self.state = "WORKSPACE_RECOVERY"
            if dist_xy > 1e-6:
                scale = self.node.base_controller.min_xy_distance / dist_xy
                trans_x = root_x + dx * scale
                trans_y = root_y + dy * scale

        if not self.node.base_controller.freeze_arms:
            self.state = "TRACKING"
            if not self.initialized_filter:
                self.filtered_pos['x'], self.filtered_pos['y'], self.filtered_pos['z'] = trans_x, trans_y, trans_z
                self.initialized_filter = True
            else:
                target_x = ALPHA * trans_x + (1.0 - ALPHA) * self.filtered_pos['x']
                target_y = ALPHA * trans_y + (1.0 - ALPHA) * self.filtered_pos['y']
                target_z = ALPHA * trans_z + (1.0 - ALPHA) * self.filtered_pos['z']

                dx_f = target_x - self.filtered_pos['x']
                dy_f = target_y - self.filtered_pos['y']
                dz_f = target_z - self.filtered_pos['z']
                dist = math.hypot(dx_f, dy_f, dz_f)
                if dist > MAX_STEP_PER_UPDATE:
                    scale = MAX_STEP_PER_UPDATE / dist
                    dx_f *= scale; dy_f *= scale; dz_f *= scale

                self.filtered_pos['x'] += dx_f
                self.filtered_pos['y'] += dy_f
                self.filtered_pos['z'] += dz_f
        else:
            self.state = "FROZEN"
            try:
                ee_tf = self.node.tf_buffer.lookup_transform(PLANNING_FRAME, self.ee_frame,
                                                             rclpy.time.Time(), timeout=Duration(seconds=1.0))
                curr_x = ee_tf.transform.translation.x
                curr_y = ee_tf.transform.translation.y

                dx_ee = curr_x - root_x
                dy_ee = curr_y - root_y
                dist_ee_xy = math.hypot(dx_ee, dy_ee)
                min_xy = self.node.base_controller.min_xy_distance

                if dist_ee_xy < min_xy:
                    if dist_ee_xy > 1e-6:
                        scale = min_xy / dist_ee_xy
                        self.filtered_pos['x'] = root_x + dx_ee * scale
                        self.filtered_pos['y'] = root_y + dy_ee * scale
                    else:
                        self.filtered_pos['x'] = root_x + min_xy
                        self.filtered_pos['y'] = root_y
                else:
                    self.filtered_pos['x'] = curr_x
                    self.filtered_pos['y'] = curr_y
                self.filtered_pos['z'] = trans_z
            except Exception:
                pass

        # Send pose command to servo node
        current_rot = self.get_current_ee_orientation()
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = PLANNING_FRAME
        pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        pose_msg.pose.position = Point(x=self.filtered_pos['x'], y=self.filtered_pos['y'], z=self.filtered_pos['z'])
        if current_rot:
            pose_msg.pose.orientation = Quaternion(x=current_rot.x, y=current_rot.y,
                                                   z=current_rot.z, w=current_rot.w)
        else:
            pose_msg.pose.orientation.w = 1.0

        self.pose_pub.publish(pose_msg)
        self.publish_target_marker(pose_msg)

        # Compute reachability error (optional)
        try:
            current_ee_tf = self.node.tf_buffer.lookup_transform(PLANNING_FRAME, self.ee_frame,
                                                                 rclpy.time.Time(), timeout=Duration(seconds=1.0))
            curr_x = current_ee_tf.transform.translation.x
            curr_y = current_ee_tf.transform.translation.y
            self.reachability_error = math.hypot(curr_x - self.filtered_pos['x'],
                                                 curr_y - self.filtered_pos['y'])
        except Exception:
            self.reachability_error = 0.0


# ----------------------------- BaseController Class -----------------------------
# (Unchanged, kept for completeness, but currently not used – see comment in callback)
class BaseController:
    def __init__(self, node):
        self.state = "IDLE"
        self.node = node
        self.cmd_vel_topic = CMD_VEL_TOPIC
        self.cmd_vel_pub = node.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        self.target_base_marker_pub = node.create_publisher(Marker, '/target_base_marker', 10)
        node.get_logger().info("✅ SciPy Region-Based Base Optimizer Initialized (currently disabled)")

        self.min_xy_distance = 0.40
        self.max_xy_distance = 1.0
        self.XY_TOLERANCE = 0.10
        self.ORI_TOLERANCE = 0.15
        self.w_close = 5.0
        self.w_far = 1.0
        self.w_angle = 4.0
        self.w_yaw = 2.0
        self.w_reg = 0.5
        self.kp_xy = 0.6
        self.kp_yaw = 0.8
        self.max_xy_speed = 0.3
        self.max_yaw_speed = 0.5
        self.smoothed_cmd = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.alpha_base = 0.3
        self.yaw_alpha = 0.2
        self.warn_counter = 0
        self.freeze_arms = False
        self.left_workspace_recovery = False
        self.right_workspace_recovery = False
        self.filtered_human_yaw = None

    def _wrap_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _publish_target_base_marker(self, tx, ty, tyaw):
        try:
            base_tf = self.node.tf_buffer.lookup_transform(PLANNING_FRAME, FOOTPRINT_FRAME,
                                                           rclpy.time.Time(), timeout=Duration(seconds=1.0))
            tz = base_tf.transform.translation.z
            arrow_marker = Marker()
            arrow_marker.header.frame_id = PLANNING_FRAME
            arrow_marker.header.stamp = self.node.get_clock().now().to_msg()
            arrow_marker.ns = "target_base_arrow"
            arrow_marker.id = 1
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            p_start = Point(x=tx, y=ty, z=tz)
            arrow_length = 0.8
            p_end = Point(x=tx + arrow_length * math.cos(tyaw),
                          y=ty + arrow_length * math.sin(tyaw),
                          z=tz)
            arrow_marker.points = [p_start, p_end]
            arrow_marker.scale.x = 0.08
            arrow_marker.scale.y = 0.15
            arrow_marker.scale.z = 0.0
            arrow_marker.color.r = 0.0
            arrow_marker.color.g = 0.4
            arrow_marker.color.b = 1.0
            arrow_marker.color.a = 0.9
            self.target_base_marker_pub.publish(arrow_marker)

            text_marker = Marker()
            text_marker.header.frame_id = PLANNING_FRAME
            text_marker.header.stamp = self.node.get_clock().now().to_msg()
            text_marker.ns = "target_base_text"
            text_marker.id = 2
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.scale.z = 0.1
            text_marker.color.r = 0.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.pose.position = Point(x=tx, y=ty, z=tz + 0.25)
            text_marker.pose.orientation.w = 1.0
            text_marker.text = f"Target Base\n{math.degrees(tyaw):.1f}°"
            self.target_base_marker_pub.publish(text_marker)
        except Exception:
            pass

    def _cost_function(self, q, targets_info, q_curr, human_yaw):
        x, y, yaw = q
        cost = 0.0
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        for tx, ty, is_left in targets_info:
            t_rel_x = (tx - x) * cos_y + (ty - y) * sin_y
            t_rel_y = -(tx - x) * sin_y + (ty - y) * cos_y
            d = math.hypot(t_rel_x, t_rel_y)
            if d < self.min_xy_distance:
                cost += self.w_close * (self.min_xy_distance - d) ** 2
            elif d > self.max_xy_distance:
                cost += self.w_far * (d - self.max_xy_distance) ** 2
            phi = math.atan2(t_rel_y, t_rel_x)
            if is_left:
                min_phi, max_phi = 0.0, math.pi / 2.0
            else:
                min_phi, max_phi = -math.pi / 2.0, 0.0
            if phi < min_phi:
                cost += self.w_angle * (phi - min_phi) ** 2
            elif phi > max_phi:
                cost += self.w_angle * (phi - max_phi) ** 2
        yaw_err = self._wrap_angle(yaw - human_yaw)
        cost += self.w_yaw * (yaw_err) ** 2
        dx_reg = x - q_curr[0]
        dy_reg = y - q_curr[1]
        dyaw_reg = self._wrap_angle(yaw - q_curr[2])
        cost += self.w_reg * (dx_reg ** 2 + dy_reg ** 2 + 0.5 * dyaw_reg ** 2)
        return cost

    def update(self, msg: LandmarkMsg, left_target=None, right_target=None, left_error=0.0, right_error=0.0):
        # This method is currently not called (see landmark_callback). Kept for potential future use.
        pass


# ----------------------------- Main ROS2 Node -----------------------------
class HandPoseServoNode(Node):
    def __init__(self):
        super().__init__('hand_pose_tracker')
        self.declare_parameter('closest_target', True)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.left_tracker = ArmTracker(self, 'left')
        self.right_tracker = ArmTracker(self, 'right')
        self.base_controller = BaseController(self)
        self.create_timer(1.0, self.log_states)

        self.human_yaw_marker_pub = self.create_publisher(Marker, '/human_yaw_marker', 10)
        self.landmark_sub = self.create_subscription(LandmarkMsg, '/processed_landmarks',
                                                     self.landmark_callback, 10)
        self.get_logger().info("Dual-Arm Hand Pose Tracker + Gripper Control Initialized.")

    def log_states(self):
        self.get_logger().info(
            f"BASE={self.base_controller.state} | "
            f"LEFT={self.left_tracker.state} | "
            f"RIGHT={self.right_tracker.state} | "
            f"Lerr={self.left_tracker.reachability_error:.2f} | "
            f"Rerr={self.right_tracker.reachability_error:.2f}"
        )

    def publish_human_yaw_marker(self, msg: LandmarkMsg):
        body_data = {bl.joint_name: [bl.x, bl.y, bl.z] for bl in msg.body_landmarks}
        if "shoulder_L" not in body_data or "shoulder_R" not in body_data:
            return
        try:
            sL, sR = body_data["shoulder_L"], body_data["shoulder_R"]
            left_shoulder = PointStamped()
            left_shoulder.header.frame_id = HAND_DATA_FRAME
            left_shoulder.point.x, left_shoulder.point.y, left_shoulder.point.z = float(sL[0]), float(sL[1]), float(sL[2])
            right_shoulder = PointStamped()
            right_shoulder.header.frame_id = HAND_DATA_FRAME
            right_shoulder.point.x, right_shoulder.point.y, right_shoulder.point.z = float(sR[0]), float(sR[1]), float(sR[2])

            left_odom = self.tf_buffer.transform(left_shoulder, PLANNING_FRAME, timeout=Duration(seconds=1.0))
            right_odom = self.tf_buffer.transform(right_shoulder, PLANNING_FRAME, timeout=Duration(seconds=1.0))

            sx = right_odom.point.x - left_odom.point.x
            sy = right_odom.point.y - left_odom.point.y
            mag_shoulder = math.hypot(sx, sy)
            if mag_shoulder > 0.1:
                sx_norm, sy_norm = sx / mag_shoulder, sy / mag_shoulder
            else:
                sx_norm, sy_norm = 0.0, 0.0
            forward_x, forward_y = -sy_norm, sx_norm
            human_yaw_raw = math.atan2(forward_y, forward_x)

            center_x = (left_odom.point.x + right_odom.point.x) / 2.0
            center_y = (left_odom.point.y + right_odom.point.y) / 2.0
            center_z = (left_odom.point.z + right_odom.point.z) / 2.0

            arrow_marker = Marker()
            arrow_marker.header.frame_id = PLANNING_FRAME
            arrow_marker.header.stamp = self.get_clock().now().to_msg()
            arrow_marker.ns = "human_yaw_arrow"
            arrow_marker.id = 1
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            arrow_marker.points = [
                Point(x=center_x, y=center_y, z=center_z),
                Point(x=center_x + forward_x * 0.6, y=center_y + forward_y * 0.6, z=center_z)
            ]
            arrow_marker.scale.x, arrow_marker.scale.y, arrow_marker.scale.z = 0.05, 0.1, 0.0
            arrow_marker.color.r, arrow_marker.color.g, arrow_marker.color.b, arrow_marker.color.a = 1.0, 0.5, 0.0, 1.0
            self.human_yaw_marker_pub.publish(arrow_marker)

            text_marker = Marker()
            text_marker.header.frame_id = PLANNING_FRAME
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "human_yaw_text"
            text_marker.id = 2
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.scale.z = 0.05
            text_marker.color.r, text_marker.color.g, text_marker.color.b, text_marker.color.a = 1.0, 0.0, 0.0, 1.0
            text_marker.pose.position = Point(x=center_x, y=center_y, z=center_z)
            text_marker.pose.orientation.w = 1.0
            text_marker.text = f"Human:{math.degrees(human_yaw_raw):.1f}°"
            self.human_yaw_marker_pub.publish(text_marker)
        except Exception:
            pass

    def landmark_callback(self, msg: LandmarkMsg):
        closest_target = self.get_parameter('closest_target').value

        try:
            left_root_tf = self.tf_buffer.lookup_transform(HAND_DATA_FRAME, LEFT_ROOT_LINK,
                                                           rclpy.time.Time(), timeout=Duration(seconds=1.0))
            right_root_tf = self.tf_buffer.lookup_transform(HAND_DATA_FRAME, RIGHT_ROOT_LINK,
                                                            rclpy.time.Time(), timeout=Duration(seconds=1.0))
            left_root_pos = (left_root_tf.transform.translation.x, left_root_tf.transform.translation.y,
                             left_root_tf.transform.translation.z)
            right_root_pos = (right_root_tf.transform.translation.x, right_root_tf.transform.translation.y,
                              right_root_tf.transform.translation.z)
        except Exception as e:
            self.get_logger().warn(f"⚠️ Failed to get arm root TF. Error: {e}")
            return

        def get_dist(hand_data, root_pos):
            return math.hypot(hand_data.wrist_m.x - root_pos[0],
                              hand_data.wrist_m.y - root_pos[1],
                              hand_data.wrist_m.z - root_pos[2])

        empty_hand = HandLandmark()
        empty_hand.present = False

        if not closest_target:
            self.left_tracker.update(msg.left_hand)
            self.right_tracker.update(msg.right_hand)
        else:
            hands = []
            if msg.left_hand.present:
                hands.append(('left', msg.left_hand))
            if msg.right_hand.present:
                hands.append(('right', msg.right_hand))
            if len(hands) == 0:
                self.left_tracker.update(empty_hand)
                self.right_tracker.update(empty_hand)
            elif len(hands) == 1:
                hand_name, hand_data = hands[0]
                if get_dist(hand_data, left_root_pos) < get_dist(hand_data, right_root_pos):
                    self.left_tracker.update(hand_data)
                    self.right_tracker.update(empty_hand)
                else:
                    self.right_tracker.update(hand_data)
                    self.left_tracker.update(empty_hand)
            else:  # two hands
                h_left = hands[0][1] if hands[0][0] == 'left' else hands[1][1]
                h_right = hands[0][1] if hands[0][0] == 'right' else hands[1][1]
                if get_dist(h_left, right_root_pos) + get_dist(h_right, left_root_pos) < \
                   get_dist(h_left, left_root_pos) + get_dist(h_right, right_root_pos):
                    self.right_tracker.update(h_left)
                    self.left_tracker.update(h_right)
                else:
                    self.left_tracker.update(h_left)
                    self.right_tracker.update(h_right)

        # Uncomment the following line to enable base controller:
        # self.base_controller.update(msg, left_target, right_target, left_error, right_error)
        # self.publish_human_yaw_marker(msg)


# ----------------------------- Main Entry Point -----------------------------
def main():
    rclpy.init()
    node = HandPoseServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from landmark_msgs.msg import LandmarkMsg, HandLandmark, BodyLandmark
from geometry_msgs.msg import Point
import json

class LandmarkProcessorNode(Node):
    def __init__(self):
        super().__init__('landmark_processor_node')
        
        self.declare_parameter('fix_x', True)
        self.declare_parameter('fix_y', True)
        self.declare_parameter('fix_z', False)
        
        self.fix_x = self.get_parameter('fix_x').value
        self.fix_y = self.get_parameter('fix_y').value
        self.fix_z = self.get_parameter('fix_z').value
        
        self.get_logger().info(f"Origin fixing enabled: X={self.fix_x}, Y={self.fix_y}, Z={self.fix_z}")
        
        self.subscription = self.create_subscription(
            String, '/raw_landmarks', self.raw_callback, 10)
        
        self.publisher = self.create_publisher(LandmarkMsg, '/processed_landmarks', 10)
        self.left_hand_pub = self.create_publisher(HandLandmark, '/left_hand', 10)
        self.right_hand_pub = self.create_publisher(HandLandmark, '/right_hand', 10)
        
        self.alpha = 0.7  # 70% history, 30% new data
        
        self.prev_wrist_pos = {
            'Left': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'Right': {'x': 0.0, 'y': 0.0, 'z': 0.0}
        }

        self.prev_joints = {
            'Left': [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(21)],
            'Right': [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(21)]
        }
        self.is_initialized = {'Left': False, 'Right': False}

        self.prev_body = {}
        self.is_body_initialized = False

    @staticmethod
    def _js_to_ros(x, y, z):
        """Transforms MediaPipe/JS coordinates (x=right, y=down, z=forward) 
           to ROS coordinates (x=forward, y=left, z=up)."""
        return float(z), -float(x), float(y)
    
    def get_distance(self, p1, p2):
        return ((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)**0.5

    def _smooth_and_format_hand(self, hand, side_label, shoulder_mid) -> HandLandmark:
        wx, wy, wz = hand["wrist_cm"]["x"], hand["wrist_cm"]["y"], hand["wrist_cm"]["z"]
        ros_wx, ros_wy, ros_wz = self._js_to_ros(wx, wy, wz)

        ros_wx -= shoulder_mid[0]
        ros_wy -= shoulder_mid[1]
        ros_wz -= shoulder_mid[2]

        if not self.is_initialized[side_label]:
            self.prev_wrist_pos[side_label] = {'x': ros_wx, 'y': ros_wy, 'z': ros_wz}
            
            joints = hand.get("joints_cm", [])
            if len(joints) == 21:
                for i in range(21):
                    jx, jy, jz = joints[i]['x'], joints[i]['y'], joints[i]['z']
                    rjx, rjy, rjz = self._js_to_ros(jx, jy, jz)
                    
                    rjx -= shoulder_mid[0]
                    rjy -= shoulder_mid[1]
                    rjz -= shoulder_mid[2]
                    
                    self.prev_joints[side_label][i] = {'x': rjx, 'y': rjy, 'z': rjz}
            self.is_initialized[side_label] = True
        
        self.prev_wrist_pos[side_label]['x'] = self.alpha * ros_wx + (1 - self.alpha) * self.prev_wrist_pos[side_label]['x']
        self.prev_wrist_pos[side_label]['y'] = self.alpha * ros_wy + (1 - self.alpha) * self.prev_wrist_pos[side_label]['y']
        self.prev_wrist_pos[side_label]['z'] = self.alpha * ros_wz + (1 - self.alpha) * self.prev_wrist_pos[side_label]['z']

        smoothed_wrist_m = Point()
        smoothed_wrist_m.x = round(self.prev_wrist_pos[side_label]['x'] / 100.0, 4)
        smoothed_wrist_m.y = round(self.prev_wrist_pos[side_label]['y'] / 100.0, 4)
        smoothed_wrist_m.z = round(self.prev_wrist_pos[side_label]['z'] / 100.0, 4)

        joints = hand.get("joints_cm", [])
        smoothed_joints_m = []
        if len(joints) == 21:
            for i in range(21):
                jx, jy, jz = joints[i]['x'], joints[i]['y'], joints[i]['z']
                rjx, rjy, rjz = self._js_to_ros(jx, jy, jz)
                
                rjx -= shoulder_mid[0]
                rjy -= shoulder_mid[1]
                rjz -= shoulder_mid[2]
                
                pj = self.prev_joints[side_label][i]
                pj['x'] = self.alpha * rjx + (1 - self.alpha) * pj['x']
                pj['y'] = self.alpha * rjy + (1 - self.alpha) * pj['y']
                pj['z'] = self.alpha * rjz + (1 - self.alpha) * pj['z']
                
                pt = Point()
                pt.x = round(pj['x'] / 100.0, 4)
                pt.y = round(pj['y'] / 100.0, 4)
                pt.z = round(pj['z'] / 100.0, 4)
                smoothed_joints_m.append(pt)

        msg = HandLandmark()
        msg.present = True
        msg.confidence = round(hand.get("confidence", 0.0), 3)
        msg.depth_m = round(hand.get("depth_cm", 0.0) / 100.0, 4)
        msg.wrist_m = smoothed_wrist_m
        msg.joints_m = smoothed_joints_m
        
        if "palm_normal" in hand and isinstance(hand["palm_normal"], list) and len(hand["palm_normal"]) == 3:
            pn = hand["palm_normal"]
            msg.palm_normal = list(self._js_to_ros(pn[0], pn[1], pn[2]))
        else:
            msg.palm_normal = hand.get("palm_normal", [])
            
        if "finger_dir" in hand and isinstance(hand["finger_dir"], list) and len(hand["finger_dir"]) == 3:
            fd = hand["finger_dir"]
            msg.finger_dir = list(self._js_to_ros(fd[0], fd[1], fd[2]))
        else:
            msg.finger_dir = hand.get("finger_dir", [])
            
        msg.finger_curl = hand.get("finger_curl", [])
        msg.pinch_m = [round(p / 100.0, 4) for p in hand.get("pinch_cm", [0.0, 0.0, 0.0, 0.0])]
        msg.grip_aperture_m = round(hand.get("grip_aperture_cm", 0.0) / 100.0, 4)
        msg.gesture = str(hand.get("gesture") or "UNKNOWN")
        msg.gesture_id = int(hand.get("gesture_id", 0))
        msg.is_grab = bool(hand.get("is_grab", False))
        
        return msg

    def raw_callback(self, msg: String):
        try:
            raw_data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("Failed to parse JSON")
            return

        present_hands = [h for h in raw_data.get("hands", []) if h.get("present")]
        assigned_left = None
        assigned_right = None

        shoulder_mid = [0.0, 0.0, 0.0]
        raw_body = raw_data.get("body", {})
        
        ls_key = None
        rs_key = None
        for k in raw_body.keys():
            k_lower = k.lower()
            if "shoulder" in k_lower:
                if "left" in k_lower or k_lower.endswith("_l"):
                    ls_key = k
                elif "right" in k_lower or k_lower.endswith("_r"):
                    rs_key = k
        
        if ls_key and rs_key:
            ls = raw_body[ls_key]
            rs = raw_body[rs_key]
            if isinstance(ls, list) and len(ls) == 3 and isinstance(rs, list) and len(rs) == 3:
                rls_x, rls_y, rls_z = self._js_to_ros(ls[0], ls[1], ls[2])
                rrs_x, rrs_y, rrs_z = self._js_to_ros(rs[0], rs[1], rs[2])
                
                shoulder_mid = [
                    (rls_x + rrs_x) / 2.0 if self.fix_x else 0.0,
                    (rls_y + rrs_y) / 2.0 if self.fix_y else 0.0,
                    (rls_z + rrs_z) / 2.0 if self.fix_z else 0.0
                ]

        def get_ros_pos(h):
            wx, wy, wz = h["wrist_cm"]["x"], h["wrist_cm"]["y"], h["wrist_cm"]["z"]
            rx, ry, rz = self._js_to_ros(wx, wy, wz)
            rx -= shoulder_mid[0]
            ry -= shoulder_mid[1]
            rz -= shoulder_mid[2]
            return {'x': rx, 'y': ry, 'z': rz}, wx

        if len(present_hands) == 1:
            hand = present_hands[0]
            pos_ros, js_x = get_ros_pos(hand)
            
            dist_left = self.get_distance(pos_ros, self.prev_wrist_pos["Left"]) if self.is_initialized["Left"] else float('inf')
            dist_right = self.get_distance(pos_ros, self.prev_wrist_pos["Right"]) if self.is_initialized["Right"] else float('inf')
            
            if not self.is_initialized["Left"] and not self.is_initialized["Right"]:
                if js_x < 0:
                    assigned_left = hand
                    self.is_initialized["Left"] = True
                else:
                    assigned_right = hand
                    self.is_initialized["Right"] = True
            elif dist_left < dist_right:
                assigned_left = hand
            else:
                assigned_right = hand

        elif len(present_hands) == 2:
            h1 = present_hands[0]
            h2 = present_hands[1]
            
            p1_ros, js_x1 = get_ros_pos(h1)
            p2_ros, js_x2 = get_ros_pos(h2)
            
            if not self.is_initialized["Left"] and not self.is_initialized["Right"]:
                if js_x1 < js_x2:
                    assigned_left, assigned_right = h1, h2
                else:
                    assigned_left, assigned_right = h2, h1
                self.is_initialized["Left"] = True
                self.is_initialized["Right"] = True
            else:
                d1L = self.get_distance(p1_ros, self.prev_wrist_pos["Left"]) if self.is_initialized["Left"] else float('inf')
                d1R = self.get_distance(p1_ros, self.prev_wrist_pos["Right"]) if self.is_initialized["Right"] else float('inf')
                d2L = self.get_distance(p2_ros, self.prev_wrist_pos["Left"]) if self.is_initialized["Left"] else float('inf')
                d2R = self.get_distance(p2_ros, self.prev_wrist_pos["Right"]) if self.is_initialized["Right"] else float('inf')
                
                if (d1L + d2R) < (d1R + d2L):
                    assigned_left, assigned_right = h1, h2
                else:
                    assigned_left, assigned_right = h2, h1

        smoothed_body_msgs = []
        
        if not self.is_body_initialized and raw_body:
            for joint, coords in raw_body.items():
                if isinstance(coords, list) and len(coords) == 3:
                    rx, ry, rz = self._js_to_ros(coords[0], coords[1], coords[2])
                    rx -= shoulder_mid[0]
                    ry -= shoulder_mid[1]
                    rz -= shoulder_mid[2]
                    self.prev_body[joint] = [rx, ry, rz]
            self.is_body_initialized = True
            
        for joint, coords in raw_body.items():
            if isinstance(coords, list) and len(coords) == 3:
                if joint not in self.prev_body:
                    rx, ry, rz = self._js_to_ros(coords[0], coords[1], coords[2])
                    rx -= shoulder_mid[0]
                    ry -= shoulder_mid[1]
                    rz -= shoulder_mid[2]
                    self.prev_body[joint] = [rx, ry, rz]
                
                nx, ny, nz = self._js_to_ros(coords[0], coords[1], coords[2])
                
                nx -= shoulder_mid[0]
                ny -= shoulder_mid[1]
                nz -= shoulder_mid[2]
                
                px, py, pz = self.prev_body[joint]
                
                sx = self.alpha * nx + (1 - self.alpha) * px
                sy = self.alpha * ny + (1 - self.alpha) * py
                sz = self.alpha * nz + (1 - self.alpha) * pz
                
                self.prev_body[joint] = [sx, sy, sz]
                
                body_msg = BodyLandmark()
                body_msg.joint_name = joint
                body_msg.x = round(sx / 100.0, 4)
                body_msg.y = round(sy / 100.0, 4)
                body_msg.z = round(sz / 100.0, 4)
                smoothed_body_msgs.append(body_msg)

        out_msg = LandmarkMsg()
        out_msg.t = float(raw_data.get("t", 0.0))
        out_msg.seq = int(raw_data.get("seq", 0))
        out_msg.fps = float(raw_data.get("fps", 0.0))
        
        out_msg.frame_info = json.dumps(raw_data.get("frame", {}))
        out_msg.calibration_info = json.dumps(raw_data.get("calibration", {}))
        out_msg.landmark_names = list(raw_data.get("landmark_names", {}).keys())
        out_msg.body_landmarks = smoothed_body_msgs

        if assigned_left:
            left_hand_msg = self._smooth_and_format_hand(assigned_left, "Left", shoulder_mid)
            out_msg.left_hand = left_hand_msg
            self.left_hand_pub.publish(left_hand_msg)
        else:
            out_msg.left_hand.present = False
            
        if assigned_right:
            right_hand_msg = self._smooth_and_format_hand(assigned_right, "Right", shoulder_mid)
            out_msg.right_hand = right_hand_msg
            self.right_hand_pub.publish(right_hand_msg)
        else:
            out_msg.right_hand.present = False

        self.publisher.publish(out_msg)

def main():
    rclpy.init()
    node = LandmarkProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
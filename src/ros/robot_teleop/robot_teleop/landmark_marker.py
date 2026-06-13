#!/usr/bin/env python3

import rclpy
import math
from rclpy.node import Node
from rclpy.duration import Duration
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from landmark_msgs.msg import LandmarkMsg

MARKER_LIFETIME_SEC = 0.1

class LandmarkMarker(Node):
    def __init__(self):
        super().__init__('landmark_marker_node')

        self.subscription = self.create_subscription(
            LandmarkMsg, '/processed_landmarks', self.callback, 10)

        self.marker_pub = self.create_publisher(MarkerArray, '/landmark_markers', 10)

        self.hand_connections = [
            (0, 1),  (1, 2),  (2, 3),  (3, 4),
            (0, 5),  (5, 6),  (6, 7),  (7, 8),
            (0, 9),  (9, 10), (10, 11),(11, 12),
            (0, 13),(13, 14),(14, 15),(15, 16),
            (0, 17),(17, 18),(18, 19),(19, 20),
            (5, 9), (9, 13),(13, 17),
        ]

        self.body_connections = [
            ("nose",       "shoulder_L"),
            ("nose",       "shoulder_R"),
            ("shoulder_L", "shoulder_R"),
            ("shoulder_L", "elbow_L"),
            ("elbow_L",    "wrist_L"),
            ("shoulder_R", "elbow_R"),
            ("elbow_R",    "wrist_R"),
            ("shoulder_L", "hip_L"),
            ("shoulder_R", "hip_R"),
            ("hip_L",      "hip_R"),
        ]

        self._body_joint_keys = set()
        for j1, j2 in self.body_connections:
            self._body_joint_keys.add(j1)
            self._body_joint_keys.add(j2)

        self._BODY_JOINT_IDS = {
            "nose": 100, "shoulder_L": 101, "shoulder_R": 102,
            "elbow_L": 103, "wrist_L": 104, "elbow_R": 105,
            "wrist_R": 106, "hip_L": 107, "hip_R": 108,
        }

        self._BODY_LINE_IDS = {
            ("nose", "shoulder_L"): 200, ("nose", "shoulder_R"): 201,
            ("shoulder_L", "shoulder_R"): 202, ("shoulder_L", "elbow_L"): 203,
            ("elbow_L", "wrist_L"): 204, ("shoulder_R", "elbow_R"): 205,
            ("elbow_R", "wrist_R"): 206, ("shoulder_L", "hip_L"): 207,
            ("shoulder_R", "hip_R"): 208, ("hip_L", "hip_R"): 209,
        }

        self._HAND_SPHERE_BASE = {"Left": 1000, "Right": 2000}
        self._HAND_LINE_BASE = {"Left": 3000, "Right": 4000}
        self._HAND_TEXT_BASE = {"Left": 5000, "Right": 5001}

        self._prev_hands_visible = {"Left": False, "Right": False}

    @staticmethod
    def _is_valid(val):
        return isinstance(val, (int, float))

    @staticmethod
    def _hand_color(label, gesture):
        palette = {
            "Left": {
                "OPEN":    (0.0, 1.0, 1.0, 1.0),
                "GRAB":    (1.0, 0.0, 1.0, 1.0),
                "POINT":   (0.2, 0.4, 1.0, 1.0),
                "PEACE":   (0.6, 0.2, 1.0, 1.0),
                "UNKNOWN": (0.5, 0.5, 1.0, 1.0),
            },
            "Right": {
                "OPEN":    (0.0, 1.0, 0.4, 1.0),
                "GRAB":    (1.0, 0.2, 0.2, 1.0),
                "POINT":   (1.0, 0.8, 0.0, 1.0),
                "PEACE":   (1.0, 0.5, 0.0, 1.0),
                "UNKNOWN": (1.0, 0.5, 0.5, 1.0),
            },
        }
        hand_palette = palette.get(label, palette["Left"])
        return hand_palette.get(gesture, hand_palette["UNKNOWN"])

    def _base_marker(self, ns, stamp):
        m = Marker()
        m.header.frame_id = "landmark"
        m.header.stamp = stamp
        m.ns = ns
        m.action = Marker.ADD
        m.lifetime = Duration(seconds=MARKER_LIFETIME_SEC).to_msg()
        m.pose.orientation.w = 1.0
        return m

    def _make_sphere(self, x, y, z, color, scale, ns, stamp, marker_id):
        if not (self._is_valid(x) and self._is_valid(y) and self._is_valid(z)):
            return None
        m = self._base_marker(ns, stamp)
        m.id = marker_id
        m.type = Marker.SPHERE_LIST
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.scale.x = m.scale.y = m.scale.z = float(scale)
        m.color.r, m.color.g, m.color.b, m.color.a = color
        return m

    def _make_line(self, pts, color, ns, stamp, marker_id):
        valid = [p for p in pts
                 if len(p) == 3 and all(self._is_valid(v) for v in p)]
        if len(valid) < 2:
            return None
        m = self._base_marker(ns, stamp)
        m.id = marker_id
        m.type = Marker.LINE_LIST
        m.scale.x = 0.01
        m.scale.y = 0.01
        m.scale.z = 0.01
        m.color.r, m.color.g, m.color.b, m.color.a = color
        for p in valid:
            pt = Point()
            pt.x, pt.y, pt.z = p[0], p[1], p[2]
            m.points.append(pt)
        return m

    def _make_text(self, x, y, z, text, color, ns, stamp, marker_id):
        if not (self._is_valid(x) and self._is_valid(y) and self._is_valid(z)):
            return None
        m = self._base_marker(ns, stamp)
        m.id = marker_id
        m.type = Marker.TEXT_VIEW_FACING
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z + 0.15
        m.scale.z = 0.01
        m.color.r, m.color.g, m.color.b, m.color.a = color
        m.text = text
        return m

    @staticmethod
    def _delete_ns(ns):
        m = Marker()
        m.action = Marker.DELETEALL
        m.ns = ns
        m.id = 0
        return m

    def callback(self, msg: LandmarkMsg):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # 1. Handle Hand Visibility Cleanup
        for label, hand_data in (("Left", msg.left_hand), ("Right", msg.right_hand)):
            currently_visible = hand_data.present
            if self._prev_hands_visible[label] and not currently_visible:
                ns_sphere = f"hand_{label.lower()}"
                ns_line = f"hand_{label.lower()}_lines"
                ns_text = f"hand_{label.lower()}_label"
                marker_array.markers.append(self._delete_ns(ns_sphere))
                marker_array.markers.append(self._delete_ns(ns_line))
                marker_array.markers.append(self._delete_ns(ns_text))
            self._prev_hands_visible[label] = currently_visible

        # 2. Draw Hands
        for label, hand_data in (("Left", msg.left_hand), ("Right", msg.right_hand)):
            if not hand_data.present:
                continue

            gesture = hand_data.gesture if hand_data.gesture else "UNKNOWN"
            color = self._hand_color(label, gesture)
            ns_sphere = f"hand_{label.lower()}"
            ns_line = f"hand_{label.lower()}_lines"
            ns_text = f"hand_{label.lower()}_label"
            sphere_base = self._HAND_SPHERE_BASE[label]
            line_base = self._HAND_LINE_BASE[label]
            text_id = self._HAND_TEXT_BASE[label]

            wrist = hand_data.wrist_m

            m = self._make_sphere(wrist.x, wrist.y, wrist.z, color, 0.08, ns_sphere, stamp, sphere_base)
            if m: marker_array.markers.append(m)

            m = self._make_text(wrist.x, wrist.y, wrist.z, label, color, ns_text, stamp, text_id)
            if m: marker_array.markers.append(m)

            joints = hand_data.joints_m
            if len(joints) == 21:
                for i, joint in enumerate(joints):
                    scale = 0.045 if i in (4, 8, 12, 16, 20) else 0.025
                    m = self._make_sphere(joint.x, joint.y, joint.z, color, scale, ns_sphere, stamp, sphere_base + 1 + i)
                    if m: marker_array.markers.append(m)

                for conn_idx, (p1_idx, p2_idx) in enumerate(self.hand_connections):
                    j1, j2 = joints[p1_idx], joints[p2_idx]
                    m = self._make_line([(j1.x, j1.y, j1.z), (j2.x, j2.y, j2.z)], color, ns_line, stamp, line_base + conn_idx)
                    if m: marker_array.markers.append(m)

        # 3. Draw Body
        body_data = {bl.joint_name: [bl.x, bl.y, bl.z] for bl in msg.body_landmarks}
        body_color = (1.0, 1.0, 0.0, 1.0)

        for joint_name in self._body_joint_keys:
            coords = body_data.get(joint_name)
            if isinstance(coords, list) and len(coords) == 3:
                m = self._make_sphere(coords[0], coords[1], coords[2], body_color, 0.04, "body", stamp, self._BODY_JOINT_IDS[joint_name])
                if m: marker_array.markers.append(m)

        for j1_name, j2_name in self.body_connections:
            p1 = body_data.get(j1_name)
            p2 = body_data.get(j2_name)
            if not (isinstance(p1, list) and len(p1) == 3 and isinstance(p2, list) and len(p2) == 3):
                continue
            m = self._make_line([(p1[0], p1[1], p1[2]), (p2[0], p2[1], p2[2])], body_color, "body_lines", stamp, self._BODY_LINE_IDS[(j1_name, j2_name)])
            if m: marker_array.markers.append(m)

        wrists = []
        if msg.left_hand.present:
            wrists.append([msg.left_hand.wrist_m.x, msg.left_hand.wrist_m.y, msg.left_hand.wrist_m.z])
        if msg.right_hand.present:
            wrists.append([msg.right_hand.wrist_m.x, msg.right_hand.wrist_m.y, msg.right_hand.wrist_m.z])

        self.marker_pub.publish(marker_array)


def main():
    rclpy.init()
    node = LandmarkMarker()
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
import numpy as np
import pandas as pd
from collections import deque

LEFT_HAND_OFFSET = 91
RIGHT_HAND_OFFSET = 112

HAND_FINGERS = {
    "thumb": {"tip": 4, "pip": 3, "mcp": 2},
    "index": {"tip": 8, "pip": 6, "mcp": 5},
    "middle": {"tip": 12, "pip": 10, "mcp": 9},
    "ring": {"tip": 16, "pip": 14, "mcp": 13},
    "pinky": {"tip": 20, "pip": 18, "mcp": 17},
}


class GestureDetector:
    def __init__(
        self,
        open_ratio: float = 0.6,
        close_ratio: float = 0.4,
        min_confidence: float = 0.3,
        extended_angle_deg: float = 160.0,
        finger_angle_overrides: dict = None,
        hysteresis_frames: int = 3,
    ):
        self.open_ratio = open_ratio
        self.close_ratio = close_ratio
        self.min_confidence = min_confidence
        self.extended_angle_deg = extended_angle_deg
        self.finger_angle_overrides = finger_angle_overrides or {}
        self.hysteresis_frames = max(1, hysteresis_frames)

        self._history = {
            "left": deque(maxlen=self.hysteresis_frames),
            "right": deque(maxlen=self.hysteresis_frames),
        }
        self._stable = {"left": "unknown", "right": "unknown"}

    @staticmethod
    def _joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Angle at point b formed by segments b->a and b->c, in degrees."""
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 180.0
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def _is_finger_extended(self, hand_pts: np.ndarray, finger: str, tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
        mcp = hand_pts[mcp_idx]
        pip = hand_pts[pip_idx]
        tip = hand_pts[tip_idx]
        angle = self._joint_angle_deg(mcp, pip, tip)
        threshold = self.finger_angle_overrides.get(finger, self.extended_angle_deg)
        return angle >= threshold

    def _smooth(self, hand_label: str, raw: str) -> str:
        hist = self._history[hand_label]
        hist.append(raw)
        if len(hist) == hist.maxlen and all(h == hist[0] for h in hist):
            self._stable[hand_label] = hist[0]
        return self._stable[hand_label]

    def detect_hand_gesture(self, hand_pts: np.ndarray, confidences: np.ndarray, hand_label: str = None) -> str:
        if hand_pts is None or len(hand_pts) < 21:
            raw = "unknown"
        else:
            mean_conf = float(np.mean(confidences))
            if mean_conf < self.min_confidence:
                raw = "unknown"
            else:
                extended_count = sum(
                    1
                    for finger, indices in HAND_FINGERS.items()
                    if self._is_finger_extended(hand_pts, finger, indices["tip"], indices["pip"], indices["mcp"])
                )
                total_fingers = len(HAND_FINGERS)

                if extended_count >= total_fingers * self.open_ratio:
                    raw = "open"
                elif extended_count <= total_fingers * self.close_ratio:
                    raw = "close"
                else:
                    raw = "partial"

        if hand_label is None:
            return raw
        return self._smooth(hand_label, raw)

    def detect_gestures_from_df(self, xyz_df: pd.DataFrame) -> tuple:
        left_gesture = self._stable["left"]
        right_gesture = self._stable["right"]

        if xyz_df is None or getattr(xyz_df, 'empty', True):
            return left_gesture, right_gesture

        if 'keypoint_id' not in xyz_df.columns:
            return left_gesture, right_gesture

        x_col = 'x_coord' if 'x_coord' in xyz_df.columns else 'x'
        y_col = 'y_coord' if 'y_coord' in xyz_df.columns else 'y'
        z_col = 'z_coord' if 'z_coord' in xyz_df.columns else 'z'

        if 'confidence' not in xyz_df.columns:
            xyz_df = xyz_df.copy()
            xyz_df['confidence'] = 1.0

        def get_hand_data(offset: int):
            hand_df = xyz_df[xyz_df['keypoint_id'].between(offset, offset + 20)]
            if len(hand_df) < 21:
                return None, None
            hand_df = hand_df.sort_values('keypoint_id')
            return hand_df[[x_col, y_col, z_col]].values, hand_df['confidence'].values

        left_pts, left_confs = get_hand_data(LEFT_HAND_OFFSET)
        if left_pts is not None:
            left_gesture = self.detect_hand_gesture(left_pts, left_confs, hand_label="left")

        right_pts, right_confs = get_hand_data(RIGHT_HAND_OFFSET)
        if right_pts is not None:
            right_gesture = self.detect_hand_gesture(right_pts, right_confs, hand_label="right")

        return left_gesture, right_gesture

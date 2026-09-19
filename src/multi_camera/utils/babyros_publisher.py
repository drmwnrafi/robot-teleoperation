import time
import babyros
from .gesture_detection import GestureDetector
from utils.skeleton import SKELETON_FORMATS

class BabyROSPublisher:
    def __init__(self, skeleton_format: str = "coco_wholebody_133"):
        self.skeleton_format = skeleton_format
        self.skeleton_config = SKELETON_FORMATS.get(skeleton_format, SKELETON_FORMATS["coco_wholebody_133"])

        self.pub_landmarks = babyros.node.Publisher(topic="landmarks")
        self.pub_gestures = babyros.node.Publisher(topic="hand_gestures")
        self.gesture_detector = GestureDetector(close_ratio=0.4, finger_angle_overrides={"thumb": 150.0}, hysteresis_frames=3)

        print(f"[BabyROS] Publishers initialized: 'landmarks' (format: {skeleton_format}), 'hand_gestures'")

    def publish_frame(self, frame_index: int, xyz_df):
        current_time = time.time()
        sec = int(current_time)
        nanosec = int((current_time % 1) * 1e9)

        header = {
            "stamp": {"sec": sec, "nanosec": nanosec},
            "frame_id": "world"
        }

        left_gesture, right_gesture = self.gesture_detector.detect_gestures_from_df(xyz_df)

        landmarks_msg = {
            "header": header,
            "skeleton_format": self.skeleton_format
        }

        if xyz_df is not None and not getattr(xyz_df, 'empty', True):
            x_col = 'x_coord' if 'x_coord' in xyz_df.columns else 'x'
            y_col = 'y_coord' if 'y_coord' in xyz_df.columns else 'y'
            z_col = 'z_coord' if 'z_coord' in xyz_df.columns else 'z'
            conf_col = 'confidence' if 'confidence' in xyz_df.columns else None

            sorted_df = xyz_df.sort_values('keypoint_id')

            for part_name, part_config in self.skeleton_config["parts"].items():
                part_indices = set(part_config["indices"])
                part_poses = []

                for _, row in sorted_df.iterrows():
                    kp_id = int(row['keypoint_id'])
                    if kp_id in part_indices:
                        pose_data = {
                            "keypoint_id": kp_id,
                            "position": {
                                "x": float(row[x_col]),
                                "y": float(row[y_col]),
                                "z": float(row[z_col])
                            },
                            "orientation": {
                                "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0
                            }
                        }
                        if conf_col:
                            pose_data["confidence"] = float(row[conf_col])

                        part_poses.append(pose_data)

                landmarks_msg[part_name] = part_poses
        else:
            for part_name in self.skeleton_config["parts"].keys():
                landmarks_msg[part_name] = []

        self.pub_landmarks.publish(landmarks_msg)

        self.pub_gestures.publish({
            "header": header,
            "left_hand": left_gesture,
            "right_hand": right_gesture
        })

    def cleanup(self):
        print("[BabyROS] Publishers cleaned up.")

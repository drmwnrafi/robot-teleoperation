import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import babyros
import base64 
import time
import threading
import pyvista as pv
import yaml

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

latest_image = None
image_lock = threading.Lock()

def image_callback(msg: dict):
    global latest_image
    try:
        img_bytes = base64.b64decode(msg['data'])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        with image_lock:
            latest_image = img
    except Exception as e:
        print(f"Error decoding image: {e}")

class GestureRecognizer:
    THUMB_TIP = 4; THUMB_IP = 3; THUMB_MCP = 2; THUMB_CMC = 1
    INDEX_TIP = 8; INDEX_PIP = 6; INDEX_MCP = 5
    MIDDLE_TIP = 12; MIDDLE_PIP = 10; MIDDLE_MCP = 9
    RING_TIP = 16; RING_PIP = 14; RING_MCP = 13
    PINKY_TIP = 20; PINKY_PIP = 18; PINKY_MCP = 17
    WRIST = 0
    
    def __init__(self, open_ratio=0.6):
        self.open_ratio = open_ratio
    
    def is_finger_extended(self, landmarks, tip_idx, pip_idx, mcp_idx=None):
        tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y, landmarks[tip_idx].z])
        pip = np.array([landmarks[pip_idx].x, landmarks[pip_idx].y, landmarks[pip_idx].z])
        wrist = np.array([landmarks[self.WRIST].x, landmarks[self.WRIST].y, landmarks[self.WRIST].z])
        if mcp_idx is not None:
            mcp = np.array([landmarks[mcp_idx].x, landmarks[mcp_idx].y, landmarks[mcp_idx].z])
            tip_to_pip = np.linalg.norm(tip - pip)
            mcp_to_pip = np.linalg.norm(mcp - pip)
            return tip_to_pip > mcp_to_pip * 1.2
        else:
            tip_to_wrist = np.linalg.norm(tip - wrist)
            pip_to_wrist = np.linalg.norm(pip - wrist)
            return tip_to_wrist > pip_to_wrist + 0.02
    
    def recognize_gesture(self, landmarks):
        if len(landmarks) < 21:
            return "unknown"
        fingers_extended = [
            self.is_finger_extended(landmarks, self.THUMB_TIP, self.THUMB_IP, self.INDEX_MCP),
            self.is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP),
            self.is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP),
            self.is_finger_extended(landmarks, self.RING_TIP, self.RING_PIP),
            self.is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_PIP)
        ]
        extended_count = sum(fingers_extended)
        total_fingers = len(fingers_extended)
        
        if extended_count >= total_fingers * self.open_ratio:
            return "open"
        elif extended_count <= total_fingers * (1 - self.open_ratio):
            return "close"
        return "partial"

class BodyPreFocus:    
    def __init__(self, hand_padding_factor=1.5, min_hand_size=100):
        self.hand_padding_factor = hand_padding_factor
        self.min_hand_size = min_hand_size
        
    def get_wrist_positions(self, pose_landmarks, frame_width, frame_height):
        if pose_landmarks is None or len(pose_landmarks.landmark) < 16:
            return None, None
        left_wrist = pose_landmarks.landmark[15]
        right_wrist = pose_landmarks.landmark[16]
        left_elbow = pose_landmarks.landmark[13]
        right_elbow = pose_landmarks.landmark[14]
        
        left_wrist_px = (int(left_wrist.x * frame_width), int(left_wrist.y * frame_height))
        right_wrist_px = (int(right_wrist.x * frame_width), int(right_wrist.y * frame_height))
        left_elbow_px = (int(left_elbow.x * frame_width), int(left_elbow.y * frame_height))
        right_elbow_px = (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height))
        return (left_wrist_px, left_elbow_px), (right_wrist_px, right_elbow_px)
    
    def create_hand_crop(self, wrist_px, elbow_px, frame_width, frame_height):
        wrist_x, wrist_y = wrist_px
        elbow_x, elbow_y = elbow_px
        arm_length = np.sqrt((wrist_x - elbow_x)**2 + (wrist_y - elbow_y)**2)
        hand_size = max(int(arm_length * self.hand_padding_factor), self.min_hand_size)
        
        x_min = max(0, wrist_x - hand_size)
        y_min = max(0, wrist_y - hand_size)
        x_max = min(frame_width, wrist_x + hand_size)
        y_max = min(frame_height, wrist_y + hand_size)
        
        if x_max - x_min < self.min_hand_size:
            center_x = (x_min + x_max) // 2
            x_min = max(0, center_x - self.min_hand_size // 2)
            x_max = min(frame_width, center_x + self.min_hand_size // 2)
        if y_max - y_min < self.min_hand_size:
            center_y = (y_min + y_max) // 2
            y_min = max(0, center_y - self.min_hand_size // 2)
            y_max = min(frame_height, center_y + self.min_hand_size // 2)
        return (x_min, y_min, x_max, y_max)
    
    def shift_landmarks_to_original(self, landmarks, crop_box, frame_width, frame_height):
        x_min, y_min, _, _ = crop_box
        for landmark in landmarks.landmark:
            landmark.x = (landmark.x * (crop_box[2] - crop_box[0]) + x_min) / frame_width
            landmark.y = (landmark.y * (crop_box[3] - crop_box[1]) + y_min) / frame_height

class EMAFilter:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.prev_points = None
    def filter(self, points):
        if self.prev_points is None:
            self.prev_points = points.copy()
            return points
        filtered = self.alpha * points + (1 - self.alpha) * self.prev_points
        self.prev_points = filtered
        return filtered

class RigidTransformFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.prev_rvec = None
        self.prev_tvec = None
    def filter(self, rvec, tvec):
        if self.prev_rvec is None:
            self.prev_rvec = rvec.copy()
            self.prev_tvec = tvec.copy()
            return rvec, tvec
        filtered_rvec = self.alpha * rvec + (1 - self.alpha) * self.prev_rvec
        filtered_tvec = self.alpha * tvec + (1 - self.alpha) * self.prev_tvec
        self.prev_rvec = filtered_rvec
        self.prev_tvec = filtered_tvec
        return filtered_rvec, filtered_tvec

def get_camera_matrix(frame_width, frame_height):
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    distortion = np.zeros((4, 1))
    return camera_matrix, distortion

def load_calibration(calib_file, frame_width, frame_height):
    if calib_file and os.path.exists(calib_file):
        try:
            calib_data = np.load(calib_file)
            return calib_data["camera_matrix"], calib_data["dist_coeffs"]
        except Exception as e:
            print(f"Error loading calibration: {e}. Falling back.")
    return get_camera_matrix(frame_width, frame_height)

class SyntheticResults:
    def __init__(self, landmarks_list, world_landmarks_list, sides_list):
        self.multi_hand_landmarks = landmarks_list
        self.multi_hand_world_landmarks = world_landmarks_list
        self.multi_handedness = None
        self.hand_sides = sides_list

def apply_anthropometric_scaling(pose_world_points, args):
    if pose_world_points is None or len(pose_world_points) < 33:
        return pose_world_points

    scaled_points = pose_world_points.copy()
    
    def get_dist(idx1, idx2):
        return np.linalg.norm(scaled_points[idx1] - scaled_points[idx2])

    if args.pelvis_width is not None and args.pelvis_width > 0:
        target_pelvis_m = args.pelvis_width / 100.0
        current_pelvis = get_dist(23, 24)
        if current_pelvis > 1e-5:
            scale = target_pelvis_m / current_pelvis
            center = (scaled_points[23] + scaled_points[24]) / 2.0
            new_l_hip = center + (scaled_points[23] - center) * scale
            new_r_hip = center + (scaled_points[24] - center) * scale
            delta_l = new_l_hip - scaled_points[23]
            delta_r = new_r_hip - scaled_points[24]
            scaled_points[23] = new_l_hip
            scaled_points[24] = new_r_hip
            for idx in [25, 27, 29, 31]: scaled_points[idx] += delta_l
            for idx in [26, 28, 30, 32]: scaled_points[idx] += delta_r

    if args.clavicle_length is not None and args.clavicle_length > 0:
        target_len_m = args.clavicle_length / 100.0
        l_shoulder = scaled_points[11]
        r_shoulder = scaled_points[12]
        current_vec = r_shoulder - l_shoulder
        current_len = np.linalg.norm(current_vec)
        if current_len > 1e-5:
            scale = target_len_m / current_len
            center = (l_shoulder + r_shoulder) / 2.0
            new_l = center + (l_shoulder - center) * scale
            new_r = center + (r_shoulder - center) * scale
            delta_l = new_l - l_shoulder
            delta_r = new_r - r_shoulder
            scaled_points[11] = new_l
            scaled_points[12] = new_r
            for idx in [13, 15, 17, 19, 21]: scaled_points[idx] += delta_l
            for idx in [14, 16, 18, 20, 22]: scaled_points[idx] += delta_r

    if args.torso_height is not None and args.torso_height > 0:
        target_torso_m = args.torso_height / 100.0
        delta_l = np.zeros(3)
        delta_r = np.zeros(3)
        current_left = np.linalg.norm(scaled_points[11] - scaled_points[23])
        if current_left > 1e-5:
            scale_l = target_torso_m / current_left
            new_l_shoulder = scaled_points[23] + (scaled_points[11] - scaled_points[23]) * scale_l
            delta_l = new_l_shoulder - scaled_points[11]
            scaled_points[11] = new_l_shoulder
            for idx in [13, 15, 17, 19, 21]: scaled_points[idx] += delta_l
        current_right = np.linalg.norm(scaled_points[12] - scaled_points[24])
        if current_right > 1e-5:
            scale_r = target_torso_m / current_right
            new_r_shoulder = scaled_points[24] + (scaled_points[12] - scaled_points[24]) * scale_r
            delta_r = new_r_shoulder - scaled_points[12]
            scaled_points[12] = new_r_shoulder
            for idx in [14, 16, 18, 20, 22]: scaled_points[idx] += delta_r
        avg_delta = (delta_l + delta_r) / 2.0
        for idx in range(0, 11): scaled_points[idx] += avg_delta

    def scale_bone_and_shift(parent_idx, child_idx, child_descendants, target_length_cm):
        if target_length_cm is None or target_length_cm <= 0: return
        target_len_m = target_length_cm / 100.0
        parent = scaled_points[parent_idx]
        child = scaled_points[child_idx]
        current_vec = child - parent
        current_len = np.linalg.norm(current_vec)
        if current_len > 1e-5:
            scale = target_len_m / current_len
            new_child = parent + current_vec * scale
            delta = new_child - child
            scaled_points[child_idx] = new_child
            for desc_idx in child_descendants: scaled_points[desc_idx] += delta

    scale_bone_and_shift(11, 13, [15, 17, 19, 21], args.humerus_length)
    scale_bone_and_shift(13, 15, [17, 19, 21], args.forearm_length)
    scale_bone_and_shift(12, 14, [16, 18, 20, 22], args.humerus_length)
    scale_bone_and_shift(14, 16, [18, 20, 22], args.forearm_length)

    return scaled_points

def load_config(config_path: str) -> argparse.Namespace:
    default_config = {
        "image_topic": "camera_image",
        "calibration": None,
        "width": 1920,
        "height": 1080,
        "window_width": 960,
        "window_height": 540,
        "max_fps": 20, 
        "smoothing": 0.10,
        "detect_conf_hand": 0.5,
        "track_conf_hand": 0.5,
        "detect_conf_pose": 0.5,
        "track_conf_pose": 0.5,
        "hand_padding": 1.0,
        "use_bpf": True,
        "show_3d": True,
        "enhance": False,
        "pelvis_width": None,
        "clavicle_length": None,
        "humerus_length": None,
        "forearm_length": None,
        "torso_height": None,
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    default_config.update(yaml_config)
            print(f"Successfully loaded configuration from: {config_path}")
        except Exception as e:
            print(f"Error loading config file '{config_path}': {e}. Using defaults.")
    else:
        print(f"Config file '{config_path}' not found. Using default values.")
    
    return argparse.Namespace(**default_config)

def parse_args():
    parser = argparse.ArgumentParser(description="3D Hand & Holistic Pose Tracking Subscriber with Gesture Recognition")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML configuration file")
    return parser.parse_args()

def main():
    cli_args = parse_args()
    args = load_config(cli_args.config)
    
    print(f"Subscribing to image topic: '{args.image_topic}'...")
    sub_image = babyros.node.Subscriber(topic=args.image_topic, callback=image_callback)
    
    pub_tracking = babyros.node.Publisher(topic="landmarks")
    pub_gesture = babyros.node.Publisher(topic="hand_gestures")
    pub_mp_image = babyros.node.Publisher(topic="mediapipe_image")
    
    print("BabyROS Publishers initialized:")
    print("  - 'landmarks'")
    print("  - 'hand_gestures'")
    print("  - 'mediapipe_image'")
    
    if args.use_bpf:
        print(f"Body Pre-Focusing ENABLED (hand_padding={args.hand_padding})")
    else:
        print("Body Pre-Focusing disabled")

    target_fps = getattr(args, 'max_fps', 30)
    frame_time = 1.0 / target_fps if target_fps > 0 else 0.0
    print(f"Target FPS limited to: {target_fps}")

    bpf = BodyPreFocus(hand_padding_factor=args.hand_padding) if args.use_bpf else None
    gesture_recognizer = GestureRecognizer()

    plotter = None
    left_hand_mesh = right_hand_mesh = None
    pose_body_mesh = pose_l_arm_mesh = pose_r_arm_mesh = pose_l_leg_mesh = pose_r_leg_mesh = None
    dummy_hand_points = np.zeros((21, 3))
    dummy_pose_points = np.zeros((33, 3))

    if args.show_3d and pv is not None:
        plotter = pv.Plotter(window_size=(1200, 800), title="3D Hand & Holistic Pose Tracking")
        plotter.set_background('#1e1e1e')

        hand_line_indices = []
        for conn in mp_hands.HAND_CONNECTIONS:
            hand_line_indices.extend([2, conn[0], conn[1]])

        left_arm_ids = {11, 13, 15, 17, 19, 21}
        right_arm_ids = {12, 14, 16, 18, 20, 22}
        left_leg_ids = {23, 25, 27, 29, 31}
        right_leg_ids = {24, 26, 28, 30, 32}
        
        body_lines, l_arm_lines, r_arm_lines, l_leg_lines, r_leg_lines = [], [], [], [], []
        for conn in mp_holistic.POSE_CONNECTIONS:
            c0, c1 = conn[0], conn[1]
            if c0 in left_arm_ids and c1 in left_arm_ids:
                l_arm_lines.extend([2, c0, c1])
            elif c0 in right_arm_ids and c1 in right_arm_ids:
                r_arm_lines.extend([2, c0, c1])
            elif c0 in left_leg_ids and c1 in left_leg_ids:
                l_leg_lines.extend([2, c0, c1])
            elif c0 in right_leg_ids and c1 in right_leg_ids:
                r_leg_lines.extend([2, c0, c1])
            else:
                body_lines.extend([2, c0, c1])

        left_hand_mesh = pv.PolyData(dummy_hand_points.copy(), lines=hand_line_indices)
        plotter.add_mesh(left_hand_mesh, color='#32CD32', line_width=4, point_size=10, render_points_as_spheres=True, name='left_hand')

        right_hand_mesh = pv.PolyData(dummy_hand_points.copy(), lines=hand_line_indices)
        plotter.add_mesh(right_hand_mesh, color='#FF0000', line_width=4, point_size=10, render_points_as_spheres=True, name='right_hand')

        pose_body_mesh = pv.PolyData(dummy_pose_points.copy(), lines=body_lines)
        plotter.add_mesh(pose_body_mesh, color='#FFFFFF', line_width=3, point_size=8, render_points_as_spheres=True, name='pose_body')

        pose_l_arm_mesh = pv.PolyData(dummy_pose_points.copy(), lines=l_arm_lines)
        plotter.add_mesh(pose_l_arm_mesh, color='#00FFFF', line_width=3, point_size=8, render_points_as_spheres=True, name='pose_l_arm')

        pose_r_arm_mesh = pv.PolyData(dummy_pose_points.copy(), lines=r_arm_lines)
        plotter.add_mesh(pose_r_arm_mesh, color='#FF00FF', line_width=3, point_size=8, render_points_as_spheres=True, name='pose_r_arm')

        pose_l_leg_mesh = pv.PolyData(dummy_pose_points.copy(), lines=l_leg_lines)
        plotter.add_mesh(pose_l_leg_mesh, color='#FFFF00', line_width=3, point_size=8, render_points_as_spheres=True, name='pose_l_leg')

        pose_r_leg_mesh = pv.PolyData(dummy_pose_points.copy(), lines=r_leg_lines)
        plotter.add_mesh(pose_r_leg_mesh, color='#FF8C00', line_width=3, point_size=8, render_points_as_spheres=True, name='pose_r_leg')

        plotter.show(auto_close=False, interactive_update=True)
        _cam_target = np.array(plotter.camera.focal_point)
        _cam_target_previous = _cam_target.copy()
        _cam_smooth_alpha = 0.05

    frame_width = args.width
    frame_height = args.height
    camera_matrix = None
    distortion = None

    hand_rigid_filter_left = RigidTransformFilter(alpha=args.smoothing)
    hand_rigid_filter_right = RigidTransformFilter(alpha=args.smoothing)
    pose_filter = EMAFilter(alpha=args.smoothing)
    pose_rigid_filter = RigidTransformFilter(alpha=args.smoothing)

    if args.enhance:
        print("Software Image Enhancement ENABLED (CLAHE + Unsharp Masking)")
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    else:
        clahe = None

    window_name = '2D External Camera (Press q to quit, b to toggle BPF)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)

    with mp_hands.Hands(
            model_complexity=1, max_num_hands=2,
            min_detection_confidence=args.detect_conf_hand,
            min_tracking_confidence=args.track_conf_hand) as hands, \
         mp_holistic.Holistic(
            model_complexity=2, smooth_landmarks=True, enable_segmentation=False,
            min_detection_confidence=args.detect_conf_pose,
            min_tracking_confidence=args.track_conf_pose) as holistic:

        print("Starting Tracking & Publishing with Gesture Recognition...")
        use_bpf = args.use_bpf
        frame_count = 0

        try:
            while True:
                loop_start_time = time.time()  
                
                with image_lock:
                    image = latest_image
                
                if image is None:
                    print("Waiting for first image from camera publisher...", end='\r')
                    time.sleep(0.1)
                    continue
                
                if camera_matrix is None:
                    frame_height, frame_width = image.shape[:2]
                    print(f"Received first image at {frame_width}x{frame_height}")
                    camera_matrix, distortion = load_calibration(args.calibration, frame_width, frame_height)

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if clahe is not None:
                    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    l = clahe.apply(l)
                    lab = cv2.merge((l, a, b))
                    image_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    blurred = cv2.GaussianBlur(image_rgb, (0, 0), 2.0)
                    image_rgb = cv2.addWeighted(image_rgb, 1.5, blurred, -0.5, 0)
                
                frame_h, frame_w = image.shape[:2]

                results_holistic = holistic.process(image_rgb)
                left_wrist_elbow, right_wrist_elbow = None, None
                left_crop, right_crop = None, None
                
                if results_holistic.pose_landmarks and bpf is not None and use_bpf:
                    left_wrist_elbow, right_wrist_elbow = bpf.get_wrist_positions(results_holistic.pose_landmarks, frame_w, frame_h)
                    if left_wrist_elbow:
                        left_crop = bpf.create_hand_crop(left_wrist_elbow[0], left_wrist_elbow[1], frame_w, frame_h)
                    if right_wrist_elbow:
                        right_crop = bpf.create_hand_crop(right_wrist_elbow[0], right_wrist_elbow[1], frame_w, frame_h)

                results_hands = None
                hands_detected_in_crop = False
                
                if use_bpf and bpf is not None and (left_crop or right_crop):
                    if left_crop and right_crop:
                        x_min = min(left_crop[0], right_crop[0])
                        y_min = min(left_crop[1], right_crop[1])
                        x_max = max(left_crop[2], right_crop[2])
                        y_max = max(left_crop[3], right_crop[3])
                        combined_crop = (x_min, y_min, x_max, y_max)
                        if x_max > x_min and y_max > y_min:
                            cropped_image = image_rgb[y_min:y_max, x_min:x_max]
                            if cropped_image.size > 0:
                                results_hands = hands.process(cropped_image)
                                if results_hands.multi_hand_landmarks:
                                    hands_detected_in_crop = True
                                    for hand_landmarks in results_hands.multi_hand_landmarks:
                                        bpf.shift_landmarks_to_original(hand_landmarks, combined_crop, frame_w, frame_h)
                    else:
                        crops_to_process = []
                        if left_crop: crops_to_process.append(('left', left_crop))
                        if right_crop: crops_to_process.append(('right', right_crop))
                        
                        all_hand_landmarks = []
                        all_hand_world_landmarks = []
                        all_hand_sides = []
                        
                        for side, crop in crops_to_process:
                            x_min, y_min, x_max, y_max = crop
                            if x_max <= x_min or y_max <= y_min: continue
                            cropped_image = image_rgb[y_min:y_max, x_min:x_max]
                            if cropped_image.size == 0: continue
                                
                            crop_results = hands.process(cropped_image)
                            if crop_results.multi_hand_landmarks:
                                hands_detected_in_crop = True
                                for idx, hand_landmarks in enumerate(crop_results.multi_hand_landmarks):
                                    bpf.shift_landmarks_to_original(hand_landmarks, crop, frame_w, frame_h)
                                    all_hand_landmarks.append(hand_landmarks)
                                    if crop_results.multi_hand_world_landmarks and idx < len(crop_results.multi_hand_world_landmarks):
                                        all_hand_world_landmarks.append(crop_results.multi_hand_world_landmarks[idx])
                                    else:
                                        all_hand_world_landmarks.append(None)
                                    all_hand_sides.append(side)
                        
                        if all_hand_landmarks:
                            results_hands = SyntheticResults(all_hand_landmarks, all_hand_world_landmarks, all_hand_sides)

                    if not hands_detected_in_crop:
                        results_hands = hands.process(image_rgb)
                else:
                    results_hands = hands.process(image_rgb)

                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                if results_holistic.pose_landmarks:
                    mp_drawing.draw_landmarks(image_bgr, results_holistic.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                              mp_drawing_styles.get_default_pose_landmarks_style())
                
                if results_hands and getattr(results_hands, 'multi_hand_landmarks', None):
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                                  mp_drawing_styles.get_default_hand_connections_style())
                    if use_bpf and bpf is not None:
                        if left_crop:
                            cv2.rectangle(image_bgr, (left_crop[0], left_crop[1]), (left_crop[2], left_crop[3]), (0, 255, 0), 2)
                            cv2.putText(image_bgr, "Left Hand ROI", (left_crop[0], left_crop[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        if right_crop:
                            cv2.rectangle(image_bgr, (right_crop[0], right_crop[1]), (right_crop[2], right_crop[3]), (255, 0, 0), 2)
                            cv2.putText(image_bgr, "Right Hand ROI", (right_crop[0], right_crop[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                pose_world_points = None
                if results_holistic.pose_world_landmarks and results_holistic.pose_landmarks:
                    model_points = np.array([[lm.x, lm.y, lm.z] for lm in results_holistic.pose_world_landmarks.landmark])
                    image_points = np.array([[lm.x * frame_width, lm.y * frame_height] for lm in results_holistic.pose_landmarks.landmark])
                    success_pnp, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)
                    if success_pnp:
                        rvec, tvec = pose_rigid_filter.filter(rvec, tvec)
                        rmat, _ = cv2.Rodrigues(rvec)
                        transformation = np.eye(4)
                        transformation[0:3, 0:3] = rmat
                        transformation[0:3, 3] = tvec.squeeze()
                        model_points_hom = np.concatenate((model_points, np.ones((33, 1))), axis=1)
                        pose_world_points = model_points_hom.dot(transformation.T)[:, :3]
                        pose_world_points[:, 1] = -pose_world_points[:, 1]
                        pose_world_points[:, 2] = -pose_world_points[:, 2]
                        pose_world_points = pose_filter.filter(pose_world_points)
                        pose_world_points = apply_anthropometric_scaling(pose_world_points, args)

                left_hand_final_points = None
                right_hand_final_points = None
                left_gesture = "unknown"
                right_gesture = "unknown"

                if results_hands and getattr(results_hands, 'multi_hand_landmarks', None):
                    for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                        gesture = gesture_recognizer.recognize_gesture(hand_landmarks.landmark)
                        if (getattr(results_hands, 'multi_hand_world_landmarks', None) and 
                            idx < len(results_hands.multi_hand_world_landmarks) and 
                            results_hands.multi_hand_world_landmarks[idx] is not None):
                            world_landmarks = results_hands.multi_hand_world_landmarks[idx]
                            hand_model_points = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks.landmark])
                        else:
                            continue 
                        
                        hand_image_points = np.array([[lm.x * frame_width, lm.y * frame_height] for lm in hand_landmarks.landmark])

                        if hasattr(results_hands, 'hand_sides') and idx < len(results_hands.hand_sides):
                            is_left_hand = (results_hands.hand_sides[idx] == 'left')
                        elif results_holistic.pose_landmarks:
                            hand_wrist_2d = hand_image_points[0]
                            pose_left_wrist_2d = np.array([results_holistic.pose_landmarks.landmark[15].x * frame_width, results_holistic.pose_landmarks.landmark[15].y * frame_height])
                            pose_right_wrist_2d = np.array([results_holistic.pose_landmarks.landmark[16].x * frame_width, results_holistic.pose_landmarks.landmark[16].y * frame_height])
                            
                            dist_left = np.linalg.norm(hand_wrist_2d - pose_left_wrist_2d)
                            dist_right = np.linalg.norm(hand_wrist_2d - pose_right_wrist_2d)
                            is_left_hand = (dist_left < dist_right)
                        else:
                            wrist_x = hand_image_points[0, 0]
                            is_left_hand = (wrist_x > frame_width / 2)
                        
                        if is_left_hand: left_gesture = gesture
                        else: right_gesture = gesture
                        
                        if np.all(hand_model_points == 0) or np.isnan(hand_model_points).any():
                            print(f"Hand {idx}: Invalid 3D landmarks, skipping.")
                            continue

                        success_pnp, rvec, tvec = cv2.solvePnP(hand_model_points, hand_image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)

                        if not success_pnp:
                            hand_model_points_flipped = hand_model_points.copy()
                            hand_model_points_flipped[:, 0] = -hand_model_points_flipped[:, 0]
                            success_pnp, rvec, tvec = cv2.solvePnP(hand_model_points_flipped, hand_image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)
                            if success_pnp:
                                hand_model_points = hand_model_points_flipped
                                
                        if not success_pnp:
                            success_pnp, rvec, tvec = cv2.solvePnP(hand_model_points, hand_image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
                            if not success_pnp:
                                hand_model_points_flipped = hand_model_points.copy()
                                hand_model_points_flipped[:, 0] = -hand_model_points_flipped[:, 0]
                                success_pnp, rvec, tvec = cv2.solvePnP(hand_model_points_flipped, hand_image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
                                if success_pnp:
                                    hand_model_points = hand_model_points_flipped
                     
                        if success_pnp:
                            if is_left_hand:
                                rvec, tvec = hand_rigid_filter_left.filter(rvec, tvec)
                            else:
                                rvec, tvec = hand_rigid_filter_right.filter(rvec, tvec)

                            rmat, _ = cv2.Rodrigues(rvec)
                            transformation = np.eye(4)
                            transformation[0:3, 0:3] = rmat
                            transformation[0:3, 3] = tvec.squeeze()
                            
                            hand_model_hom = np.concatenate((hand_model_points, np.ones((21, 1))), axis=1)
                            hand_world_points = hand_model_hom.dot(transformation.T)[:, :3]
                            hand_world_points[:, 1] = -hand_world_points[:, 1]
                            hand_world_points[:, 2] = -hand_world_points[:, 2]
                            
                            if pose_world_points is not None:
                                if is_left_hand and len(pose_world_points) > 15:
                                    hand_world_points = pose_world_points[15] + (hand_world_points - hand_world_points[0])
                                elif not is_left_hand and len(pose_world_points) > 16:
                                    hand_world_points = pose_world_points[16] + (hand_world_points - hand_world_points[0])
                            
                            if is_left_hand:
                                left_hand_final_points = hand_world_points.copy()
                            else:
                                right_hand_final_points = hand_world_points.copy()

                poses_list = []
                if pose_world_points is not None and len(pose_world_points) == 33:
                    for idx in range(33):
                        pt = pose_world_points[idx]
                        poses_list.append({"position": {"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])}, "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}})
                if left_hand_final_points is not None and len(left_hand_final_points) == 21:
                    for pt in left_hand_final_points:
                        poses_list.append({"position": {"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])}, "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}})
                if right_hand_final_points is not None and len(right_hand_final_points) == 21:
                    for pt in right_hand_final_points:
                        poses_list.append({"position": {"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])}, "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}})

                if len(poses_list) > 0:
                    pub_tracking.publish({"header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "world"}, "poses": poses_list})

                pub_gesture.publish({"header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "world"}, "left_hand": left_gesture, "right_hand": right_gesture})

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                success_encode, encoded_image = cv2.imencode('.jpg', image_bgr, encode_param)
                if success_encode:
                    img_bytes = encoded_image.tobytes()
                    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    mp_img_msg = {
                        "header": {"stamp": {"sec": int(time.time()), "nanosec": 0}, "frame_id": "camera"},
                        "format": "jpeg",
                        "data": img_b64 
                    }
                    pub_mp_image.publish(mp_img_msg)

                if args.show_3d and plotter is not None:
                    if left_hand_final_points is not None:
                        left_hand_mesh.points = left_hand_final_points
                    else:
                        left_hand_mesh.points = dummy_hand_points
                        
                    if right_hand_final_points is not None:
                        right_hand_mesh.points = right_hand_final_points
                    else:
                        right_hand_mesh.points = dummy_hand_points

                    if pose_world_points is not None:
                        pose_body_mesh.points = pose_world_points
                        pose_l_arm_mesh.points = pose_world_points
                        pose_r_arm_mesh.points = pose_world_points
                        pose_l_leg_mesh.points = pose_world_points
                        pose_r_leg_mesh.points = pose_world_points
                        
                        torso_center = (pose_world_points[11] + pose_world_points[12] + 
                                        pose_world_points[23] + pose_world_points[24]) / 4.0
                        
                        smooth_target = _cam_smooth_alpha * torso_center + (1 - _cam_smooth_alpha) * _cam_target
                        delta = smooth_target - _cam_target_previous
                        
                        plotter.camera.focal_point += delta
                        plotter.camera.position += delta
                        
                        _cam_target = smooth_target
                        _cam_target_previous = smooth_target
                        
                    else:
                        pose_body_mesh.points = dummy_pose_points
                        pose_l_arm_mesh.points = dummy_pose_points
                        pose_r_arm_mesh.points = dummy_pose_points
                        pose_l_leg_mesh.points = dummy_pose_points
                        pose_r_leg_mesh.points = dummy_pose_points
                        
                    plotter.update()

                image_bgr_mirror = cv2.flip(image_bgr, 1)
                cv2.putText(image_bgr_mirror, f"BPF: {'ON' if use_bpf else 'OFF'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if use_bpf else (0, 0, 255), 2)
                cv2.putText(image_bgr_mirror, f"Enhance: {'ON' if clahe is not None else 'OFF'}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if clahe is not None else (0, 0, 255), 2)
                cv2.putText(image_bgr_mirror, f"Left: {left_gesture}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(image_bgr_mirror, f"Right: {right_gesture}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                cv2.imshow(window_name, image_bgr_mirror)
                
                # Calculate dynamic delay to maintain target FPS
                elapsed_time = time.time() - loop_start_time
                if frame_time > 0:
                    wait_ms = max(1, int((frame_time - elapsed_time) * 1000))
                else:
                    wait_ms = 1
                    
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('b'):
                    use_bpf = not use_bpf
                    print(f"Body Pre-Focusing {'enabled' if use_bpf else 'disabled'}")

                frame_count += 1

        except KeyboardInterrupt:
            print("Tracking interrupted by user.")
        finally:
            sub_image.delete()
            cv2.destroyAllWindows()
            if args.show_3d and plotter is not None:
                plotter.close()
            print("Subscriber cleanup complete.")

if __name__ == "__main__":
    main()
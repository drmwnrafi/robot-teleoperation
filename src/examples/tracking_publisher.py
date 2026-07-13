import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import babyros
import platform
import base64  # <-- ADDED: Required for Base64 encoding of image bytes

try:
    import pyvista as pv
except ImportError:
    pv = None

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


class GestureRecognizer:
    """Recognize hand gestures (open/close) from MediaPipe landmarks."""
    
    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_MCP = 2
    THUMB_CMC = 1
    INDEX_TIP = 8
    INDEX_PIP = 6
    INDEX_MCP = 5
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    MIDDLE_MCP = 9
    RING_TIP = 16
    RING_PIP = 14
    RING_MCP = 13
    PINKY_TIP = 20
    PINKY_PIP = 18
    PINKY_MCP = 17
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
        
        fingers_extended = []
        
        # Thumb
        thumb_extended = self.is_finger_extended(landmarks, self.THUMB_TIP, self.THUMB_IP, self.INDEX_MCP)
        fingers_extended.append(thumb_extended)
        
        # Index
        fingers_extended.append(self.is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP))
        
        # Middle
        fingers_extended.append(self.is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP))
        
        # Ring
        fingers_extended.append(self.is_finger_extended(landmarks, self.RING_TIP, self.RING_PIP))
        
        # Pinky
        fingers_extended.append(self.is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_PIP))
        
        extended_count = sum(fingers_extended)
        total_fingers = len(fingers_extended)
        
        if extended_count >= total_fingers * self.open_ratio:
            return "open"
        elif extended_count <= total_fingers * (1 - self.open_ratio):
            return "close"
        else:
            return "partial"


class BodyPreFocus:    
    def __init__(self, hand_padding_factor=1.5, min_hand_size=100):
        self.hand_padding_factor = hand_padding_factor
        self.min_hand_size = min_hand_size
        self.last_hand_rois = {'left': None, 'right': None}
        
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
        hand_size = int(arm_length * self.hand_padding_factor)
        hand_size = max(hand_size, self.min_hand_size)
        
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
            print(f" Error loading calibration: {e}. Falling back.")
    return get_camera_matrix(frame_width, frame_height)


def parse_args():
    parser = argparse.ArgumentParser(description="3D Hand & Pose Tracking Publisher with Gesture Recognition")
    parser.add_argument("--calibration", "-c", type=str, default="cam_calib_(10x7)_22.0mm.npz")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    
    # --- NEW: Display Window Size Arguments ---
    parser.add_argument("--window-width", type=int, default=960, help="Display window width (default: 960)")
    parser.add_argument("--window-height", type=int, default=540, help="Display window height (default: 540)")
    # ------------------------------------------

    parser.add_argument("--smoothing", type=float, default=0.10)
    parser.add_argument("--detect_conf_hand", type=float, default=0.3)
    parser.add_argument("--track_conf_hand", type=float, default=0.3)
    parser.add_argument("--detect_conf_pose", type=float, default=0.5)
    parser.add_argument("--track_conf_pose", type=float, default=0.5)
    parser.add_argument("--hand-padding", type=float, default=3.0,
                        help="Padding factor for hand crop around wrist (default: 3.0)")
    parser.add_argument("--use-bpf", action="store_true", 
                        help="Enable Body Pre-Focusing for improved far-range detection")
    parser.add_argument("--show-3d", action="store_true", help="Enable PyVista 3D visualization")
    
    # --- Camera Hardware Controls ---
    parser.add_argument("--brightness", type=float, default=-1.0, help="Camera brightness (0-255, -1 to ignore)")
    parser.add_argument("--contrast", type=float, default=-1.0, help="Camera contrast (0-255, -1 to ignore)")
    parser.add_argument("--sharpness", type=float, default=-1.0, help="Camera sharpness (0-255, -1 to ignore)")
    parser.add_argument("--gain", type=float, default=-1.0, help="Camera gain/ISO (0-255, -1 to ignore)")
    parser.add_argument("--exposure", type=float, default=-1.0, help="Camera exposure (-1 to ignore)")
    parser.add_argument("--camera-settings", action="store_true", 
                        help="Open native camera properties dialog (Works best on Windows)")
    parser.add_argument("--enhance", action="store_true", help="Enable software CLAHE and sharpness enhancement")
    
    return parser.parse_args()

# Define SyntheticResults outside the loop for efficiency
class SyntheticResults:
    def __init__(self, landmarks_list, sides_list):
        self.multi_hand_landmarks = landmarks_list
        self.multi_handedness = None
        self.multi_hand_world_landmarks = None
        self.hand_sides = sides_list

def main():
    args = parse_args()
    
    # Publishers
    pub_tracking = babyros.node.Publisher(topic="landmarks")
    pub_gesture = babyros.node.Publisher(topic="hand_gestures")
    pub_image = babyros.node.Publisher(topic="image_compressed")
    
    print("✓ BabyROS Publishers initialized:")
    print("  - 'landmarks'")
    print("  - 'hand_gestures'")
    print("  - 'image_compressed'")
    print(f"  Resolution: {args.width}x{args.height}")
    
    if args.use_bpf:
        print(f"  Body Pre-Focusing ENABLED (hand_padding={args.hand_padding})")
    else:
        print("  Body Pre-Focusing disabled")

    bpf = BodyPreFocus(hand_padding_factor=args.hand_padding) if args.use_bpf else None
    gesture_recognizer = GestureRecognizer()

    plotter = None
    left_hand_cloud = right_hand_cloud = pose_cloud = None
    left_hand_lines = right_hand_lines = pose_lines = None

    if args.show_3d and pv is not None:
        plotter = pv.Plotter(window_size=(900, 700))
        plotter.background_color = 'lightgrey'
        plotter.show_grid()
        plotter.add_axes()

        dummy_hand_points = np.zeros((21, 3))
        dummy_pose_points = np.zeros((33, 3))

        hand_line_indices = []
        for conn in mp_hands.HAND_CONNECTIONS:
            hand_line_indices.extend([2, conn[0], conn[1]])

        pose_line_indices = []
        for conn in mp_pose.POSE_CONNECTIONS:
            pose_line_indices.extend([2, conn[0], conn[1]])

        left_hand_cloud = pv.PolyData(dummy_hand_points)
        plotter.add_points(left_hand_cloud, point_size=10, color='green', render_points_as_spheres=True)
        left_hand_lines = pv.PolyData(dummy_hand_points, lines=hand_line_indices)
        plotter.add_mesh(left_hand_lines, color='lime', line_width=3)

        right_hand_cloud = pv.PolyData(dummy_hand_points)
        plotter.add_points(right_hand_cloud, point_size=10, color='yellow', render_points_as_spheres=True)
        right_hand_lines = pv.PolyData(dummy_hand_points, lines=hand_line_indices)
        plotter.add_mesh(right_hand_lines, color='orange', line_width=3)

        pose_cloud = pv.PolyData(dummy_pose_points)
        plotter.add_points(pose_cloud, point_size=10, color='red', render_points_as_spheres=True)
        pose_lines = pv.PolyData(dummy_pose_points, lines=pose_line_indices)
        plotter.add_mesh(pose_lines, color='red', line_width=3)

        plotter.show(auto_close=False, interactive_update=True)


    cap = cv2.VideoCapture(args.camera, cv2.CAP_ANY)
    if not cap.isOpened():
        for idx in [0, 2, 3]:
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if cap.isOpened():
                break
    if not cap.isOpened():
        print("Error: Could not open any camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # --- Apply Camera Hardware Properties ---
    if args.camera_settings:
        print("Opening native camera settings dialog...")
        cap.set(cv2.CAP_PROP_SETTINGS, 1)
        
    if args.brightness >= 0:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, args.brightness)
    if args.contrast >= 0:
        cap.set(cv2.CAP_PROP_CONTRAST, args.contrast)
    if args.sharpness >= 0:
        cap.set(cv2.CAP_PROP_SHARPNESS, args.sharpness)
    if args.gain >= 0:
        cap.set(cv2.CAP_PROP_GAIN, args.gain)
    if args.exposure >= 0:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera opened at {frame_width}x{frame_height}")

    camera_matrix, distortion = load_calibration(args.calibration, frame_width, frame_height)

    hand_rigid_filter_left = RigidTransformFilter(alpha=args.smoothing)
    hand_rigid_filter_right = RigidTransformFilter(alpha=args.smoothing)
    pose_filter = EMAFilter(alpha=args.smoothing)
    pose_rigid_filter = RigidTransformFilter(alpha=args.smoothing)

    # --- Software Enhancement Initialization ---
    if args.enhance:
        print("✓ Software Image Enhancement ENABLED (CLAHE + Unsharp Masking)")
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    else:
        clahe = None

    # --- NEW: Initialize Display Window with Custom Size ---
    window_name = '2D External Camera (Press q to quit, b to toggle BPF)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)
    # -----------------------------------------------------

    with mp_hands.Hands(
            model_complexity=1,
            max_num_hands=2,
            min_detection_confidence=args.detect_conf_hand,
            min_tracking_confidence=args.track_conf_hand) as hands, \
            mp_pose.Pose(
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=True,
            min_detection_confidence=args.detect_conf_pose,
            min_tracking_confidence=args.track_conf_pose) as pose:

        print("\nStarting Tracking & Publishing with Gesture Recognition...")
        print("  Gestures: 'open' (open hand), 'close' (fist), 'partial' (in between)")

        use_bpf = args.use_bpf
        frame_count = 0

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # --- Software Image Enhancement ---
            if clahe is not None:
                lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                l = clahe.apply(l)
                lab = cv2.merge((l, a, b))
                image_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                
                blurred = cv2.GaussianBlur(image_rgb, (0, 0), 2.0)
                image_rgb = cv2.addWeighted(image_rgb, 1.5, blurred, -0.5, 0)
            # -----------------------------------
            
            frame_h, frame_w = image.shape[:2]

            results_pose = pose.process(image_rgb)
            
            left_wrist_elbow, right_wrist_elbow = None, None
            left_crop, right_crop = None, None
            
            if results_pose.pose_landmarks and bpf is not None and use_bpf:
                left_wrist_elbow, right_wrist_elbow = bpf.get_wrist_positions(
                    results_pose.pose_landmarks, frame_w, frame_h
                )
                
                if left_wrist_elbow:
                    left_wrist_px, left_elbow_px = left_wrist_elbow
                    left_crop = bpf.create_hand_crop(left_wrist_px, left_elbow_px, frame_w, frame_h)

                if right_wrist_elbow:
                    right_wrist_px, right_elbow_px = right_wrist_elbow
                    right_crop = bpf.create_hand_crop(right_wrist_px, right_elbow_px, frame_w, frame_h)

            # --- FIXED BPF LOGIC ---
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
                    if left_crop:
                        crops_to_process.append(('left', left_crop))
                    if right_crop:
                        crops_to_process.append(('right', right_crop))
                    
                    all_hand_landmarks = []
                    all_hand_sides = []
                    
                    for side, crop in crops_to_process:
                        x_min, y_min, x_max, y_max = crop
                        if x_max <= x_min or y_max <= y_min:
                            continue
                        cropped_image = image_rgb[y_min:y_max, x_min:x_max]
                        if cropped_image.size == 0:
                            continue
                            
                        crop_results = hands.process(cropped_image)
                                        
                        if crop_results.multi_hand_landmarks:
                            hands_detected_in_crop = True
                            for hand_landmarks in crop_results.multi_hand_landmarks:
                                bpf.shift_landmarks_to_original(hand_landmarks, crop, frame_w, frame_h)
                                all_hand_landmarks.append(hand_landmarks)
                                all_hand_sides.append(side)
                    
                    if all_hand_landmarks:
                        results_hands = SyntheticResults(all_hand_landmarks, all_hand_sides)

                if not hands_detected_in_crop:
                    results_hands = hands.process(image_rgb)
            else:
                results_hands = hands.process(image_rgb)
            # -----------------------

            # Draw pose landmarks
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Draw hand landmarks
            if results_hands and getattr(results_hands, 'multi_hand_landmarks', None):
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                
                if use_bpf and bpf is not None:
                    if left_crop:
                        x_min, y_min, x_max, y_max = left_crop
                        cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(image_bgr, "Left Hand ROI", (x_min, y_min-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if right_crop:
                        x_min, y_min, x_max, y_max = right_crop
                        cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        cv2.putText(image_bgr, "Right Hand ROI", (x_min, y_min-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Process 3D WORLD coordinates for Pose
            pose_world_points = None
            if results_pose.pose_world_landmarks and results_pose.pose_landmarks:
                model_points = np.array([[lm.x, lm.y, lm.z] for lm in results_pose.pose_world_landmarks.landmark])
                image_points = np.array([[lm.x * frame_width, lm.y * frame_height] for lm in results_pose.pose_landmarks.landmark])

                success_pnp, rvec, tvec = cv2.solvePnP(
                    model_points, image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)
                    
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
                    
                    if args.show_3d and pose_cloud is not None:
                        pose_cloud.points = pose_world_points
                        pose_lines.points = pose_world_points

            # Process Hands and recognize gestures
            left_hand_final_points = None
            right_hand_final_points = None
            left_gesture = "unknown"
            right_gesture = "unknown"

            if results_hands and getattr(results_hands, 'multi_hand_landmarks', None):
                for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                    gesture = gesture_recognizer.recognize_gesture(hand_landmarks.landmark)
                    
                    if results_hands.multi_hand_world_landmarks and idx < len(results_hands.multi_hand_world_landmarks):
                        world_landmarks = results_hands.multi_hand_world_landmarks[idx]
                        hand_model_points = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks.landmark])
                    else:
                        hand_model_points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    
                    hand_image_points = np.array([[lm.x * frame_width, lm.y * frame_height] for lm in hand_landmarks.landmark])
                    
                    if hasattr(results_hands, 'hand_sides') and idx < len(results_hands.hand_sides):
                        is_left_hand = (results_hands.hand_sides[idx] == 'left')
                    else:
                        wrist_x = hand_image_points[0, 0]
                        is_left_hand = (wrist_x > frame_width / 2)
                    
                    if is_left_hand:
                        left_gesture = gesture
                    else:
                        right_gesture = gesture
                    
                    success_pnp, rvec, tvec = cv2.solvePnP(
                        hand_model_points, hand_image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)
                                        
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
                                offset = hand_world_points - hand_world_points[0]
                                hand_world_points = pose_world_points[15] + offset
                            elif not is_left_hand and len(pose_world_points) > 16:
                                offset = hand_world_points - hand_world_points[0]
                                hand_world_points = pose_world_points[16] + offset
                        
                        if is_left_hand:
                            if args.show_3d and left_hand_cloud is not None:
                                left_hand_cloud.points = hand_world_points
                                left_hand_lines.points = hand_world_points
                            left_hand_final_points = hand_world_points.copy()
                        else:
                            if args.show_3d and right_hand_cloud is not None:
                                right_hand_cloud.points = hand_world_points
                                right_hand_lines.points = hand_world_points
                            right_hand_final_points = hand_world_points.copy()

            # --- PUBLISH TRACKING DATA ---
            poses_list = []
            if pose_world_points is not None and len(pose_world_points) == 33:
                for idx in range(33):
                    pt = pose_world_points[idx]
                    poses_list.append({
                        "position": {"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                    })

            if left_hand_final_points is not None and len(left_hand_final_points) == 21:
                for pt in left_hand_final_points:
                    poses_list.append({
                        "position": {"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                    })

            if right_hand_final_points is not None and len(right_hand_final_points) == 21:
                for pt in right_hand_final_points:
                    poses_list.append({
                        "position": {"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                    })

            if len(poses_list) > 0:
                msg = {
                    "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "world"},
                    "poses": poses_list
                }
                pub_tracking.publish(msg)

            # --- PUBLISH GESTURE DATA ---
            gesture_msg = {
                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "world"},
                "left_hand": left_gesture,
                "right_hand": right_gesture
            }
            pub_gesture.publish(gesture_msg)

            # --- PUBLISH COMPRESSED IMAGE (FIXED) ---
            # Compress the BGR image to JPEG. Quality 80 is a great balance.
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            success_encode, encoded_image = cv2.imencode('.jpg', image_bgr, encode_param)
            if success_encode:
                # Convert numpy array to raw bytes, then encode to Base64 string for JSON serialization
                img_bytes = encoded_image.tobytes()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                img_msg = {
                    "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera"},
                    "format": "jpeg",
                    "data": img_b64  # <-- Now a JSON-serializable string
                }
                pub_image.publish(img_msg)
            # ----------------------------------------

            # Update 3D view if enabled
            if args.show_3d and plotter is not None:
                plotter.update()

            # Show 2D webcam feed with gesture info
            image_bgr_mirror = cv2.flip(image_bgr, 1)
            
            if use_bpf:
                cv2.putText(image_bgr_mirror, "BPF: ON", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(image_bgr_mirror, "BPF: OFF", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if clahe is not None:
                cv2.putText(image_bgr_mirror, "Enhance: ON", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(image_bgr_mirror, "Enhance: OFF", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(image_bgr_mirror, f"Left: {left_gesture}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(image_bgr_mirror, f"Right: {right_gesture}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # --- UPDATED: Use the predefined window_name ---
            cv2.imshow(window_name, image_bgr_mirror)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                use_bpf = not use_bpf
                print(f"{'✓' if use_bpf else '✗'} Body Pre-Focusing {'enabled' if use_bpf else 'disabled'}")

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    if args.show_3d and plotter is not None:
        plotter.close()


if __name__ == "__main__":
    main()
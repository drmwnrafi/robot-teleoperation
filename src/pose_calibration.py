import os, sys
import cv2
import numpy as np
import onnxruntime as ort
from utils.yolo_detector import YoloOnnxPoseDetector
import time, glob
from utils.helpers import (
    h36m17_to_h36m16, normalize_keypoints, draw_3d_skeleton_cv2,
    mouse_callback, compensate_shoulder_yaw
)

CAPTURE_INTERVAL = 3.0     
JPEG_QUALITY     = 95
YOLO_MODEL = 'models/yolo26s-pose.onnx'
POSE_MODEL = 'models/poseaug_videopose_1x16x2.onnx'
CALIB_DIR  = 'calib_photos'
CALIB_FILE = 'calib_layer.npz'

LABELS = {
    'f': np.array([0, 0, 1]), 'b': np.array([0, 0,-1]),
    'u': np.array([0, 1, 0]), 'd': np.array([0,-1, 0]),
    'r': np.array([1, 0, 0]), 'l': np.array([-1,0, 0]),
}
DISPLAY_SCALE = 0.5
H36M17_SKELETON = [
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16), (0, 4), (4, 5), (5, 6), (0, 1), (1, 2), (2, 3)
]

I17 = dict(ls=11, le=12, lw=13, rs=14, re=15, rw=16, pelvis=0, thorax=8)

def _build_i16():
    probe  = np.tile(np.arange(17, dtype=np.float32).reshape(17, 1), (1, 2))
    mapped = h36m17_to_h36m16(probe)[:, 0].astype(int)
    pos = {orig: i for i, orig in enumerate(mapped)}
    return dict(ls=pos[11], le=pos[12], lw=pos[13], rs=pos[14],
                re=pos[15], rw=pos[16], pelvis=pos[0], thorax=pos[8])
I16 = _build_i16()
print("H36M-16 joint map:", I16)

def extract_3d_arm_vec(pose_raw):
    LS, LE, LW = 10, 11, 12
    RS, RE, RW = 13, 14, 15
    mid_sh = (pose_raw[LS] + pose_raw[RS]) / 2.0
    return (pose_raw[[LE, LW, RE, RW]] - mid_sh).flatten()

detector = YoloOnnxPoseDetector(model_path=YOLO_MODEL, conf_thres=0.5)
session  = ort.InferenceSession(POSE_MODEL)
IN_NAME  = session.get_inputs()[0].name

def force_straight_arms_torso_frame(pose_3d):
    pose = pose_3d.copy()
    LS, LE, LW, RS, RE, RW = 10, 11, 12, 13, 14, 15
    l_sh, r_sh = pose[LS], pose[RS]
    sh_dx, sh_dz = r_sh[0] - l_sh[0], r_sh[2] - l_sh[2]
    norm_sh = np.hypot(sh_dx, sh_dz)
    if norm_sh < 1e-6: return pose
    fwd_x, fwd_z = -sh_dz / norm_sh, sh_dx / norm_sh
    for S_idx, E_idx, W_idx in [(LS, LE, LW), (RS, RE, RW)]:
        S, E, W = pose[S_idx], pose[E_idx], pose[W_idx]
        len_e, len_w = np.hypot(E[0] - S[0], E[2] - S[2]), np.hypot(W[0] - S[0], W[2] - S[2])
        sign_e = np.sign((E[0] - S[0]) * fwd_x + (E[2] - S[2]) * fwd_z) or 1
        sign_w = np.sign((W[0] - S[0]) * fwd_x + (W[2] - S[2]) * fwd_z) or 1
        pose[E_idx] = np.array([S[0] + sign_e * len_e * fwd_x, E[1], S[2] + sign_e * len_e * fwd_z])
        pose[W_idx] = np.array([S[0] + sign_w * len_w * fwd_x, W[1], S[2] + sign_w * len_w * fwd_z])
    return pose

def apply_force_straight_with_confidence(pose_raw, confidence_score):
    return pose_raw * (1.0 - confidence_score) + force_straight_arms_torso_frame(pose_raw.copy()) * confidence_score

class AnthroArgs:
    def __init__(self):
        self.pelvis_width, self.clavicle_length = 35.0, 42.0
        self.humerus_length, self.forearm_length, self.torso_height = 33.5, 27.0, 50.0

ANTHRO_ARGS = AnthroArgs()

def apply_anthropometric_scaling(pose_world_points, args):
    if pose_world_points is None or len(pose_world_points) < 16:
        return pose_world_points
    scaled_points = pose_world_points.copy()
    L_HIP, R_HIP, L_SH, R_SH, L_EL, L_WR, R_EL, R_WR = 4, 1, 10, 13, 11, 12, 14, 15
    L_KNEE, L_ANKLE, R_KNEE, R_ANKLE, SPINE, THORAX, HEAD = 5, 6, 2, 3, 7, 8, 9

    def get_dist(i, j): return np.linalg.norm(scaled_points[i] - scaled_points[j])

    if args.pelvis_width > 0:
        current = get_dist(L_HIP, R_HIP)
        if current > 1e-5:
            scale = (args.pelvis_width / 100.0) / current
            center = (scaled_points[L_HIP] + scaled_points[R_HIP]) / 2.0
            dl, dr = (center + (scaled_points[L_HIP] - center) * scale) - scaled_points[L_HIP], (center + (scaled_points[R_HIP] - center) * scale) - scaled_points[R_HIP]
            scaled_points[L_HIP], scaled_points[R_HIP] = scaled_points[L_HIP] + dl, scaled_points[R_HIP] + dr
            for idx in [L_KNEE, L_ANKLE]: scaled_points[idx] += dl
            for idx in [R_KNEE, R_ANKLE]: scaled_points[idx] += dr

    if args.clavicle_length > 0:
        current = get_dist(L_SH, R_SH)
        if current > 1e-5:
            scale = (args.clavicle_length / 100.0) / current
            center = (scaled_points[L_SH] + scaled_points[R_SH]) / 2.0
            dl, dr = (center + (scaled_points[L_SH] - center) * scale) - scaled_points[L_SH], (center + (scaled_points[R_SH] - center) * scale) - scaled_points[R_SH]
            scaled_points[L_SH], scaled_points[R_SH] = scaled_points[L_SH] + dl, scaled_points[R_SH] + dr
            for idx in [L_EL, L_WR]: scaled_points[idx] += dl
            for idx in [R_EL, R_WR]: scaled_points[idx] += dr

    if args.torso_height > 0:
        dl, dr = np.zeros(3), np.zeros(3)
        for side, hip, sh, el, wr in [('L', L_HIP, L_SH, L_EL, L_WR), ('R', R_HIP, R_SH, R_EL, R_WR)]:
            current = np.linalg.norm(scaled_points[sh] - scaled_points[hip])
            if current > 1e-5:
                new_sh = scaled_points[hip] + (scaled_points[sh] - scaled_points[hip]) * ((args.torso_height / 100.0) / current)
                delta = new_sh - scaled_points[sh]
                scaled_points[sh] = new_sh
                for idx in [el, wr]: scaled_points[idx] += delta
                if side == 'L': dl = delta
                else: dr = delta
        for idx in [SPINE, THORAX, HEAD]: scaled_points[idx] += (dl + dr) / 2.0

    def scale_bone(p_idx, c_idx, desc, target_cm):
        if target_cm <= 0: return
        current = np.linalg.norm(scaled_points[c_idx] - scaled_points[p_idx])
        if current > 1e-5:
            delta = (scaled_points[p_idx] + (scaled_points[c_idx] - scaled_points[p_idx]) * ((target_cm / 100.0) / current)) - scaled_points[c_idx]
            scaled_points[c_idx] += delta
            for d in desc: scaled_points[d] += delta

    scale_bone(L_SH, L_EL, [L_WR], args.humerus_length)
    scale_bone(L_EL, L_WR, [], args.forearm_length)
    scale_bone(R_SH, R_EL, [R_WR], args.humerus_length)
    scale_bone(R_EL, R_WR, [], args.forearm_length)
    return scaled_points

def predict(frame):
    h, w = frame.shape[:2]
    _, _, kp17 = detector(frame)
    if kp17 is None: return None, None
    kp16 = h36m17_to_h36m16(kp17)
    inp = normalize_keypoints(kp16, h, w).astype(np.float32).reshape(1, 16, 2)
    pose_raw = session.run(None, {IN_NAME: inp})[0].reshape(-1, 3)
    return pose_raw.astype(np.float32), kp17

def calibrate_from_arm_joints(pose_3d):
    LS, LE, LW, RS, RE, RW, L_HIP, R_HIP = 10, 11, 12, 13, 14, 15, 4, 1
    arm_vectors = [pose_3d[LE]-pose_3d[LS], pose_3d[LW]-pose_3d[LE], pose_3d[LW]-pose_3d[LS],
                   pose_3d[RE]-pose_3d[RS], pose_3d[RW]-pose_3d[RE], pose_3d[RW]-pose_3d[RS]]
    avg_forward = np.mean(arm_vectors, axis=0)
    fwd_xz = np.array([avg_forward[0], avg_forward[2]])
    if np.linalg.norm(fwd_xz) < 1e-6:
        shoulder_vec = pose_3d[RS] - pose_3d[LS]
        fwd_xz = np.array([-shoulder_vec[2], shoulder_vec[0]])
    forward_xz = fwd_xz / np.linalg.norm(fwd_xz)
    
    shoulder_xz = np.array([pose_3d[RS][0] - pose_3d[LS][0], pose_3d[RS][2] - pose_3d[LS][2]])
    right_xz = shoulder_xz / np.linalg.norm(shoulder_xz) if np.linalg.norm(shoulder_xz) > 1e-6 else np.array([-forward_xz[1], forward_xz[0]])
    right_xz = right_xz - np.dot(right_xz, forward_xz) * forward_xz
    right_xz = right_xz / np.linalg.norm(right_xz) if np.linalg.norm(right_xz) > 1e-6 else np.array([-forward_xz[1], forward_xz[0]])
    
    forward_3d, right_3d = np.array([forward_xz[0], 0.0, forward_xz[1]]), np.array([right_xz[0], 0.0, right_xz[1]])
    up_3d = ((pose_3d[LS] + pose_3d[RS]) / 2.0) - ((pose_3d[L_HIP] + pose_3d[R_HIP]) / 2.0)
    up_3d = up_3d - np.dot(up_3d, forward_3d) * forward_3d - np.dot(up_3d, right_3d) * right_3d
    up_3d = up_3d / np.linalg.norm(up_3d) if np.linalg.norm(up_3d) > 1e-6 else np.cross(forward_3d, right_3d)
    
    return {'forward': forward_3d, 'right': right_3d, 'up': up_3d, 'basis': np.stack([right_3d, up_3d, forward_3d]), 'arm_vectors': np.array(arm_vectors)}

class CalibrationLayer:
    def __init__(self):
        self.V, self.T, self.U = {'L': [], 'R': []}, [], []
        self.X, self.mode = None, None
        self.forward_ref_2d, self.forward_ref_3d_raw = None, None

    def add(self, target, vL, vR, up):
        self.T.append(target); self.U.append(up); self.V['L'].append(vL); self.V['R'].append(vR)

    def finish(self):
        T = np.array(self.T)
        if np.linalg.matrix_rank(T, tol=0.15) >= 3:
            self.mode = 'matrix'; self.X = {}
            for s in ('L', 'R'):
                Vm = np.array(self.V[s])
                self.X[s], *_ = np.linalg.lstsq(Vm, T, rcond=None)
                print(f"  {s} arm solved, mean residual {np.mean(np.linalg.norm(Vm @ self.X[s] - T, axis=1)):.3f}")
            print(f"✔ 3x3 layer built ({len(T)} samples)")
        else:
            self.mode = 'basis'
            Z = np.mean(np.array(self.V['L']) + np.array(self.V['R']), axis=0); Z /= np.linalg.norm(Z) + 1e-6
            Y = np.mean(self.U, axis=0); Y = Y - np.dot(Y, Z) * Z; Y /= np.linalg.norm(Y) + 1e-6
            Xr = np.cross(Y, Z); Xr /= np.linalg.norm(Xr) + 1e-6
            self.X = {'L': np.stack([Xr, Y, Z]).T, 'R': np.stack([Xr, Y, Z]).T}
            print(f"⚠ rank < 3 -> basis mode")
        return self

    def compute_forward_score_2d(self, kp2d):
        if self.forward_ref_2d is None: return 1.0
        ref, cur = self.forward_ref_2d, kp2d
        ref_center, cur_center = (ref[11] + ref[14]) / 2.0, (cur[11] + cur[14]) / 2.0
        ref_vec = (ref[[12, 13, 15, 16]] - ref_center).flatten()
        cur_vec = (cur[[12, 13, 15, 16]] - cur_center).flatten()
        similarity = float(np.dot(ref_vec / (np.linalg.norm(ref_vec) + 1e-6), cur_vec / (np.linalg.norm(cur_vec) + 1e-6)))
        score = np.clip((similarity - 0.60) / 0.30, 0.0, 1.0)
        if (ref[13][0] < ref[16][0]) != (cur[13][0] < cur[16][0]): score *= 0.25
        return float(score)

    def compute_forward_score_3d(self, pose_raw):
        if self.forward_ref_3d_raw is None: return 1.0
        cur_vec, ref_vec = extract_3d_arm_vec(pose_raw), self.forward_ref_3d_raw
        similarity = float(np.dot(cur_vec / (np.linalg.norm(cur_vec) + 1e-6), ref_vec / (np.linalg.norm(ref_vec) + 1e-6)))
        return float(np.clip((similarity - 0.60) / 0.30, 0.0, 1.0))

    def compute_forward_score(self, kp2d, pose_raw=None):
        s2d = self.compute_forward_score_2d(kp2d)
        s3d = self.compute_forward_score_3d(pose_raw) if pose_raw is not None else 1.0
        return s2d * s3d, s2d, s3d

    def correct(self, vec, side): return vec @ self.X[side]

    def save(self, path=CALIB_FILE):
        np.savez(path, mode=self.mode, XL=self.X['L'], XR=self.X['R'],
                 forward_ref_2d=self.forward_ref_2d if self.forward_ref_2d is not None else np.array([]),
                 forward_ref_3d_raw=self.forward_ref_3d_raw if self.forward_ref_3d_raw is not None else np.array([]))
        print("saved", path)

    def load(self, path=CALIB_FILE):
        d = np.load(path, allow_pickle=True)
        self.mode, self.X = str(d['mode']), {'L': d['XL'], 'R': d['XR']}
        ref2, ref3 = d.get('forward_ref_2d', None), d.get('forward_ref_3d_raw', None)
        if ref2 is not None and ref2.size > 0: self.forward_ref_2d = ref2
        if ref3 is not None and ref3.size > 0: self.forward_ref_3d_raw = ref3
        print(f"loaded {path} ({self.mode})")
        return self

def collect(cam=0, width=1920, height=1080):
    calib, counts, current_label, countdown_start = CalibrationLayer(), {k: 0 for k in LABELS}, None, None
    mouse_state = {'rot_x': 0.0, 'rot_y': 0.0, 'zoom_level': 1.0, 'dragging': False, 'zooming': False, 'last_mx': 0, 'last_my': 0}

    cap = cv2.VideoCapture(cam, cv2.CAP_ANY)
    if not cap.isOpened():
        for idx in [0, 2, 3]:
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if cap.isOpened(): break
    if not cap.isOpened(): return print("✗ Could not open any camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1); cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    actual_w, actual_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_w, disp_h = int(actual_w * DISPLAY_SCALE), int(actual_h * DISPLAY_SCALE)
    vp_w, out_w = disp_w // 2, disp_w + vp_w
    print(f"Display: {disp_w}x{disp_h} + 3D Viewport: {vp_w}x{disp_h}")

    for _ in range(10): cap.read()
    cv2.namedWindow('COLLECT', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('COLLECT', mouse_callback, mouse_state) # <-- PASS STATE HERE

    print("1) direction key f/u/d/l/r/b   2) SPACE = countdown+capture   ENTER = finish")
    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(0.05); continue
        disp_frame = cv2.resize(frame, (disp_w, disp_h))
        canvas = np.zeros((disp_h, out_w, 3), dtype=np.uint8)
        canvas[:, :disp_w] = disp_frame
        pose_raw, kp17 = predict(frame)
        
        if pose_raw is not None and kp17 is not None:
            pose_processed = apply_anthropometric_scaling(apply_force_straight_with_confidence(pose_raw, 1.0), ANTHRO_ARGS)
            for i, j in H36M17_SKELETON:
                cv2.line(canvas, (int(kp17[i][0]*DISPLAY_SCALE), int(kp17[i][1]*DISPLAY_SCALE)), (int(kp17[j][0]*DISPLAY_SCALE), int(kp17[j][1]*DISPLAY_SCALE)), (0, 255, 0), 2)
            for i in range(17):
                cv2.circle(canvas, (int(kp17[i][0]*DISPLAY_SCALE), int(kp17[i][1]*DISPLAY_SCALE)), 3, (0, 0, 255), -1)
            draw_3d_skeleton_cv2(canvas, pose_processed, vp_x=disp_w, vp_y=0, vp_w=vp_w, vp_h=disp_h,
                                 angle_x=mouse_state['rot_x'], angle_y=mouse_state['rot_y'], zoom=mouse_state['zoom_level']) # <-- USE STATE HERE

        cv2.putText(canvas, f"selected: {current_label}" if current_label else "select: f u d l r b", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(canvas, f"captured: {counts}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, "SPACE=countdown+capture | ENTER=finish", (10, disp_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        if countdown_start is not None:
            remaining = CAPTURE_INTERVAL - (time.time() - countdown_start)
            if remaining > 0:
                cv2.putText(canvas, str(int(remaining) + 1), (disp_w // 2 - 15, disp_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4)
            else:
                countdown_start = None
                d = os.path.join(CALIB_DIR, current_label); os.makedirs(d, exist_ok=True)
                counts[current_label] += 1; base_name = f"{counts[current_label]:02d}"
                jpg_path, npz_path = os.path.join(d, f"{base_name}.jpg"), os.path.join(d, f"{base_name}.npz")
                cv2.imwrite(jpg_path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if pose_raw is not None and kp17 is not None:
                    pose_processed = apply_anthropometric_scaling(apply_force_straight_with_confidence(pose_raw, 1.0), ANTHRO_ARGS)
                    vL, vR = pose_processed[I16['lw']] - pose_processed[I16['ls']], pose_processed[I16['rw']] - pose_processed[I16['rs']]
                    calib_result = calibrate_from_arm_joints(pose_processed)
                    np.savez_compressed(npz_path, kp2d=kp17, pose3d_raw=pose_raw, pose3d_processed=pose_processed, label=current_label,
                                        arm_length_L=np.linalg.norm(vL), arm_length_R=np.linalg.norm(vR),
                                        forward_axis=calib_result['forward'], right_axis=calib_result['right'], up_axis=calib_result['up'],
                                        arm_vectors=calib_result['arm_vectors'], basis=calib_result['basis'])
                    calib.add(LABELS[current_label], vL, vR, pose_processed[I16['thorax']] - pose_processed[I16['pelvis']])
                    print(f'✔ {jpg_path} (arm L:{np.linalg.norm(vL)*100:.1f}cm R:{np.linalg.norm(vR)*100:.1f}cm)')
                else: print(f'✖ {jpg_path} (no person detected)')

        cv2.imshow('COLLECT', canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == ord('q'): break
        elif key == ord(' ') and countdown_start is None:
            if current_label is None: print('select a direction key first (f/u/d/l/r/b)')
            else: countdown_start = time.time()
        elif 0 <= key < 256 and chr(key) in LABELS: current_label = chr(key)

    cap.release(); cv2.destroyAllWindows()
    if calib.T: calib.finish().save()
    else: print("no samples added. Photos in", CALIB_DIR, "— run: uv run python pose_calibration.py build")

def live(cam=0, width=1920, height=1080):
    calib = CalibrationLayer().load()
    mouse_state = {'rot_x': 90.0, 'rot_y': 0.0, 'zoom_level': 1.0, 'dragging': False, 'zooming': False, 'last_mx': 0, 'last_my': 0} # <-- INIT STATE

    cap = cv2.VideoCapture(cam, cv2.CAP_ANY)
    if not cap.isOpened():
        for idx in [0, 2, 3]:
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if cap.isOpened(): break
    if not cap.isOpened(): return print("✗ Could not open camera for live view.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1); cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    actual_w, actual_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_w, disp_h = int(actual_w * DISPLAY_SCALE), int(actual_h * DISPLAY_SCALE)
    vp_w, out_w = disp_w // 2, disp_w + vp_w
    for _ in range(10): cap.read()

    cv2.namedWindow('LIVE CALIBRATED (q=quit)', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('LIVE CALIBRATED (q=quit)', mouse_callback, mouse_state) # <-- PASS STATE HERE

    while True:
        ret, frame = cap.read()
        if not ret: break
        disp_frame = cv2.resize(frame, (disp_w, disp_h))
        canvas = np.zeros((disp_h, out_w, 3), dtype=np.uint8)
        canvas[:, :disp_w] = disp_frame
        pose_raw, kp17 = predict(frame)
        
        if pose_raw is not None and kp17 is not None:
            fwd_score, s2d, s3d = calib.compute_forward_score(kp17, pose_raw)
            pose_processed = apply_anthropometric_scaling(apply_force_straight_with_confidence(pose_raw, fwd_score), ANTHRO_ARGS)
            for i, j in H36M17_SKELETON:
                cv2.line(canvas, (int(kp17[i][0]*DISPLAY_SCALE), int(kp17[i][1]*DISPLAY_SCALE)), (int(kp17[j][0]*DISPLAY_SCALE), int(kp17[j][1]*DISPLAY_SCALE)), (0, 255, 0), 2)
            for i in range(17):
                cv2.circle(canvas, (int(kp17[i][0]*DISPLAY_SCALE), int(kp17[i][1]*DISPLAY_SCALE)), 3, (0, 0, 255), -1)
            
            draw_3d_skeleton_cv2(canvas, compensate_shoulder_yaw(pose_processed), vp_x=disp_w, vp_y=0, vp_w=vp_w, vp_h=disp_h,
                                 angle_x=mouse_state['rot_x'], angle_y=mouse_state['rot_y'], zoom=mouse_state['zoom_level']) # <-- USE STATE HERE

            vL, vR = pose_processed[I16['lw']] - pose_processed[I16['ls']], pose_processed[I16['rw']] - pose_processed[I16['rs']]
            cL, cR = calib.correct(vL, 'L'), calib.correct(vR, 'R')
            cL[2] *= fwd_score; cR[2] *= fwd_score
            
            cv2.putText(canvas, f"2d:{s2d:.2f} 3d:{s3d:.2f} -> {fwd_score:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(canvas, f"L  x{cL[0]:+.2f} y{cL[1]:+.2f} z{cL[2]:+.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
            cv2.putText(canvas, f"R  x{cR[0]:+.2f} y{cR[1]:+.2f} z{cR[2]:+.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
            
        cv2.imshow('LIVE CALIBRATED (q=quit)', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

def photo(file_path):
    calib = CalibrationLayer().load()
    mouse_state = {'rot_x': 0.0, 'rot_y': 0.0, 'zoom_level': 1.0, 'dragging': False, 'zooming': False, 'last_mx': 0, 'last_my': 0} # <-- INIT STATE

    if not os.path.exists(file_path): return print(f"✗ File not found: {file_path}")
    frame = cv2.imread(file_path)
    if frame is None: return print(f"✗ Could not read image: {file_path}")

    actual_h, actual_w = frame.shape[:2]
    disp_w, disp_h = int(actual_w * DISPLAY_SCALE), int(actual_h * DISPLAY_SCALE)
    vp_w, out_w = disp_w // 2, disp_w + vp_w
    
    cv2.namedWindow('PHOTO MODE (q=quit)', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('PHOTO MODE (q=quit)', mouse_callback, mouse_state) # <-- PASS STATE HERE
    print(f"Processing: {file_path} ({actual_w}x{actual_h})")
    
    pose_raw, kp17 = predict(frame)
    disp_frame = cv2.resize(frame, (disp_w, disp_h))
    base_canvas = np.zeros((disp_h, out_w, 3), dtype=np.uint8)
    base_canvas[:, :disp_w] = disp_frame
    fwd_score, s2d, s3d, cL_text, cR_text = 1.0, 1.0, 1.0, "N/A", "N/A"
    
    if pose_raw is not None and kp17 is not None:
        fwd_score, s2d, s3d = calib.compute_forward_score(kp17, pose_raw)
        print(f"✔ 2D Score: {s2d:.3f}  |  3D Score: {s3d:.3f}  |  Combined: {fwd_score:.3f}")
        pose_processed = apply_anthropometric_scaling(apply_force_straight_with_confidence(pose_raw, fwd_score), ANTHRO_ARGS)
        for i, j in H36M17_SKELETON:
            cv2.line(base_canvas, (int(kp17[i][0]*DISPLAY_SCALE), int(kp17[i][1]*DISPLAY_SCALE)), (int(kp17[j][0]*DISPLAY_SCALE), int(kp17[j][1]*DISPLAY_SCALE)), (0, 255, 0), 2)
        for i in range(17):
            cv2.circle(base_canvas, (int(kp17[i][0]*DISPLAY_SCALE), int(kp17[i][1]*DISPLAY_SCALE)), 3, (0, 0, 255), -1)
            
        vL, vR = pose_processed[I16['lw']] - pose_processed[I16['ls']], pose_processed[I16['rw']] - pose_processed[I16['rs']]
        cL, cR = calib.correct(vL, 'L'), calib.correct(vR, 'R')
        cL[2] *= fwd_score; cR[2] *= fwd_score
        cL_text, cR_text = f"L  x{cL[0]:+.2f} y{cL[1]:+.2f} z{cL[2]:+.2f}", f"R  x{cR[0]:+.2f} y{cR[1]:+.2f} z{cR[2]:+.2f}"
    else:
        print("✗ NO PERSON DETECTED")
        cv2.putText(base_canvas, "NO PERSON DETECTED", (10, disp_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    print("Drag mouse on 3D view to rotate. Scroll to zoom. Press 'q' to quit.")
    while True:
        canvas = base_canvas.copy()
        if pose_raw is not None and kp17 is not None:
            draw_3d_skeleton_cv2(canvas, compensate_shoulder_yaw(pose_processed), vp_x=disp_w, vp_y=0, vp_w=vp_w, vp_h=disp_h,
                                 angle_x=mouse_state['rot_x'], angle_y=mouse_state['rot_y'], zoom=mouse_state['zoom_level']) # <-- USE STATE HERE
            cv2.putText(canvas, f"2d:{s2d:.2f} 3d:{s3d:.2f} -> {fwd_score:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(canvas, cL_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
            cv2.putText(canvas, cR_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
            
        cv2.imshow('PHOTO MODE (q=quit)', canvas)
        if cv2.waitKey(30) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

def build_from_folder(base_dir=CALIB_DIR):
    calib, forward_2d_list, forward_3d_list = CalibrationLayer(), [], []
    for label in LABELS:
        folder = os.path.join(base_dir, label)
        if not os.path.isdir(folder): continue
        npz_files = sorted(glob.glob(os.path.join(folder, '*.npz')))
        if npz_files:
            for p in npz_files:
                data = np.load(p)
                pose_processed = apply_anthropometric_scaling(apply_force_straight_with_confidence(data['pose3d_raw'], 1.0), ANTHRO_ARGS)
                vL, vR = pose_processed[I16['lw']] - pose_processed[I16['ls']], pose_processed[I16['rw']] - pose_processed[I16['rs']]
                calib.add(LABELS[label], vL, vR, pose_processed[I16['thorax']] - pose_processed[I16['pelvis']])
                if label == 'f':
                    forward_2d_list.append(data['kp2d']); forward_3d_list.append(extract_3d_arm_vec(data['pose3d_raw']))
                print(f'✔ {p} (from NPZ, arm L:{np.linalg.norm(vL)*100:.1f}cm R:{np.linalg.norm(vR)*100:.1f}cm)')
        else:
            for p in sorted(glob.glob(os.path.join(folder, '*.jpg')) + glob.glob(os.path.join(folder, '*.jpeg'))):
                frame = cv2.imread(p)
                if frame is None: continue
                pose_raw, kp17 = predict(frame)
                if pose_raw is None: print(f'✖ no person: {p}'); continue
                pose_processed = apply_anthropometric_scaling(apply_force_straight_with_confidence(pose_raw, 1.0), ANTHRO_ARGS)
                vL, vR = pose_processed[I16['lw']] - pose_processed[I16['ls']], pose_processed[I16['rw']] - pose_processed[I16['rs']]
                calib.add(LABELS[label], vL, vR, pose_processed[I16['thorax']] - pose_processed[I16['pelvis']])
                if label == 'f':
                    forward_2d_list.append(kp17); forward_3d_list.append(extract_3d_arm_vec(pose_raw))
                print(f'✔ {p} (arm L:{np.linalg.norm(vL)*100:.1f}cm R:{np.linalg.norm(vR)*100:.1f}cm)')

    if forward_2d_list:
        calib.forward_ref_2d = np.mean(forward_2d_list, axis=0).astype(np.float32)
        print(f"✔ Forward 2D reference built from {len(forward_2d_list)} samples")
    if forward_3d_list:
        calib.forward_ref_3d_raw = np.mean(forward_3d_list, axis=0).astype(np.float32)
        print(f"✔ Forward 3D reference built from {len(forward_3d_list)} samples")
                
    if calib.T: calib.finish().save()
    else: print('no usable photos in', base_dir)

if __name__ == '__main__':
    if len(sys.argv) >= 4 and sys.argv[1] == 'photo' and sys.argv[2] == '--file':
        photo(sys.argv[3])
    else:
        cmd = sys.argv[1] if len(sys.argv) > 1 else ''
        if cmd == 'collect': collect()
        elif cmd == 'build': build_from_folder()
        elif cmd == 'live':  live()
        else:
            if os.path.exists(CALIB_FILE): live()
            else: print("usage: uv run python pose_calibration.py [collect|build|live] OR uv run python pose_calibration.py photo --file <path>")
import numpy as np
import cv2

FIXED_SCALE = 500.0

def h36m17_to_h36m16(kp17):
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
    return kp17[indices]

def normalize_keypoints(kp17, h, w):
    return kp17 / w * 2 - np.array([1, h / w], dtype=np.float32)

def view_rotation(angle_y, angle_x):
    rad_x = np.radians(angle_x)
    rad_y = np.radians(angle_y)
    Ry = np.array([
        [ np.cos(rad_y), 0, np.sin(rad_y)],
        [ 0,             1, 0            ],
        [-np.sin(rad_y), 0, np.cos(rad_y)]
    ], dtype=np.float32)
    Rx = np.array([
        [1, 0,              0             ],
        [0, np.cos(rad_x), -np.sin(rad_x)],
        [0, np.sin(rad_x),  np.cos(rad_x)]
    ], dtype=np.float32)
    return Rx @ Ry

def project_points(points_3d, R, zoom, scale, center_x, center_y):
    rotated = points_3d @ R.T
    depth = rotated[:, 2].copy()
    depth = np.where(depth < 1e-6, 1e-6, depth)
    factor = (scale * zoom) / depth
    pts_2d = np.zeros((points_3d.shape[0], 2), dtype=np.float32)
    pts_2d[:, 0] = rotated[:, 0] * factor + center_x
    pts_2d[:, 1] = rotated[:, 1] * factor + center_y
    return pts_2d, depth

def draw_3d_skeleton_cv2(img, pose_3d, vp_x, vp_y, vp_w, vp_h, angle_x, angle_y, zoom=1.0):
    if pose_3d is None or pose_3d.ndim != 2:
        return img

    num_joints = pose_3d.shape[0]
    if num_joints == 16:
        SKELETON = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (8, 10), (10, 11), (11, 12), (8, 13), (13, 14), (14, 15)]
        left_shoulder_idx, right_shoulder_idx = 10, 13
    elif num_joints == 17:
        SKELETON = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
        left_shoulder_idx, right_shoulder_idx = 11, 14
    else:
        return img

    left_shoulder = pose_3d[left_shoulder_idx]
    right_shoulder = pose_3d[right_shoulder_idx]
    left_valid = not np.isnan(left_shoulder).any()
    right_valid = not np.isnan(right_shoulder).any()

    if left_valid and right_valid:
        center_point = (left_shoulder + right_shoulder) * 0.5
    elif left_valid:
        center_point = left_shoulder.copy()
    elif right_valid:
        center_point = right_shoulder.copy()
    elif not np.isnan(pose_3d[0]).any():
        center_point = pose_3d[0].copy()
    else:
        center_point = np.zeros(3, dtype=np.float32)

    centered = pose_3d - np.asarray(center_point, dtype=np.float32)
    valid_mask = ~np.isnan(centered).any(axis=1)
    if not valid_mask.any():
        return img

    R = view_rotation(angle_y, angle_x)
    center_x = vp_x + vp_w / 2.0
    center_y = vp_y + vp_h / 2.0
    pts_2d, depth = project_points(centered, R, zoom, FIXED_SCALE, center_x, center_y)
    
    cv2.rectangle(img, (vp_x, vp_y), (vp_x + vp_w, vp_y + vp_h), (24, 24, 28), -1)
    cv2.rectangle(img, (vp_x, vp_y), (vp_x + vp_w, vp_y + vp_h), (100, 100, 100), 2)
    
    bone_colors = [(255, 170, 0)]*3 + [(0, 200, 0)]*3 + [(0, 255, 255)]*3 + [(255, 0, 255)]*3 + [(0, 0, 255)]*3
    bone_order = sorted(range(len(SKELETON)), key=lambda b: depth[SKELETON[b][0]] + depth[SKELETON[b][1]])

    for bi in bone_order:
        i, j = SKELETON[bi]
        if not valid_mask[i] or not valid_mask[j]:
            continue
        pt1 = (int(pts_2d[i][0]), int(pts_2d[i][1]))
        pt2 = (int(pts_2d[j][0]), int(pts_2d[j][1]))
        cv2.line(img, pt1, pt2, bone_colors[bi % len(bone_colors)], 3, cv2.LINE_AA)

    for k in np.argsort(depth):
        if valid_mask[k]:
            cv2.circle(img, (int(pts_2d[k][0]), int(pts_2d[k][1])), 4, (255, 255, 255), -1, cv2.LINE_AA)
    return img

def compensate_shoulder_yaw(pose):
    pose = np.asarray(pose, dtype=np.float32).copy()
    left_shoulder, right_shoulder = pose[10], pose[13]
    shoulder_mid = (left_shoulder + right_shoulder) * 0.5
    dx, dz = left_shoulder[0] - right_shoulder[0], left_shoulder[2] - right_shoulder[2]
    if np.hypot(dx, dz) < 1e-6:
        return pose
    angle = np.arctan2(dz, dx)
    Ry = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32)
    return (pose - shoulder_mid) @ Ry.T + shoulder_mid

def mouse_callback(event, x, y, flags, state):
    """Updates the caller's local state dictionary directly."""
    if event == cv2.EVENT_LBUTTONDOWN:
        state['dragging'] = True
        state['last_mx'], state['last_my'] = x, y
    elif event == cv2.EVENT_MBUTTONDOWN:
        state['zooming'] = True
        state['last_mx'], state['last_my'] = x, y
    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_MBUTTONUP):
        state['dragging'] = False
        state['zooming'] = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if state['dragging'] or state['zooming']:
            dx, dy = x - state['last_mx'], y - state['last_my']
            state['last_mx'], state['last_my'] = x, y
            if state['dragging']:
                state['rot_y'] += dx * 0.5
                state['rot_x'] = float(np.clip(state['rot_x'] + dy * 0.5, -90, 90))
            elif state['zooming']:
                state['zoom_level'] = float(np.clip(state['zoom_level'] * (1.0 - dy * 0.01), 0.1, 5.0))
    elif event == cv2.EVENT_MOUSEWHEEL:
        delta = flags >> 16
        if delta > 32767: delta -= 65536
        if delta > 0:
            state['zoom_level'] = min(state['zoom_level'] * 1.15, 5.0)
        elif delta < 0:
            state['zoom_level'] = max(state['zoom_level'] / 1.15, 0.1)
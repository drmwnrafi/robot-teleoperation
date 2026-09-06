# utils/viewer.py
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from typing import List, Tuple, Optional

from utils.skeleton import SKELETON_FORMATS

class PoseViewer:
    def __init__(self, window_name: str, base_widths: List[int], base_heights: List[int], 
                 view_3d_base_width: int, confidence_threshold: float = 0.3, 
                 skeleton_format: str = "coco_wholebody_133"):
        
        self.window_name = window_name
        self.confidence_threshold = confidence_threshold
        
        if skeleton_format not in SKELETON_FORMATS:
            raise ValueError(f"Unknown skeleton format: {skeleton_format}. Choose from {list(SKELETON_FORMATS.keys())}")
        self.skeleton_config = SKELETON_FORMATS[skeleton_format]
        self.num_keypoints = self.skeleton_config["num_keypoints"]
        
        self.screen_width, self.screen_height = self._get_screen_resolution()
        self.scale, self.target_h, self.target_widths, self.target_heights = self._calculate_scaling(
            base_widths, base_heights, view_3d_base_width
        )
        
        self.yaw = 0.5
        self.pitch = 0.2
        self.zoom = 1.0  # Added zoom state
        
        self.is_dragging = False
        self.is_zooming = False
        self.last_x = 0
        self.last_y = 0
        self.last_zoom_y = 0
        
        sep_width = 5
        self.offset_x_3d = sum(self.target_widths[:-1]) + (len(self.target_widths) - 1) * sep_width
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _get_screen_resolution(self) -> Tuple[int, int]:
        try:
            root = tk.Tk()
            root.withdraw()
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            return w, h
        except Exception:
            return 1920, 1080

    def _calculate_scaling(self, base_widths, base_heights, view_3d_base_width) -> Tuple[float, int, List[int], List[int]]:
        sep_width = 5
        total_base_w = sum(base_widths) + view_3d_base_width + (len(base_widths) * sep_width)
        max_base_h = max(base_heights)
        
        scale = min((self.screen_width - 20) / total_base_w, (self.screen_height - 80) / max_base_h)
        
        target_h = int(max_base_h * scale)
        target_widths = [int(w * scale) for w in base_widths]
        target_widths.append(int(view_3d_base_width * scale))
        
        target_heights = [int(h * scale) for h in base_heights]
        target_heights.append(target_h)
        
        return scale, target_h, target_widths, target_heights

    def _mouse_callback(self, event, x, y, flags, param):
        view_w = self.target_widths[-1]
        if self.offset_x_3d <= x < self.offset_x_3d + view_w:
            # Left click: Rotate
            if event == cv2.EVENT_LBUTTONDOWN:
                self.is_dragging = True
                self.last_x, self.last_y = x, y
            elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
                self.yaw += (x - self.last_x) * 0.015   
                self.pitch += (y - self.last_y) * 0.015 
                self.last_x, self.last_y = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                self.is_dragging = False
            
            # Right click: Zoom
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.is_zooming = True
                self.last_zoom_y = y
            elif event == cv2.EVENT_MOUSEMOVE and self.is_zooming:
                delta_y = self.last_zoom_y - y
                self.zoom += delta_y * 0.01
                # Clamp zoom to reasonable bounds (10% to 500%)
                self.zoom = max(0.1, min(5.0, self.zoom))
                self.last_zoom_y = y
            elif event == cv2.EVENT_RBUTTONUP:
                self.is_zooming = False

    def reset_view(self):
        self.yaw, self.pitch = 0.5, 0.2
        self.zoom = 1.0

    def _get_rotation_matrix(self):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        return Rx @ Ry

    def _draw_pose_generic(self, output: np.ndarray, kp_map: dict, is_3d: bool = False):
        """Generic drawing function that works for ANY format in SKELETON_FORMATS."""
        scale_factor = 1.5 if is_3d else 1.0
        
        for part_name, part_config in self.skeleton_config["parts"].items():
            color = part_config["color"]
            radius = int(part_config["radius"] * scale_factor)
            thickness = int(part_config["thickness"] * scale_factor)
            
            for idx in part_config["indices"]:
                if idx in kp_map:
                    cv2.circle(output, kp_map[idx], radius, color, -1)
            
            for a, b in part_config["skeleton"]:
                if a in kp_map and b in kp_map:
                    cv2.line(output, kp_map[a], kp_map[b], color, thickness)

    def _draw_2d_frame(self, frame, packet, cam_id):
        output = frame.copy()
        if not packet:
            # Reduced font scale from 1.2 to 0.7
            cv2.putText(output, f"Cam {cam_id} (No Detection)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return output

        kp_map = {int(pid): (int(x), int(y)) for pid, (x, y), conf in zip(packet.keypoint_id, packet.img_loc, packet.confidence) 
                  if float(conf) >= self.confidence_threshold and 0 <= pid < self.num_keypoints}

        self._draw_pose_generic(output, kp_map, is_3d=False)
        # Reduced font scale from 1.2 to 0.7
        cv2.putText(output, f"Cam {cam_id}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return output

    def _render_3d_view(self, xyz_df, w, h):
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 20 
        step = max(20, w // 20)
        for i in range(0, w, step): cv2.line(canvas, (i, 0), (i, h), (40, 40, 40), 1)
        for i in range(0, h, step): cv2.line(canvas, (0, i), (w, i), (40, 40, 40), 1)
            
        if xyz_df is None or getattr(xyz_df, 'empty', True):
            cv2.putText(canvas, "No 3D Data", (w//2 - 70, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return canvas
            
        pts = xyz_df[['x_coord', 'y_coord', 'z_coord']].values
        kp_ids = xyz_df['keypoint_id'].values.astype(int)
        
        min_pt, max_pt = np.min(pts, axis=0), np.max(pts, axis=0)
        center = (min_pt + max_pt) / 2
        extent = np.max(max_pt - min_pt) if np.max(max_pt - min_pt) > 0 else 1.0
        pts_scaled = (pts - center) / extent * 1.5 
        
        pts_rot = (self._get_rotation_matrix() @ pts_scaled.T).T
        
        focal = 2.5
        z = np.maximum(pts_rot[:, 2] + focal, 0.1)
        
        norm_x = pts_rot[:, 0] / z
        norm_y = pts_rot[:, 1] / z
        
        proj_x = (norm_x * self.zoom) * (w / 2) + w / 2
        proj_y = (norm_y * self.zoom) * (h / 2) + h / 2
        
        kp_map = {int(kp): (int(x), int(y)) for kp, x, y in zip(kp_ids, proj_x, proj_y)}

        self._draw_pose_generic(canvas, kp_map, is_3d=True)

        cv2.putText(canvas, "3D View (Left Drag: Rotate | Right Drag: Zoom)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, "Press 'R' to Reset View", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return canvas

    def show(self, raw_frames: List[np.ndarray], packets: List, xyz_df: Optional[pd.DataFrame], frame_index: int) -> bool:
        drawn_2d_frames = [self._draw_2d_frame(f, p, i) for i, (f, p) in enumerate(zip(raw_frames, packets))]
        
        resized_2d = []
        for f, w, h in zip(drawn_2d_frames, self.target_widths[:-1], self.target_heights[:-1]):
            resized = cv2.resize(f, (w, h))
            pad_top = (self.target_h - h) // 2
            pad_bottom = self.target_h - h - pad_top
            padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[20, 20, 20])
            resized_2d.append(padded)
            
        view_3d = self._render_3d_view(xyz_df, self.target_widths[-1], self.target_h)
        
        sep = np.ones((self.target_h, 5, 3), dtype=np.uint8) * 255
        
        combined_parts = []
        for frame in resized_2d:
            combined_parts.append(frame)
            combined_parts.append(sep)
        combined_parts.append(view_3d)
        
        combined = cv2.hconcat(combined_parts)
        
        pts_count = len(xyz_df) if xyz_df is not None and not getattr(xyz_df, 'empty', True) else 0
        cv2.putText(combined, f"Frame: {frame_index} | 3D Pts: {pts_count}", 
                    (20, combined.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 2)
        
        cv2.resizeWindow(self.window_name, combined.shape[1], combined.shape[0])
        cv2.imshow(self.window_name, combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: return False
        elif key in (ord('r'), ord('R')): self.reset_view()
            
        return True

    def destroy(self):
        cv2.destroyAllWindows()